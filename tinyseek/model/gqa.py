from __future__ import annotations

import torch
from torch import Tensor, nn

from tinyseek.model.rope import RotaryEmbedding


class GroupedQueryAttention(nn.Module):
    """Grouped Query Attention with causal masking and RoPE.

    Input shape:
        [batch_size, sequence_length, hidden_size]

    Query shape:
        [B, num_heads, T, head_dim]

    Key / Value shape before expansion:
        [B, num_kv_heads, T, head_dim]

    Attention weights:
        [B, num_heads, T, T]
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        bias: bool = False,
    ) -> None:
        super().__init__()

        if hidden_size <= 0:
            raise ValueError(
                f"hidden_size must be positive, got {hidden_size}"
            )

        if num_heads <= 0:
            raise ValueError(
                f"num_heads must be positive, got {num_heads}"
            )

        if num_kv_heads <= 0:
            raise ValueError(
                f"num_kv_heads must be positive, got {num_kv_heads}"
            )

        if hidden_size % num_heads != 0:
            raise ValueError(
                "hidden_size must be divisible by num_heads: "
                f"got hidden_size={hidden_size}, "
                f"num_heads={num_heads}"
            )

        if num_heads % num_kv_heads != 0:
            raise ValueError(
                "num_heads must be divisible by num_kv_heads: "
                f"got num_heads={num_heads}, "
                f"num_kv_heads={num_kv_heads}"
            )

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads

        self.head_dim = hidden_size // num_heads

        if self.head_dim % 2 != 0:
            raise ValueError(
                "head_dim must be even for RoPE: "
                f"got head_dim={self.head_dim}"
            )

        # 每个 KV Head 要被多少个 Query Head 共享。
        #
        # 例如：
        #
        # num_heads = 8
        # num_kv_heads = 2
        #
        # 则：
        #
        # Q0 Q1 Q2 Q3 -> KV0
        # Q4 Q5 Q6 Q7 -> KV1
        self.num_queries_per_kv = (
            num_heads // num_kv_heads
        )

        # Attention 缩放使用每个头的实际维度。
        self.scale = self.head_dim ** -0.5

        # --------------------------------------------------
        # Q projection
        #
        # Query 仍然保留所有 Query Heads。
        #
        # D -> num_heads * head_dim
        #
        # 由于：
        #
        # num_heads * head_dim == hidden_size
        #
        # 所以实际上是：
        #
        # D -> D
        # --------------------------------------------------

        self.q_proj = nn.Linear(
            hidden_size,
            num_heads * self.head_dim,
            bias=bias,
        )

        # --------------------------------------------------
        # K / V projection
        #
        # 注意这里是真正减少维度的地方。
        #
        # MHA:
        #     D -> num_heads * head_dim = D
        #
        # GQA:
        #     D -> num_kv_heads * head_dim
        #
        # 例如：
        #
        # D = 16
        # num_heads = 4
        # num_kv_heads = 2
        # head_dim = 4
        #
        # Q: 16 -> 16
        # K: 16 -> 8
        # V: 16 -> 8
        # --------------------------------------------------

        kv_size = num_kv_heads * self.head_dim

        self.k_proj = nn.Linear(
            hidden_size,
            kv_size,
            bias=bias,
        )

        self.v_proj = nn.Linear(
            hidden_size,
            kv_size,
            bias=bias,
        )

        # 所有 Query Heads 的结果最终还是合成 hidden_size。
        self.out_proj = nn.Linear(
            hidden_size,
            hidden_size,
            bias=bias,
        )

        # RoPE 是在每个 Head 内部执行的。
        self.rope = RotaryEmbedding(
            dim=self.head_dim,
        )

    def _split_query_heads(
        self,
        x: Tensor,
    ) -> Tensor:
        """Split Query into num_heads.

        Input:
            [B, T, num_heads * head_dim]

        Output:
            [B, num_heads, T, head_dim]
        """

        batch_size, sequence_length, _ = x.shape

        x = x.view(
            batch_size,
            sequence_length,
            self.num_heads,
            self.head_dim,
        )

        return x.transpose(1, 2)

    def _split_kv_heads(
        self,
        x: Tensor,
    ) -> Tensor:
        """Split Key / Value into num_kv_heads.

        Input:
            [B, T, num_kv_heads * head_dim]

        Output:
            [B, num_kv_heads, T, head_dim]
        """

        batch_size, sequence_length, _ = x.shape

        x = x.view(
            batch_size,
            sequence_length,
            self.num_kv_heads,
            self.head_dim,
        )

        return x.transpose(1, 2)

    def _repeat_kv(
        self,
        x: Tensor,
    ) -> Tensor:
        """Map KV Heads to Query Heads.

        Input:
            [B, num_kv_heads, T, head_dim]

        Output:
            [B, num_heads, T, head_dim]

        Example:

            KV0 KV1

        with:

            num_heads = 4
            num_kv_heads = 2

        becomes logically:

            KV0 KV0 KV1 KV1
        """

        if self.num_queries_per_kv == 1:
            return x

        return x.repeat_interleave(
            self.num_queries_per_kv,
            dim=1,
        )

    def _merge_query_heads(
        self,
        x: Tensor,
    ) -> Tensor:
        """Merge Query Heads back to hidden_size.

        Input:
            [B, num_heads, T, head_dim]

        Output:
            [B, T, hidden_size]
        """

        batch_size, _, sequence_length, _ = x.shape

        x = x.transpose(1, 2)

        x = x.contiguous()

        return x.view(
            batch_size,
            sequence_length,
            self.hidden_size,
        )

    def forward(
        self,
        x: Tensor,
        positions: Tensor | None = None,
        return_attention: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:

        if x.ndim != 3:
            raise ValueError(
                "input must have shape "
                "[batch_size, sequence_length, hidden_size]"
            )

        if not x.is_floating_point():
            raise TypeError(
                f"input must be floating point, got {x.dtype}"
            )

        if x.shape[-1] != self.hidden_size:
            raise ValueError(
                "The final input dimension must equal hidden_size: "
                f"expected {self.hidden_size}, "
                f"got {x.shape[-1]}"
            )

        _, sequence_length, _ = x.shape

        # ==================================================
        # 1. 生成 Q / K / V
        # ==================================================

        query = self.q_proj(x)

        key = self.k_proj(x)

        value = self.v_proj(x)

        # 假设：
        #
        # B = 2
        # T = 5
        # hidden_size = 16
        # num_heads = 4
        # num_kv_heads = 2
        # head_dim = 4
        #
        # 那么：
        #
        # query : [2, 5, 16]
        # key   : [2, 5, 8]
        # value : [2, 5, 8]

        # ==================================================
        # 2. 拆头
        # ==================================================

        query = self._split_query_heads(query)

        key = self._split_kv_heads(key)

        value = self._split_kv_heads(value)

        # 现在：
        #
        # query:
        # [B, 4, T, 4]
        #
        # key/value:
        # [B, 2, T, 4]

        # ==================================================
        # 3. 给 Q 和 K 加 RoPE
        # ==================================================

        query = self.rope(
            query,
            positions=positions,
        )

        key = self.rope(
            key,
            positions=positions,
        )

        # ==================================================
        # 4. 将少量 KV Heads 映射给多个 Query Heads
        # ==================================================

        key_for_attention = self._repeat_kv(key)

        value_for_attention = self._repeat_kv(value)

        # 例如：
        #
        # key 原本：
        #
        # [K0, K1]
        #
        # 逻辑上扩展为：
        #
        # [K0, K0, K1, K1]
        #
        # 这样就可以分别和：
        #
        # [Q0, Q1, Q2, Q3]
        #
        # 做 Attention。

        # ==================================================
        # 5. QK^T
        # ==================================================

        scores = torch.matmul(
            query,
            key_for_attention.transpose(-2, -1),
        )

        # query:
        # [B, H, T, Dh]
        #
        # key^T:
        # [B, H, Dh, T]
        #
        # scores:
        # [B, H, T, T]

        scores = scores * self.scale

        # ==================================================
        # 6. Causal Mask
        # ==================================================

        causal_mask = torch.triu(
            torch.ones(
                sequence_length,
                sequence_length,
                device=x.device,
                dtype=torch.bool,
            ),
            diagonal=1,
        )

        scores = scores.masked_fill(
            causal_mask,
            float("-inf"),
        )

        # ==================================================
        # 7. Softmax
        # ==================================================

        attention_weights = torch.softmax(
            scores,
            dim=-1,
        )

        # ==================================================
        # 8. Attention weights × V
        # ==================================================

        context = torch.matmul(
            attention_weights,
            value_for_attention,
        )

        # [B, H, T, T]
        # @
        # [B, H, T, Dh]
        #
        # ->
        #
        # [B, H, T, Dh]

        # ==================================================
        # 9. 合并所有 Query Heads
        # ==================================================

        context = self._merge_query_heads(
            context
        )

        # [B, H, T, Dh]
        #
        # ->
        #
        # [B, T, hidden_size]

        # ==================================================
        # 10. Output Projection
        # ==================================================

        output = self.out_proj(context)

        if return_attention:
            return output, attention_weights

        return output

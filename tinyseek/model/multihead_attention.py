from __future__ import annotations

import torch
from torch import Tensor, nn

from tinyseek.model.rope import RotaryEmbedding


class MultiHeadCausalSelfAttention(nn.Module):
    """Multi-head causal self-attention.

    Expected input shape:

        [batch_size, sequence_length, hidden_size]

    Attention weights shape:

        [batch_size, num_heads, sequence_length, sequence_length]
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
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

        if hidden_size % num_heads != 0:
            raise ValueError(
                "hidden_size must be divisible by num_heads: "
                f"got hidden_size={hidden_size}, "
                f"num_heads={num_heads}"
            )

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        # 当前 RoPE 实现将每两个特征组成一个二维旋转平面，
        # 所以每个头的维度必须为偶数。
        if self.head_dim % 2 != 0:
            raise ValueError(
                "head_dim must be even for RoPE: "
                f"got head_dim={self.head_dim}"
            )

        # 每个头实际参与点积的维度是 head_dim，
        # 因此缩放系数是 1 / sqrt(head_dim)。
        self.scale = self.head_dim ** -0.5

        # 一次性生成所有头的 Q、K、V。
        #
        # 输入：[B, T, hidden_size]
        # 输出：[B, T, hidden_size]
        self.q_proj = nn.Linear(
            hidden_size,
            hidden_size,
            bias=bias,
        )

        self.k_proj = nn.Linear(
            hidden_size,
            hidden_size,
            bias=bias,
        )

        self.v_proj = nn.Linear(
            hidden_size,
            hidden_size,
            bias=bias,
        )

        # 合并所有头后，再重新混合不同头的特征。
        self.out_proj = nn.Linear(
            hidden_size,
            hidden_size,
            bias=bias,
        )

        # RoPE 在每个头内部执行，所以 dim=head_dim。
        self.rope = RotaryEmbedding(
            dim=self.head_dim,
        )

    def _split_heads(self, x: Tensor) -> Tensor:
        """Split the hidden dimension into multiple heads.

        Input:
            [B, T, hidden_size]

        Output:
            [B, num_heads, T, head_dim]
        """

        batch_size, sequence_length, _ = x.shape

        # [B, T, hidden_size]
        # → [B, T, num_heads, head_dim]
        x = x.view(
            batch_size,
            sequence_length,
            self.num_heads,
            self.head_dim,
        )

        # [B, T, H, Dh]
        # → [B, H, T, Dh]
        return x.transpose(1, 2)

    def _merge_heads(self, x: Tensor) -> Tensor:
        """Merge multiple heads back into the hidden dimension.

        Input:
            [B, num_heads, T, head_dim]

        Output:
            [B, T, hidden_size]
        """

        batch_size, _, sequence_length, _ = x.shape

        # [B, H, T, Dh]
        # → [B, T, H, Dh]
        x = x.transpose(1, 2)

        # transpose 后内存排列通常不连续。
        # contiguous() 创建连续排列，便于随后 view。
        x = x.contiguous()

        # [B, T, H, Dh]
        # → [B, T, H * Dh]
        # → [B, T, hidden_size]
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
                f"expected {self.hidden_size}, got {x.shape[-1]}"
            )

        batch_size, sequence_length, _ = x.shape

        # 先对原始隐藏状态做投影。
        #
        # [B, T, D] → [B, T, D]
        query = self.q_proj(x)
        key = self.k_proj(x)
        value = self.v_proj(x)

        # 将隐藏维拆成多个头。
        #
        # [B, T, D] → [B, H, T, Dh]
        query = self._split_heads(query)
        key = self._split_heads(key)
        value = self._split_heads(value)

        # RoPE 的输入要求：
        # [..., sequence_length, head_dim]
        #
        # 当前形状 [B, H, T, Dh] 正好符合。
        query = self.rope(
            query,
            positions=positions,
        )

        key = self.rope(
            key,
            positions=positions,
        )

        # key.transpose(-2, -1):
        #
        # [B, H, T, Dh]
        # → [B, H, Dh, T]
        #
        # scores:
        #
        # [B, H, T, Dh] @ [B, H, Dh, T]
        # → [B, H, T, T]
        scores = torch.matmul(
            query,
            key.transpose(-2, -1),
        )

        scores = scores * self.scale

        # True 表示未来位置，需要屏蔽。
        #
        # T=4 时：
        #
        # False True  True  True
        # False False True  True
        # False False False True
        # False False False False
        causal_mask = torch.triu(
            torch.ones(
                sequence_length,
                sequence_length,
                device=x.device,
                dtype=torch.bool,
            ),
            diagonal=1,
        )

        # mask 形状为 [T,T]，
        # 会广播到 [B,H,T,T]。
        scores = scores.masked_fill(
            causal_mask,
            float("-inf"),
        )

        # 每个 batch、每个 head、每个 Query，
        # 对所有 Key 的权重之和为 1。
        attention_weights = torch.softmax(
            scores,
            dim=-1,
        )

        # [B, H, T, T] @ [B, H, T, Dh]
        # → [B, H, T, Dh]
        context = torch.matmul(
            attention_weights,
            value,
        )

        # [B, H, T, Dh]
        # → [B, T, D]
        context = self._merge_heads(context)

        # 融合不同头的信息，并返回主隐藏空间。
        output = self.out_proj(context)

        if return_attention:
            return output, attention_weights

        return output

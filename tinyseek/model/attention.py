from __future__ import annotations

import torch
from torch import Tensor, nn

from tinyseek.model.rope import RotaryEmbedding


class CausalSelfAttention(nn.Module):
    """Single-head causal self-attention.

    Expected input shape:

        [batch_size, sequence_length, hidden_size]
    """

    def __init__(
        self,
        hidden_size: int,
        bias: bool = False,
    ) -> None:
        super().__init__()

        if hidden_size <= 0:
            raise ValueError(
                f"hidden_size must be positive, got {hidden_size}"
            )

        # 当前单头版本中：
        # head_dim == hidden_size
        #
        # RoPE 要求每两个维度组成一个二维旋转平面，
        # 因此 hidden_size 必须为偶数。
        if hidden_size % 2 != 0:
            raise ValueError(
                f"hidden_size must be even, got {hidden_size}"
            )

        self.hidden_size = hidden_size
        self.scale = hidden_size ** -0.5

        # 同一个输入 x 经过三套不同参数，分别得到 Q、K、V。
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

        # 把 Attention 聚合后的信息再投影回隐藏空间。
        self.out_proj = nn.Linear(
            hidden_size,
            hidden_size,
            bias=bias,
        )

        self.rope = RotaryEmbedding(
            dim=hidden_size,
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

        sequence_length = x.shape[1]

        # [B, T, D] → [B, T, D]
        query = self.q_proj(x)
        key = self.k_proj(x)
        value = self.v_proj(x)

        # 只对 Q 和 K 加入位置信息。
        query = self.rope(
            query,
            positions=positions,
        )

        key = self.rope(
            key,
            positions=positions,
        )

        # key.transpose(-2, -1):
        # [B, T, D] → [B, D, T]
        #
        # scores:
        # [B, T, D] @ [B, D, T]
        # → [B, T, T]
        scores = torch.matmul(
            query,
            key.transpose(-2, -1),
        )

        scores = scores * self.scale

        # 上三角区域代表未来 token。
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

        # [T, T] 会自动广播到 [B, T, T]。
        scores = scores.masked_fill(
            causal_mask,
            float("-inf"),
        )

        # 对最后一维做 softmax：
        # 每个 Query 对所有 Key 的权重之和为 1。
        attention_weights = torch.softmax(
            scores,
            dim=-1,
        )

        # [B, T, T] @ [B, T, D]
        # → [B, T, D]
        context = torch.matmul(
            attention_weights,
            value,
        )

        output = self.out_proj(context)

        if return_attention:
            return output, attention_weights

        return output

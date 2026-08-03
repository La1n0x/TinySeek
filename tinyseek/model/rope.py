from __future__ import annotations

import torch
from torch import Tensor, nn


class RotaryEmbedding(nn.Module):
    def __init__(
        self,
        dim: int,
        base: float = 10000.0,
    ) -> None:
        super().__init__()

        if dim <= 0:
            raise ValueError(
                f"dim must be positive, got {dim}"
            )

        if dim % 2 != 0:
            raise ValueError(
                f"dim must be even, got {dim}"
            )

        if base <= 0:
            raise ValueError(
                f"base must be positive, got {base}"
            )

        self.dim = dim
        self.base = base

        dimension_indices = torch.arange(
            0,
            dim,
            2,
            dtype=torch.float32,
        )

        # 计算每个二维特征对的旋转频率
        inv_freq = 1.0 / (
            base ** (dimension_indices / dim)
        )

        self.register_buffer(
            "inv_freq",
            inv_freq,
            persistent=False,
        )

    def forward(
        self,
        x: Tensor,
        positions: Tensor | None = None,
    ) -> Tensor:
        if x.ndim < 2:
            raise ValueError(
                "input must have at least two dimensions: "
                "[..., sequence_length, head_dim]"
            )

        if not x.is_floating_point():
            raise TypeError(
                f"input must be floating point, got {x.dtype}"
            )

        if x.shape[-1] != self.dim:
            raise ValueError(
                "The final input dimension must equal dim: "
                f"expected {self.dim}, got {x.shape[-1]}"
            )

        sequence_length = x.shape[-2]

        if positions is None:
            positions = torch.arange(
                sequence_length,
                device=x.device,
            )

        if positions.ndim != 1:
            raise ValueError(
                "positions must be a one-dimensional tensor"
            )

        if positions.shape[0] != sequence_length:
            raise ValueError(
                "positions length must equal sequence length: "
                f"expected {sequence_length}, "
                f"got {positions.shape[0]}"
            )

        input_dtype = x.dtype

        if input_dtype in (
            torch.float16,
            torch.bfloat16,
        ):
            compute_dtype = torch.float32
        else:
            compute_dtype = input_dtype

        # 你原来的代码漏掉了这一步
        x_compute = x.to(dtype=compute_dtype)

        positions_compute = positions.to(
            device=x.device,
            dtype=compute_dtype,
        )

        inv_freq = self.inv_freq.to(
            device=x.device,
            dtype=compute_dtype,
        )

        # [T] 外积 [D/2]，得到 [T, D/2]
        freqs = torch.outer(
            positions_compute,
            inv_freq,
        )

        broadcast_shape = (
            [1] * (x.ndim - 2)
            + [sequence_length, self.dim // 2]
        )

        cos = freqs.cos().view(broadcast_shape)
        sin = freqs.sin().view(broadcast_shape)

        x_even = x_compute[..., 0::2]
        x_odd = x_compute[..., 1::2]

        # 二维旋转公式
        rotated_even = (
            x_even * cos
            - x_odd * sin
        )

        rotated_odd = (
            x_even * sin
            + x_odd * cos
        )

        output = torch.stack(
            [rotated_even, rotated_odd],
            dim=-1,
        ).flatten(start_dim=-2)

        return output.to(dtype=input_dtype)

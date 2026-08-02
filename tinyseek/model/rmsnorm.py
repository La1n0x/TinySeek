from __future__ import annotations

import torch
from torch import Tensor, nn


class RMSNorm(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()

        if hidden_size <= 0:
            raise ValueError(
                f"hidden_size must be positive, got {hidden_size}"
            )

        if eps <= 0:
            raise ValueError(
                f"eps must be positive, got {eps}"
            )

        self.hidden_size = hidden_size
        self.eps = eps

        self.weight = nn.Parameter(
            torch.ones(hidden_size)
        )

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim == 0:
            raise ValueError(
                "input must have at least one dimension"
            )

        if not x.is_floating_point():
            raise TypeError(
                f"input must be floating point, got {x.dtype}"
            )

        if x.shape[-1] != self.hidden_size:
            raise ValueError(
                "The last dimension must equal hidden_size: "
                f"expected {self.hidden_size}, got {x.shape[-1]}"
            )

        input_dtype = x.dtype

        # 使用 float32 计算归一化统计量。
        x_float = x.float()

        mean_square = x_float.pow(2).mean(
            dim=-1,
            keepdim=True,
        )

        inverse_rms = torch.rsqrt(
            mean_square + self.eps
        )

        normalized = x_float * inverse_rms

        output = normalized * self.weight.float()

        return output.to(dtype=input_dtype)

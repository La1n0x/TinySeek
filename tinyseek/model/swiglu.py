from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class SwiGLU(nn.Module):
    """SwiGLU feed-forward network.

    Input:
        [B, T, hidden_size]

    Intermediate:
        [B, T, intermediate_size]

    Output:
        [B, T, hidden_size]
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        bias: bool = False,
    ) -> None:
        super().__init__()

        if hidden_size <= 0:
            raise ValueError(
                f"hidden_size must be positive, got {hidden_size}"
            )

        if intermediate_size <= 0:
            raise ValueError(
                "intermediate_size must be positive, "
                f"got {intermediate_size}"
            )

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size

        # Gate branch:
        #
        # [B, T, hidden_size]
        # ->
        # [B, T, intermediate_size]
        self.gate_proj = nn.Linear(
            hidden_size,
            intermediate_size,
            bias=bias,
        )

        # Up branch:
        #
        # [B, T, hidden_size]
        # ->
        # [B, T, intermediate_size]
        self.up_proj = nn.Linear(
            hidden_size,
            intermediate_size,
            bias=bias,
        )

        # Project the gated intermediate representation
        # back to hidden_size.
        #
        # [B, T, intermediate_size]
        # ->
        # [B, T, hidden_size]
        self.down_proj = nn.Linear(
            intermediate_size,
            hidden_size,
            bias=bias,
        )

    def forward(
        self,
        x: Tensor,
    ) -> Tensor:

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

        # ==================================================
        # 1. Gate branch
        # ==================================================

        gate = self.gate_proj(x)

        # [B, T, hidden_size]
        # ->
        # [B, T, intermediate_size]

        # ==================================================
        # 2. Up branch
        # ==================================================

        up = self.up_proj(x)

        # [B, T, hidden_size]
        # ->
        # [B, T, intermediate_size]

        # ==================================================
        # 3. SwiGLU gating
        # ==================================================

        hidden = F.silu(gate) * up

        # Both tensors:
        #
        # [B, T, intermediate_size]
        #
        # "*" is element-wise multiplication.
        #
        # Output:
        #
        # [B, T, intermediate_size]

        # ==================================================
        # 4. Project back to hidden_size
        # ==================================================

        output = self.down_proj(hidden)

        # [B, T, intermediate_size]
        # ->
        # [B, T, hidden_size]

        return output

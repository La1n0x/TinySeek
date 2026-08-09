from __future__ import annotations

from torch import Tensor, nn

from tinyseek.model.rmsnorm import RMSNorm
from tinyseek.model.gqa import GroupedQueryAttention
from tinyseek.model.swiglu import SwiGLU


class DecoderBlock(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        intermediate_size: int,
        eps: float = 1e-6,
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

        if eps <= 0:
            raise ValueError(
                f"eps must be positive, got {eps}"
            )

        self.hidden_size = hidden_size

        # ==================================================
        # Attention branch
        # ==================================================

        self.attn_norm = RMSNorm(
            hidden_size=hidden_size,
            eps=eps,
        )

        self.attention = GroupedQueryAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            bias=bias,
        )

        # ==================================================
        # MLP branch
        # ==================================================

        self.mlp_norm = RMSNorm(
            hidden_size=hidden_size,
            eps=eps,
        )

        self.mlp = SwiGLU(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            bias=bias,
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

        # ==================================================
        # 1. Attention sub-layer
        # ==================================================

        residual = x

        hidden = self.attn_norm(x)

        if return_attention:
            attention_output, attention_weights = self.attention(
                hidden,
                positions=positions,
                return_attention=True,
            )
        else:
            attention_output = self.attention(
                hidden,
                positions=positions,
            )

        x = residual + attention_output

        # ==================================================
        # 2. SwiGLU sub-layer
        # ==================================================

        residual = x

        hidden = self.mlp_norm(x)

        mlp_output = self.mlp(hidden)

        x = residual + mlp_output

        # ==================================================
        # 3. Output
        # ==================================================

        if return_attention:
            return x, attention_weights

        return x

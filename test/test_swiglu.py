import pytest
import torch
import torch.nn.functional as F

from tinyseek.model.swiglu import SwiGLU


def test_output_shape():
    model = SwiGLU(
        hidden_size=16,
        intermediate_size=32,
    )

    x = torch.randn(2, 5, 16)

    y = model(x)

    assert y.shape == (2, 5, 16)


def test_projection_shapes():
    model = SwiGLU(
        hidden_size=16,
        intermediate_size=32,
    )

    assert model.gate_proj.weight.shape == (32, 16)
    assert model.up_proj.weight.shape == (32, 16)
    assert model.down_proj.weight.shape == (16, 32)


def test_intermediate_shapes():
    model = SwiGLU(
        hidden_size=16,
        intermediate_size=32,
    )

    x = torch.randn(2, 5, 16)

    gate = model.gate_proj(x)
    up = model.up_proj(x)

    assert gate.shape == (2, 5, 32)
    assert up.shape == (2, 5, 32)

    hidden = F.silu(gate) * up

    assert hidden.shape == (2, 5, 32)


def test_matches_manual_computation():
    torch.manual_seed(0)

    model = SwiGLU(
        hidden_size=8,
        intermediate_size=16,
        bias=False,
    )

    x = torch.randn(2, 4, 8)

    actual = model(x)

    gate = model.gate_proj(x)
    up = model.up_proj(x)

    hidden = F.silu(gate) * up

    expected = model.down_proj(hidden)

    assert torch.allclose(
        actual,
        expected,
        atol=1e-6,
        rtol=1e-6,
    )


def test_gate_and_up_are_independent_parameters():
    model = SwiGLU(
        hidden_size=16,
        intermediate_size=32,
        bias=False,
    )

    assert (
        model.gate_proj.weight.data_ptr()
        != model.up_proj.weight.data_ptr()
    )


def test_parameter_count_without_bias():
    model = SwiGLU(
        hidden_size=16,
        intermediate_size=32,
        bias=False,
    )

    num_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    expected = (
        16 * 32
        + 16 * 32
        + 32 * 16
    )

    assert num_parameters == expected
    assert num_parameters == 1536


def test_bias_false():
    model = SwiGLU(
        hidden_size=16,
        intermediate_size=32,
        bias=False,
    )

    assert model.gate_proj.bias is None
    assert model.up_proj.bias is None
    assert model.down_proj.bias is None


def test_bias_true():
    model = SwiGLU(
        hidden_size=16,
        intermediate_size=32,
        bias=True,
    )

    assert model.gate_proj.bias is not None
    assert model.up_proj.bias is not None
    assert model.down_proj.bias is not None

    assert model.gate_proj.bias.shape == (32,)
    assert model.up_proj.bias.shape == (32,)
    assert model.down_proj.bias.shape == (16,)


def test_backward():
    torch.manual_seed(0)

    model = SwiGLU(
        hidden_size=16,
        intermediate_size=32,
    )

    x = torch.randn(
        2,
        5,
        16,
        requires_grad=True,
    )

    y = model(x)

    loss = y.square().mean()
    loss.backward()

    assert x.grad is not None

    assert model.gate_proj.weight.grad is not None
    assert model.up_proj.weight.grad is not None
    assert model.down_proj.weight.grad is not None


def test_gradient_shapes():
    model = SwiGLU(
        hidden_size=16,
        intermediate_size=32,
    )

    x = torch.randn(
        2,
        5,
        16,
        requires_grad=True,
    )

    model(x).sum().backward()

    assert model.gate_proj.weight.grad.shape == (32, 16)
    assert model.up_proj.weight.grad.shape == (32, 16)
    assert model.down_proj.weight.grad.shape == (16, 32)


def test_different_tokens_are_processed_independently():
    """
    SwiGLU itself does not mix sequence positions.

    Changing a future token must not change another token's
    output because every Linear acts only on the last dimension.
    """

    torch.manual_seed(0)

    model = SwiGLU(
        hidden_size=16,
        intermediate_size=32,
    )

    x1 = torch.randn(1, 5, 16)
    x2 = x1.clone()

    # Only modify token 4.
    x2[:, 4, :] += 100.0

    y1 = model(x1)
    y2 = model(x2)

    # Tokens 0~3 must remain exactly unchanged.
    assert torch.allclose(
        y1[:, :4, :],
        y2[:, :4, :],
        atol=1e-6,
        rtol=1e-6,
    )

    # The modified token should normally change.
    assert not torch.allclose(
        y1[:, 4, :],
        y2[:, 4, :],
    )


@pytest.mark.parametrize(
    "hidden_size, intermediate_size",
    [
        (8, 16),
        (16, 32),
        (32, 64),
        (64, 128),
    ],
)
def test_multiple_sizes(
    hidden_size,
    intermediate_size,
):
    model = SwiGLU(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
    )

    x = torch.randn(
        2,
        3,
        hidden_size,
    )

    y = model(x)

    assert y.shape == (
        2,
        3,
        hidden_size,
    )


@pytest.mark.parametrize(
    "hidden_size",
    [
        0,
        -1,
        -16,
    ],
)
def test_invalid_hidden_size(hidden_size):
    with pytest.raises(ValueError):
        SwiGLU(
            hidden_size=hidden_size,
            intermediate_size=32,
        )


@pytest.mark.parametrize(
    "intermediate_size",
    [
        0,
        -1,
        -32,
    ],
)
def test_invalid_intermediate_size(intermediate_size):
    with pytest.raises(ValueError):
        SwiGLU(
            hidden_size=16,
            intermediate_size=intermediate_size,
        )


def test_invalid_input_rank():
    model = SwiGLU(
        hidden_size=16,
        intermediate_size=32,
    )

    x = torch.randn(2, 16)

    with pytest.raises(ValueError):
        model(x)


def test_invalid_hidden_dimension():
    model = SwiGLU(
        hidden_size=16,
        intermediate_size=32,
    )

    x = torch.randn(2, 5, 15)

    with pytest.raises(ValueError):
        model(x)


def test_integer_input_rejected():
    model = SwiGLU(
        hidden_size=16,
        intermediate_size=32,
    )

    x = torch.ones(
        2,
        5,
        16,
        dtype=torch.int64,
    )

    with pytest.raises(TypeError):
        model(x)

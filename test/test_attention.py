import pytest
import torch

from tinyseek.model.attention import CausalSelfAttention


def test_output_shape_and_dtype() -> None:
    """Attention 不应改变输入形状和数据类型。"""

    torch.manual_seed(0)

    model = CausalSelfAttention(hidden_size=8)
    x = torch.randn(2, 5, 8)

    output = model(x)

    assert isinstance(output, torch.Tensor)
    assert output.shape == x.shape
    assert output.dtype == x.dtype


def test_return_attention_shape() -> None:
    """开启 return_attention 后，应同时返回输出和注意力矩阵。"""

    torch.manual_seed(1)

    model = CausalSelfAttention(hidden_size=8)
    x = torch.randn(2, 5, 8)

    output, attention_weights = model(
        x,
        return_attention=True,
    )

    assert output.shape == (2, 5, 8)
    assert attention_weights.shape == (2, 5, 5)


def test_attention_rows_sum_to_one() -> None:
    """每个 Query 对所有合法 Key 的权重之和应为1。"""

    torch.manual_seed(2)

    model = CausalSelfAttention(hidden_size=8)
    x = torch.randn(3, 6, 8)

    _, attention_weights = model(
        x,
        return_attention=True,
    )

    row_sums = attention_weights.sum(dim=-1)

    torch.testing.assert_close(
        row_sums,
        torch.ones_like(row_sums),
        rtol=1e-6,
        atol=1e-6,
    )


def test_future_attention_weights_are_zero() -> None:
    """因果 Mask 应使所有未来位置的注意力权重为0。"""

    torch.manual_seed(3)

    sequence_length = 6

    model = CausalSelfAttention(hidden_size=8)
    x = torch.randn(2, sequence_length, 8)

    _, attention_weights = model(
        x,
        return_attention=True,
    )

    future_mask = torch.triu(
        torch.ones(
            sequence_length,
            sequence_length,
            dtype=torch.bool,
        ),
        diagonal=1,
    )

    future_weights = attention_weights[:, future_mask]

    torch.testing.assert_close(
        future_weights,
        torch.zeros_like(future_weights),
        rtol=0.0,
        atol=0.0,
    )


def test_first_token_only_attends_to_itself() -> None:
    """第0个 token 没有过去，只能关注自己。"""

    torch.manual_seed(4)

    model = CausalSelfAttention(hidden_size=8)
    x = torch.randn(2, 5, 8)

    _, attention_weights = model(
        x,
        return_attention=True,
    )

    expected = torch.tensor(
        [1.0, 0.0, 0.0, 0.0, 0.0],
        dtype=attention_weights.dtype,
    )

    expected = expected.expand(2, -1)

    torch.testing.assert_close(
        attention_weights[:, 0, :],
        expected,
        rtol=0.0,
        atol=0.0,
    )


def test_future_tokens_do_not_affect_past_outputs() -> None:
    """改变未来 token，不应影响过去位置的输出。

    这里修改位置3、4，只比较位置0、1、2的输出。
    """

    torch.manual_seed(5)

    model = CausalSelfAttention(hidden_size=8)
    model.eval()

    x_original = torch.randn(1, 5, 8)
    x_modified = x_original.clone()

    # 大幅修改未来两个 token。
    x_modified[:, 3:, :] = torch.randn(1, 2, 8) * 100.0

    output_original = model(x_original)
    output_modified = model(x_modified)

    torch.testing.assert_close(
        output_original[:, :3, :],
        output_modified[:, :3, :],
        rtol=1e-5,
        atol=1e-6,
    )


def test_current_token_can_affect_its_own_output() -> None:
    """修改当前位置本身，通常应改变该位置的输出。"""

    torch.manual_seed(6)

    model = CausalSelfAttention(hidden_size=8)
    model.eval()

    x_original = torch.randn(1, 4, 8)
    x_modified = x_original.clone()

    x_modified[:, 2, :] += 10.0

    output_original = model(x_original)
    output_modified = model(x_modified)

    assert not torch.allclose(
        output_original[:, 2, :],
        output_modified[:, 2, :],
    )


def test_custom_positions_are_supported() -> None:
    """应支持手动传入绝对位置，用于后续 KV Cache。"""

    torch.manual_seed(7)

    model = CausalSelfAttention(hidden_size=8)
    x = torch.randn(2, 3, 8)

    positions = torch.tensor([10, 11, 12])

    output = model(
        x,
        positions=positions,
    )

    assert output.shape == x.shape
    assert torch.isfinite(output).all()


def test_backward_pass() -> None:
    """输入与所有线性投影参数都应获得有限梯度。"""

    torch.manual_seed(8)

    model = CausalSelfAttention(hidden_size=8)

    x = torch.randn(
        2,
        5,
        8,
        requires_grad=True,
    )

    output = model(x)
    loss = output.square().mean()

    loss.backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()

    projection_layers = [
        model.q_proj,
        model.k_proj,
        model.v_proj,
        model.out_proj,
    ]

    for layer in projection_layers:
        assert layer.weight.grad is not None
        assert torch.isfinite(layer.weight.grad).all()


def test_projection_parameter_shapes() -> None:
    """单头版本的四个投影矩阵都应为 [D, D]。"""

    hidden_size = 8
    model = CausalSelfAttention(hidden_size=hidden_size)

    assert model.q_proj.weight.shape == (
        hidden_size,
        hidden_size,
    )
    assert model.k_proj.weight.shape == (
        hidden_size,
        hidden_size,
    )
    assert model.v_proj.weight.shape == (
        hidden_size,
        hidden_size,
    )
    assert model.out_proj.weight.shape == (
        hidden_size,
        hidden_size,
    )


def test_bias_option() -> None:
    """bias 参数应控制线性层是否包含偏置。"""

    model_without_bias = CausalSelfAttention(
        hidden_size=8,
        bias=False,
    )

    assert model_without_bias.q_proj.bias is None
    assert model_without_bias.k_proj.bias is None
    assert model_without_bias.v_proj.bias is None
    assert model_without_bias.out_proj.bias is None

    model_with_bias = CausalSelfAttention(
        hidden_size=8,
        bias=True,
    )

    assert model_with_bias.q_proj.bias is not None
    assert model_with_bias.k_proj.bias is not None
    assert model_with_bias.v_proj.bias is not None
    assert model_with_bias.out_proj.bias is not None


def test_non_positive_hidden_size_is_rejected() -> None:
    with pytest.raises(ValueError):
        CausalSelfAttention(hidden_size=0)


def test_odd_hidden_size_is_rejected() -> None:
    """当前版本使用全维 RoPE，因此 hidden_size 必须为偶数。"""

    with pytest.raises(ValueError):
        CausalSelfAttention(hidden_size=7)


def test_wrong_number_of_dimensions_is_rejected() -> None:
    model = CausalSelfAttention(hidden_size=8)

    # 少了 batch 维度。
    x = torch.randn(5, 8)

    with pytest.raises(ValueError):
        model(x)


def test_wrong_hidden_dimension_is_rejected() -> None:
    model = CausalSelfAttention(hidden_size=8)

    x = torch.randn(2, 5, 10)

    with pytest.raises(ValueError):
        model(x)


def test_integer_input_is_rejected() -> None:
    model = CausalSelfAttention(hidden_size=8)

    x = torch.ones(
        2,
        5,
        8,
        dtype=torch.int64,
    )

    with pytest.raises(TypeError):
        model(x)


def test_wrong_positions_length_is_rejected() -> None:
    model = CausalSelfAttention(hidden_size=8)

    x = torch.randn(2, 5, 8)
    positions = torch.tensor([0, 1, 2])

    with pytest.raises(ValueError):
        model(
            x,
            positions=positions,
        )

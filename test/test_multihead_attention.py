import pytest
import torch

from tinyseek.model.attention import CausalSelfAttention
from tinyseek.model.multihead_attention import (
    MultiHeadCausalSelfAttention,
)


def test_output_shape_and_dtype() -> None:
    """多头 Attention 不应改变输入形状和数据类型。"""

    torch.manual_seed(0)

    model = MultiHeadCausalSelfAttention(
        hidden_size=8,
        num_heads=2,
    )

    x = torch.randn(2, 5, 8)
    output = model(x)

    assert isinstance(output, torch.Tensor)
    assert output.shape == x.shape
    assert output.dtype == x.dtype


def test_attention_weights_shape() -> None:
    """每个注意力头都应拥有一张独立的注意力矩阵。"""

    torch.manual_seed(1)

    model = MultiHeadCausalSelfAttention(
        hidden_size=8,
        num_heads=2,
    )

    x = torch.randn(3, 6, 8)

    output, attention_weights = model(
        x,
        return_attention=True,
    )

    assert output.shape == (3, 6, 8)

    assert attention_weights.shape == (
        3,  # batch_size
        2,  # num_heads
        6,  # query sequence length
        6,  # key sequence length
    )


def test_each_head_rows_sum_to_one() -> None:
    """每个头中，每个 Query 的注意力权重之和都应为1。"""

    torch.manual_seed(2)

    model = MultiHeadCausalSelfAttention(
        hidden_size=12,
        num_heads=3,
    )

    x = torch.randn(2, 7, 12)

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
    """所有注意力头都不能关注未来 token。"""

    torch.manual_seed(3)

    sequence_length = 6

    model = MultiHeadCausalSelfAttention(
        hidden_size=8,
        num_heads=2,
    )

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

    # attention_weights: [B, H, T, T]
    # 取出最后两维上所有未来位置。
    future_weights = attention_weights[..., future_mask]

    torch.testing.assert_close(
        future_weights,
        torch.zeros_like(future_weights),
        rtol=0.0,
        atol=0.0,
    )


def test_first_token_only_attends_to_itself() -> None:
    """第0个 token 在每个头中都只能关注自己。"""

    torch.manual_seed(4)

    model = MultiHeadCausalSelfAttention(
        hidden_size=8,
        num_heads=2,
    )

    x = torch.randn(3, 5, 8)

    _, attention_weights = model(
        x,
        return_attention=True,
    )

    # [B, H, T]
    first_token_weights = attention_weights[:, :, 0, :]

    expected = torch.zeros_like(first_token_weights)
    expected[:, :, 0] = 1.0

    torch.testing.assert_close(
        first_token_weights,
        expected,
        rtol=0.0,
        atol=0.0,
    )


def test_future_tokens_do_not_affect_past_outputs() -> None:
    """修改未来 token 不应改变过去位置的输出。"""

    torch.manual_seed(5)

    model = MultiHeadCausalSelfAttention(
        hidden_size=8,
        num_heads=2,
    )
    model.eval()

    x_original = torch.randn(1, 6, 8)
    x_modified = x_original.clone()

    # 大幅修改位置3、4、5。
    x_modified[:, 3:, :] = (
        torch.randn(1, 3, 8) * 1000.0
    )

    output_original = model(x_original)
    output_modified = model(x_modified)

    # 位置0、1、2绝对不能受到未来位置影响。
    torch.testing.assert_close(
        output_original[:, :3, :],
        output_modified[:, :3, :],
        rtol=1e-5,
        atol=1e-6,
    )


def test_split_heads_shape() -> None:
    """拆头后的形状应为 [B,H,T,Dh]。"""

    model = MultiHeadCausalSelfAttention(
        hidden_size=12,
        num_heads=3,
    )

    x = torch.randn(2, 5, 12)

    split = model._split_heads(x)

    assert split.shape == (
        2,  # batch_size
        3,  # num_heads
        5,  # sequence_length
        4,  # head_dim
    )


def test_split_and_merge_are_inverse_operations() -> None:
    """拆头后再合头，应恢复原始 Tensor。"""

    torch.manual_seed(6)

    model = MultiHeadCausalSelfAttention(
        hidden_size=12,
        num_heads=3,
    )

    x = torch.randn(2, 5, 12)

    split = model._split_heads(x)
    merged = model._merge_heads(split)

    assert merged.shape == x.shape

    torch.testing.assert_close(
        merged,
        x,
        rtol=0.0,
        atol=0.0,
    )


def test_one_head_matches_single_head_implementation() -> None:
    """num_heads=1 时，多头版本应与单头版本完全对齐。"""

    torch.manual_seed(7)

    single_head = CausalSelfAttention(
        hidden_size=8,
        bias=False,
    )

    multi_head = MultiHeadCausalSelfAttention(
        hidden_size=8,
        num_heads=1,
        bias=False,
    )

    # 两个模型结构在 num_heads=1 时参数形状相同。
    # 将单头模型参数复制给多头模型。
    multi_head.load_state_dict(
        single_head.state_dict(),
        strict=True,
    )

    x = torch.randn(2, 5, 8)

    single_output, single_weights = single_head(
        x,
        return_attention=True,
    )

    multi_output, multi_weights = multi_head(
        x,
        return_attention=True,
    )

    torch.testing.assert_close(
        multi_output,
        single_output,
        rtol=1e-5,
        atol=1e-6,
    )

    # 多头版本多一个 head 维度，将其去掉后比较。
    torch.testing.assert_close(
        multi_weights[:, 0, :, :],
        single_weights,
        rtol=1e-5,
        atol=1e-6,
    )


def test_global_position_shift_does_not_change_output() -> None:
    """同时平移全部位置，不应改变相对位置注意力结果。"""

    torch.manual_seed(8)

    model = MultiHeadCausalSelfAttention(
        hidden_size=8,
        num_heads=2,
    )
    model.eval()

    x = torch.randn(2, 4, 8)

    positions_a = torch.tensor([0, 1, 2, 3])
    positions_b = torch.tensor([10, 11, 12, 13])

    output_a, weights_a = model(
        x,
        positions=positions_a,
        return_attention=True,
    )

    output_b, weights_b = model(
        x,
        positions=positions_b,
        return_attention=True,
    )

    # 两组位置的绝对值不同，但相对距离完全相同。
    torch.testing.assert_close(
        weights_a,
        weights_b,
        rtol=1e-5,
        atol=1e-6,
    )

    torch.testing.assert_close(
        output_a,
        output_b,
        rtol=1e-5,
        atol=1e-6,
    )


def test_backward_pass() -> None:
    """梯度应传回输入和所有可训练投影矩阵。"""

    torch.manual_seed(9)

    model = MultiHeadCausalSelfAttention(
        hidden_size=8,
        num_heads=2,
    )

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
    assert x.grad.shape == x.shape
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

    # RoPE 的 inv_freq 是 Buffer，不应成为训练参数。
    assert model.rope.inv_freq.grad is None


def test_projection_parameter_shapes() -> None:
    """四个投影矩阵都应保持 [D,D] 的形状。"""

    hidden_size = 12

    model = MultiHeadCausalSelfAttention(
        hidden_size=hidden_size,
        num_heads=3,
    )

    expected_shape = (
        hidden_size,
        hidden_size,
    )

    assert model.q_proj.weight.shape == expected_shape
    assert model.k_proj.weight.shape == expected_shape
    assert model.v_proj.weight.shape == expected_shape
    assert model.out_proj.weight.shape == expected_shape


def test_scale_uses_head_dimension() -> None:
    """缩放系数应为 1/sqrt(head_dim)，而不是 1/sqrt(D)。"""

    model = MultiHeadCausalSelfAttention(
        hidden_size=16,
        num_heads=4,
    )

    assert model.head_dim == 4
    assert model.scale == pytest.approx(4 ** -0.5)


def test_bias_option() -> None:
    """bias 参数应控制所有线性层是否包含偏置。"""

    without_bias = MultiHeadCausalSelfAttention(
        hidden_size=8,
        num_heads=2,
        bias=False,
    )

    assert without_bias.q_proj.bias is None
    assert without_bias.k_proj.bias is None
    assert without_bias.v_proj.bias is None
    assert without_bias.out_proj.bias is None

    with_bias = MultiHeadCausalSelfAttention(
        hidden_size=8,
        num_heads=2,
        bias=True,
    )

    assert with_bias.q_proj.bias is not None
    assert with_bias.k_proj.bias is not None
    assert with_bias.v_proj.bias is not None
    assert with_bias.out_proj.bias is not None


def test_non_positive_hidden_size_is_rejected() -> None:
    with pytest.raises(ValueError):
        MultiHeadCausalSelfAttention(
            hidden_size=0,
            num_heads=2,
        )


def test_non_positive_num_heads_is_rejected() -> None:
    with pytest.raises(ValueError):
        MultiHeadCausalSelfAttention(
            hidden_size=8,
            num_heads=0,
        )


def test_hidden_size_must_be_divisible_by_num_heads() -> None:
    with pytest.raises(ValueError):
        MultiHeadCausalSelfAttention(
            hidden_size=10,
            num_heads=3,
        )


def test_odd_head_dimension_is_rejected() -> None:
    """当前 RoPE 要求每个头的维度为偶数。"""

    # 12 / 4 = 3，head_dim 为奇数。
    with pytest.raises(ValueError):
        MultiHeadCausalSelfAttention(
            hidden_size=12,
            num_heads=4,
        )


def test_wrong_number_of_dimensions_is_rejected() -> None:
    model = MultiHeadCausalSelfAttention(
        hidden_size=8,
        num_heads=2,
    )

    x = torch.randn(5, 8)

    with pytest.raises(ValueError):
        model(x)


def test_wrong_hidden_dimension_is_rejected() -> None:
    model = MultiHeadCausalSelfAttention(
        hidden_size=8,
        num_heads=2,
    )

    x = torch.randn(2, 5, 10)

    with pytest.raises(ValueError):
        model(x)


def test_integer_input_is_rejected() -> None:
    model = MultiHeadCausalSelfAttention(
        hidden_size=8,
        num_heads=2,
    )

    x = torch.ones(
        2,
        5,
        8,
        dtype=torch.int64,
    )

    with pytest.raises(TypeError):
        model(x)


def test_wrong_positions_length_is_rejected() -> None:
    model = MultiHeadCausalSelfAttention(
        hidden_size=8,
        num_heads=2,
    )

    x = torch.randn(2, 5, 8)

    # 序列长度为5，这里只提供3个位置。
    positions = torch.tensor([0, 1, 2])

    with pytest.raises(ValueError):
        model(
            x,
            positions=positions,
        )

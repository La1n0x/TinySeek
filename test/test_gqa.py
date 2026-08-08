import pytest
import torch

from tinyseek.model.gqa import GroupedQueryAttention
from tinyseek.model.multihead_attention import (
    MultiHeadCausalSelfAttention,
)


def test_output_shape_and_dtype() -> None:
    """GQA 不应改变输入形状和数据类型。"""

    torch.manual_seed(0)

    model = GroupedQueryAttention(
        hidden_size=16,
        num_heads=4,
        num_kv_heads=2,
    )

    x = torch.randn(2, 5, 16)
    output = model(x)

    assert isinstance(output, torch.Tensor)
    assert output.shape == x.shape
    assert output.dtype == x.dtype


def test_attention_weights_shape() -> None:
    """注意力矩阵数量由 Query Head 数量决定。"""

    torch.manual_seed(1)

    model = GroupedQueryAttention(
        hidden_size=16,
        num_heads=4,
        num_kv_heads=2,
    )

    x = torch.randn(3, 6, 16)

    output, attention_weights = model(
        x,
        return_attention=True,
    )

    assert output.shape == (3, 6, 16)
    assert attention_weights.shape == (
        3,  # batch_size
        4,  # num_heads / Query Heads
        6,  # query sequence length
        6,  # key sequence length
    )


def test_q_k_v_projection_shapes() -> None:
    """Q 保持完整维度，而 K/V 的总维度应按 KV Head 数量缩小。"""

    model = GroupedQueryAttention(
        hidden_size=16,
        num_heads=4,
        num_kv_heads=2,
    )

    x = torch.randn(2, 5, 16)

    query = model.q_proj(x)
    key = model.k_proj(x)
    value = model.v_proj(x)

    assert query.shape == (2, 5, 16)
    assert key.shape == (2, 5, 8)
    assert value.shape == (2, 5, 8)


def test_split_query_heads_shape() -> None:
    """Query 拆头后应为 [B, num_heads, T, head_dim]。"""

    model = GroupedQueryAttention(
        hidden_size=16,
        num_heads=4,
        num_kv_heads=2,
    )

    x = torch.randn(2, 5, 16)
    split = model._split_query_heads(x)

    assert split.shape == (
        2,  # batch_size
        4,  # num_heads
        5,  # sequence_length
        4,  # head_dim
    )


def test_split_kv_heads_shape() -> None:
    """K/V 拆头后应为 [B, num_kv_heads, T, head_dim]。"""

    model = GroupedQueryAttention(
        hidden_size=16,
        num_heads=4,
        num_kv_heads=2,
    )

    x = torch.randn(2, 5, 8)
    split = model._split_kv_heads(x)

    assert split.shape == (
        2,  # batch_size
        2,  # num_kv_heads
        5,  # sequence_length
        4,  # head_dim
    )


def test_repeat_kv_mapping() -> None:
    """两个 KV Head 应按组映射为 K0,K0,K1,K1。"""

    model = GroupedQueryAttention(
        hidden_size=16,
        num_heads=4,
        num_kv_heads=2,
    )

    # 每个 KV Head 填入不同常数，方便直接观察映射。
    kv = torch.empty(1, 2, 3, 4)
    kv[:, 0, :, :] = 10.0
    kv[:, 1, :, :] = 20.0

    repeated = model._repeat_kv(kv)

    assert repeated.shape == (1, 4, 3, 4)

    torch.testing.assert_close(
        repeated[:, 0],
        kv[:, 0],
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        repeated[:, 1],
        kv[:, 0],
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        repeated[:, 2],
        kv[:, 1],
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        repeated[:, 3],
        kv[:, 1],
        rtol=0.0,
        atol=0.0,
    )


def test_repeat_kv_is_identity_when_every_query_has_own_kv_head() -> None:
    """num_kv_heads == num_heads 时，不需要复制 KV Head。"""

    model = GroupedQueryAttention(
        hidden_size=16,
        num_heads=4,
        num_kv_heads=4,
    )

    kv = torch.randn(2, 4, 5, 4)
    repeated = model._repeat_kv(kv)

    assert repeated is kv


def test_split_and_merge_query_heads_are_inverse_operations() -> None:
    """Query 拆头后再合头，应恢复原 Tensor。"""

    torch.manual_seed(2)

    model = GroupedQueryAttention(
        hidden_size=16,
        num_heads=4,
        num_kv_heads=2,
    )

    x = torch.randn(2, 5, 16)

    split = model._split_query_heads(x)
    merged = model._merge_query_heads(split)

    assert merged.shape == x.shape

    torch.testing.assert_close(
        merged,
        x,
        rtol=0.0,
        atol=0.0,
    )


def test_each_query_head_rows_sum_to_one() -> None:
    """每个 Query Head 中，每个 Query 的注意力权重之和都应为 1。"""

    torch.manual_seed(3)

    model = GroupedQueryAttention(
        hidden_size=16,
        num_heads=4,
        num_kv_heads=2,
    )

    x = torch.randn(2, 7, 16)

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
    """所有 Query Head 都不能关注未来 token。"""

    torch.manual_seed(4)

    sequence_length = 6

    model = GroupedQueryAttention(
        hidden_size=16,
        num_heads=4,
        num_kv_heads=2,
    )

    x = torch.randn(2, sequence_length, 16)

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

    future_weights = attention_weights[..., future_mask]

    torch.testing.assert_close(
        future_weights,
        torch.zeros_like(future_weights),
        rtol=0.0,
        atol=0.0,
    )


def test_first_token_only_attends_to_itself() -> None:
    """第 0 个 token 在每个 Query Head 中都只能关注自己。"""

    torch.manual_seed(5)

    model = GroupedQueryAttention(
        hidden_size=16,
        num_heads=4,
        num_kv_heads=2,
    )

    x = torch.randn(3, 5, 16)

    _, attention_weights = model(
        x,
        return_attention=True,
    )

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

    torch.manual_seed(6)

    model = GroupedQueryAttention(
        hidden_size=16,
        num_heads=4,
        num_kv_heads=2,
    )
    model.eval()

    x_original = torch.randn(1, 6, 16)
    x_modified = x_original.clone()

    x_modified[:, 3:, :] = (
        torch.randn(1, 3, 16) * 1000.0
    )

    output_original = model(x_original)
    output_modified = model(x_modified)

    torch.testing.assert_close(
        output_original[:, :3, :],
        output_modified[:, :3, :],
        rtol=1e-5,
        atol=1e-6,
    )


def test_global_position_shift_does_not_change_output() -> None:
    """同时平移全部位置，不应改变仅依赖相对位置的注意力结果。"""

    torch.manual_seed(7)

    model = GroupedQueryAttention(
        hidden_size=16,
        num_heads=4,
        num_kv_heads=2,
    )
    model.eval()

    x = torch.randn(2, 4, 16)

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

    torch.manual_seed(8)

    model = GroupedQueryAttention(
        hidden_size=16,
        num_heads=4,
        num_kv_heads=2,
    )

    x = torch.randn(
        2,
        5,
        16,
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

    assert model.rope.inv_freq.grad is None


def test_projection_parameter_shapes() -> None:
    """GQA 的 K/V 投影矩阵应比 Q 投影矩阵窄。"""

    model = GroupedQueryAttention(
        hidden_size=16,
        num_heads=4,
        num_kv_heads=2,
    )

    assert model.head_dim == 4

    assert model.q_proj.weight.shape == (16, 16)
    assert model.k_proj.weight.shape == (8, 16)
    assert model.v_proj.weight.shape == (8, 16)
    assert model.out_proj.weight.shape == (16, 16)


def test_scale_uses_head_dimension() -> None:
    """缩放系数应为 1/sqrt(head_dim)。"""

    model = GroupedQueryAttention(
        hidden_size=16,
        num_heads=4,
        num_kv_heads=2,
    )

    assert model.head_dim == 4
    assert model.scale == pytest.approx(4 ** -0.5)


def test_bias_option() -> None:
    """bias 参数应控制所有线性层是否包含偏置。"""

    without_bias = GroupedQueryAttention(
        hidden_size=16,
        num_heads=4,
        num_kv_heads=2,
        bias=False,
    )

    assert without_bias.q_proj.bias is None
    assert without_bias.k_proj.bias is None
    assert without_bias.v_proj.bias is None
    assert without_bias.out_proj.bias is None

    with_bias = GroupedQueryAttention(
        hidden_size=16,
        num_heads=4,
        num_kv_heads=2,
        bias=True,
    )

    assert with_bias.q_proj.bias is not None
    assert with_bias.k_proj.bias is not None
    assert with_bias.v_proj.bias is not None
    assert with_bias.out_proj.bias is not None


def test_gqa_matches_mha_when_num_kv_heads_equals_num_heads() -> None:
    """当每个 Query Head 都有自己的 KV Head 时，GQA 必须严格退化为 MHA。"""

    torch.manual_seed(9)

    mha = MultiHeadCausalSelfAttention(
        hidden_size=16,
        num_heads=4,
        bias=False,
    )

    gqa = GroupedQueryAttention(
        hidden_size=16,
        num_heads=4,
        num_kv_heads=4,
        bias=False,
    )

    # 此时两个模型的 Q/K/V/out projection 参数形状完全相同。
    gqa.load_state_dict(
        mha.state_dict(),
        strict=True,
    )

    x = torch.randn(2, 5, 16)

    mha_output, mha_weights = mha(
        x,
        return_attention=True,
    )

    gqa_output, gqa_weights = gqa(
        x,
        return_attention=True,
    )

    torch.testing.assert_close(
        gqa_output,
        mha_output,
        rtol=1e-5,
        atol=1e-6,
    )

    torch.testing.assert_close(
        gqa_weights,
        mha_weights,
        rtol=1e-5,
        atol=1e-6,
    )


def test_gqa_uses_fewer_projection_parameters_than_mha() -> None:
    """减少 K/V Head 后，GQA 的可训练投影参数量应下降。"""

    hidden_size = 16
    num_heads = 4
    num_kv_heads = 2

    mha = MultiHeadCausalSelfAttention(
        hidden_size=hidden_size,
        num_heads=num_heads,
        bias=False,
    )

    gqa = GroupedQueryAttention(
        hidden_size=hidden_size,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        bias=False,
    )

    mha_parameter_count = sum(
        parameter.numel()
        for parameter in mha.parameters()
    )

    gqa_parameter_count = sum(
        parameter.numel()
        for parameter in gqa.parameters()
    )

    assert gqa_parameter_count < mha_parameter_count

    head_dim = hidden_size // num_heads
    kv_size = num_kv_heads * head_dim

    expected_reduction = (
        2 * hidden_size * (hidden_size - kv_size)
    )

    assert (
        mha_parameter_count - gqa_parameter_count
        == expected_reduction
    )


def test_theoretical_kv_cache_size_is_reduced() -> None:
    """GQA 的理论 KV Cache 元素数应按 num_kv_heads/num_heads 比例下降。"""

    batch_size = 2
    sequence_length = 10
    hidden_size = 16
    num_heads = 4
    num_kv_heads = 2

    head_dim = hidden_size // num_heads

    mha_kv_elements = (
        2
        * batch_size
        * num_heads
        * sequence_length
        * head_dim
    )

    gqa_kv_elements = (
        2
        * batch_size
        * num_kv_heads
        * sequence_length
        * head_dim
    )

    assert gqa_kv_elements < mha_kv_elements
    assert gqa_kv_elements * num_heads == (
        mha_kv_elements * num_kv_heads
    )

    # 本例 num_kv_heads / num_heads = 2 / 4 = 1/2。
    assert gqa_kv_elements == mha_kv_elements // 2


def test_non_positive_hidden_size_is_rejected() -> None:
    with pytest.raises(ValueError):
        GroupedQueryAttention(
            hidden_size=0,
            num_heads=4,
            num_kv_heads=2,
        )


def test_non_positive_num_heads_is_rejected() -> None:
    with pytest.raises(ValueError):
        GroupedQueryAttention(
            hidden_size=16,
            num_heads=0,
            num_kv_heads=2,
        )


def test_non_positive_num_kv_heads_is_rejected() -> None:
    with pytest.raises(ValueError):
        GroupedQueryAttention(
            hidden_size=16,
            num_heads=4,
            num_kv_heads=0,
        )


def test_hidden_size_must_be_divisible_by_num_heads() -> None:
    with pytest.raises(ValueError):
        GroupedQueryAttention(
            hidden_size=10,
            num_heads=3,
            num_kv_heads=1,
        )


def test_num_heads_must_be_divisible_by_num_kv_heads() -> None:
    with pytest.raises(ValueError):
        GroupedQueryAttention(
            hidden_size=20,
            num_heads=5,
            num_kv_heads=2,
        )


def test_odd_head_dimension_is_rejected() -> None:
    """当前 RoPE 要求每个 Head 的维度为偶数。"""

    # 12 / 4 = 3，head_dim 为奇数。
    with pytest.raises(ValueError):
        GroupedQueryAttention(
            hidden_size=12,
            num_heads=4,
            num_kv_heads=2,
        )


def test_wrong_number_of_dimensions_is_rejected() -> None:
    model = GroupedQueryAttention(
        hidden_size=16,
        num_heads=4,
        num_kv_heads=2,
    )

    x = torch.randn(5, 16)

    with pytest.raises(ValueError):
        model(x)


def test_wrong_hidden_dimension_is_rejected() -> None:
    model = GroupedQueryAttention(
        hidden_size=16,
        num_heads=4,
        num_kv_heads=2,
    )

    x = torch.randn(2, 5, 12)

    with pytest.raises(ValueError):
        model(x)


def test_integer_input_is_rejected() -> None:
    model = GroupedQueryAttention(
        hidden_size=16,
        num_heads=4,
        num_kv_heads=2,
    )

    x = torch.ones(
        2,
        5,
        16,
        dtype=torch.int64,
    )

    with pytest.raises(TypeError):
        model(x)


def test_wrong_positions_length_is_rejected() -> None:
    model = GroupedQueryAttention(
        hidden_size=16,
        num_heads=4,
        num_kv_heads=2,
    )

    x = torch.randn(2, 5, 16)
    positions = torch.tensor([0, 1, 2])

    with pytest.raises(ValueError):
        model(
            x,
            positions=positions,
        )

import pytest
import torch

from tinyseek.model.decoder_block import DecoderBlock


def make_block(
    hidden_size=16,
    num_heads=4,
    num_kv_heads=2,
    intermediate_size=32,
    bias=False,
):
    return DecoderBlock(
        hidden_size=hidden_size,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        intermediate_size=intermediate_size,
        bias=bias,
    )


# ============================================================
# 1. 基本 shape
# ============================================================


def test_output_shape():
    block = make_block()

    x = torch.randn(2, 5, 16)

    y = block(x)

    assert y.shape == (2, 5, 16)


def test_return_attention_shape():
    block = make_block()

    x = torch.randn(2, 5, 16)

    y, attention = block(
        x,
        return_attention=True,
    )

    assert y.shape == (2, 5, 16)

    # B = 2
    # H = 4
    # T = 5
    assert attention.shape == (2, 4, 5, 5)


# ============================================================
# 2. 验证 Decoder Block 的完整数学逻辑
#
# h = x + Attention(Norm1(x))
#
# y = h + MLP(Norm2(h))
# ============================================================


def test_matches_manual_pre_norm_computation():
    torch.manual_seed(0)

    block = make_block()

    x = torch.randn(2, 5, 16)

    # ----------------------------
    # 手工执行第一半
    # ----------------------------

    normed_attention_input = block.attn_norm(x)

    attention_output = block.attention(
        normed_attention_input
    )

    hidden = x + attention_output

    # ----------------------------
    # 手工执行第二半
    # ----------------------------

    normed_mlp_input = block.mlp_norm(hidden)

    mlp_output = block.mlp(
        normed_mlp_input
    )

    expected = hidden + mlp_output

    # ----------------------------
    # 正常 forward
    # ----------------------------

    actual = block(x)

    assert torch.allclose(
        actual,
        expected,
        atol=1e-6,
        rtol=1e-6,
    )


# ============================================================
# 3. Residual Connection
# ============================================================


def test_residual_identity_when_sublayers_are_zero():
    """
    如果 Attention 和 MLP 都输出 0：

        h = x + 0
        y = h + 0

    因此最终必须严格得到 x。
    """

    block = make_block()

    with torch.no_grad():

        for parameter in block.attention.parameters():
            parameter.zero_()

        for parameter in block.mlp.parameters():
            parameter.zero_()

    x = torch.randn(2, 5, 16)

    y = block(x)

    assert torch.allclose(
        y,
        x,
        atol=1e-6,
        rtol=1e-6,
    )


def test_second_residual_when_mlp_is_zero():
    """
    如果只把 MLP 设为 0：

        h = x + Attention(Norm(x))
        y = h + 0

    所以最终输出必须等于 h。
    """

    torch.manual_seed(0)

    block = make_block()

    with torch.no_grad():
        for parameter in block.mlp.parameters():
            parameter.zero_()

    x = torch.randn(2, 5, 16)

    attention_output = block.attention(
        block.attn_norm(x)
    )

    expected = x + attention_output

    actual = block(x)

    assert torch.allclose(
        actual,
        expected,
        atol=1e-6,
        rtol=1e-6,
    )


# ============================================================
# 4. Attention 权重
# ============================================================


def test_attention_weights_sum_to_one():
    block = make_block()

    x = torch.randn(2, 5, 16)

    _, attention = block(
        x,
        return_attention=True,
    )

    sums = attention.sum(dim=-1)

    assert torch.allclose(
        sums,
        torch.ones_like(sums),
        atol=1e-6,
        rtol=1e-6,
    )


def test_causal_attention_weights():
    block = make_block()

    x = torch.randn(1, 5, 16)

    _, attention = block(
        x,
        return_attention=True,
    )

    # future positions:
    #
    # [0,1], [0,2], ...
    # [1,2], [1,3], ...
    #
    # 都应该严格为 0。

    future_mask = torch.triu(
        torch.ones(
            5,
            5,
            dtype=torch.bool,
        ),
        diagonal=1,
    )

    future_weights = attention[
        :,
        :,
        future_mask,
    ]

    assert torch.all(
        future_weights == 0
    )


def test_first_token_can_only_attend_to_itself():
    block = make_block()

    x = torch.randn(2, 5, 16)

    _, attention = block(
        x,
        return_attention=True,
    )

    first_token_attention = attention[
        :,
        :,
        0,
        :,
    ]

    expected = torch.zeros_like(
        first_token_attention
    )

    expected[..., 0] = 1.0

    assert torch.allclose(
        first_token_attention,
        expected,
        atol=1e-6,
        rtol=1e-6,
    )


# ============================================================
# 5. 整个 Decoder Block 必须保持因果性
# ============================================================


def test_future_token_does_not_affect_past_output():
    torch.manual_seed(0)

    block = make_block()

    x1 = torch.randn(1, 5, 16)

    x2 = x1.clone()

    # 只疯狂修改最后一个 token
    x2[:, 4, :] += 100.0

    y1 = block(x1)
    y2 = block(x2)

    # token 0~3 不能受未来 token4 影响
    assert torch.allclose(
        y1[:, :4, :],
        y2[:, :4, :],
        atol=1e-5,
        rtol=1e-5,
    )

    # token4 自己应该发生变化
    assert not torch.allclose(
        y1[:, 4, :],
        y2[:, 4, :],
    )


# ============================================================
# 6. RoPE 相对位置性质
# ============================================================


def test_global_position_shift_invariance():
    """
    RoPE Attention 只应依赖相对位置。

    positions:
        0 1 2 3 4

    全部平移：

        10 11 12 13 14

    相对位置不变，因此输出应保持一致。
    """

    torch.manual_seed(0)

    block = make_block()

    x = torch.randn(2, 5, 16)

    positions1 = torch.arange(5)
    positions2 = positions1 + 10

    y1 = block(
        x,
        positions=positions1,
    )

    y2 = block(
        x,
        positions=positions2,
    )

    assert torch.allclose(
        y1,
        y2,
        atol=1e-5,
        rtol=1e-5,
    )


# ============================================================
# 7. 两个 RMSNorm 必须独立
# ============================================================


def test_two_norms_are_independent():
    block = make_block()

    assert (
        block.attn_norm.weight.data_ptr()
        != block.mlp_norm.weight.data_ptr()
    )


# ============================================================
# 8. Gradient
# ============================================================


def test_backward():
    torch.manual_seed(0)

    block = make_block()

    x = torch.randn(
        2,
        5,
        16,
        requires_grad=True,
    )

    y = block(x)

    loss = y.square().mean()

    loss.backward()

    # 输入必须收到梯度
    assert x.grad is not None

    # RMSNorm
    assert block.attn_norm.weight.grad is not None
    assert block.mlp_norm.weight.grad is not None

    # Attention
    assert block.attention.q_proj.weight.grad is not None
    assert block.attention.k_proj.weight.grad is not None
    assert block.attention.v_proj.weight.grad is not None
    assert block.attention.out_proj.weight.grad is not None

    # SwiGLU
    assert block.mlp.gate_proj.weight.grad is not None
    assert block.mlp.up_proj.weight.grad is not None
    assert block.mlp.down_proj.weight.grad is not None


def test_gradient_shapes():
    block = make_block()

    x = torch.randn(
        2,
        5,
        16,
        requires_grad=True,
    )

    block(x).sum().backward()

    assert (
        block.attention.q_proj.weight.grad.shape
        == (16, 16)
    )

    # GQA:
    # Hkv = 2
    # Dh  = 4
    #
    # K/V projection:
    # 16 -> 8

    assert (
        block.attention.k_proj.weight.grad.shape
        == (8, 16)
    )

    assert (
        block.attention.v_proj.weight.grad.shape
        == (8, 16)
    )

    assert (
        block.mlp.gate_proj.weight.grad.shape
        == (32, 16)
    )

    assert (
        block.mlp.up_proj.weight.grad.shape
        == (32, 16)
    )

    assert (
        block.mlp.down_proj.weight.grad.shape
        == (16, 32)
    )


# ============================================================
# 9. bias
# ============================================================


def test_bias_false():
    block = make_block(
        bias=False
    )

    assert block.attention.q_proj.bias is None
    assert block.attention.k_proj.bias is None
    assert block.attention.v_proj.bias is None
    assert block.attention.out_proj.bias is None

    assert block.mlp.gate_proj.bias is None
    assert block.mlp.up_proj.bias is None
    assert block.mlp.down_proj.bias is None


def test_bias_true():
    block = make_block(
        bias=True
    )

    assert block.attention.q_proj.bias is not None
    assert block.attention.k_proj.bias is not None
    assert block.attention.v_proj.bias is not None
    assert block.attention.out_proj.bias is not None

    assert block.mlp.gate_proj.bias is not None
    assert block.mlp.up_proj.bias is not None
    assert block.mlp.down_proj.bias is not None


# ============================================================
# 10. 多种尺寸
# ============================================================


@pytest.mark.parametrize(
    (
        "hidden_size",
        "num_heads",
        "num_kv_heads",
        "intermediate_size",
    ),
    [
        (8, 2, 1, 16),
        (16, 4, 2, 32),
        (32, 8, 2, 64),
        (64, 8, 4, 128),
    ],
)
def test_multiple_model_sizes(
    hidden_size,
    num_heads,
    num_kv_heads,
    intermediate_size,
):
    block = DecoderBlock(
        hidden_size=hidden_size,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        intermediate_size=intermediate_size,
    )

    x = torch.randn(
        2,
        3,
        hidden_size,
    )

    y = block(x)

    assert y.shape == (
        2,
        3,
        hidden_size,
    )


# ============================================================
# 11. Constructor validation
# ============================================================


@pytest.mark.parametrize(
    "hidden_size",
    [
        0,
        -1,
        -16,
    ],
)
def test_invalid_hidden_size(
    hidden_size,
):
    with pytest.raises(ValueError):
        DecoderBlock(
            hidden_size=hidden_size,
            num_heads=4,
            num_kv_heads=2,
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
def test_invalid_intermediate_size(
    intermediate_size,
):
    with pytest.raises(ValueError):
        DecoderBlock(
            hidden_size=16,
            num_heads=4,
            num_kv_heads=2,
            intermediate_size=intermediate_size,
        )


@pytest.mark.parametrize(
    "eps",
    [
        0,
        -1e-6,
        -1.0,
    ],
)
def test_invalid_eps(
    eps,
):
    with pytest.raises(ValueError):
        DecoderBlock(
            hidden_size=16,
            num_heads=4,
            num_kv_heads=2,
            intermediate_size=32,
            eps=eps,
        )


def test_invalid_num_heads_propagates():
    with pytest.raises(ValueError):
        DecoderBlock(
            hidden_size=16,
            num_heads=3,
            num_kv_heads=1,
            intermediate_size=32,
        )


def test_invalid_num_kv_heads_propagates():
    with pytest.raises(ValueError):
        DecoderBlock(
            hidden_size=16,
            num_heads=4,
            num_kv_heads=3,
            intermediate_size=32,
        )


# ============================================================
# 12. Input validation
# ============================================================


def test_invalid_input_rank():
    block = make_block()

    x = torch.randn(
        2,
        16,
    )

    with pytest.raises(ValueError):
        block(x)


def test_invalid_hidden_dimension():
    block = make_block()

    x = torch.randn(
        2,
        5,
        15,
    )

    with pytest.raises(ValueError):
        block(x)


def test_integer_input_rejected():
    block = make_block()

    x = torch.ones(
        2,
        5,
        16,
        dtype=torch.int64,
    )

    with pytest.raises(TypeError):
        block(x)

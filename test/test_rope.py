import math

import pytest
import torch

from tinyseek.model.rope import RotaryEmbedding


def test_output_shape() -> None:
    """RoPE 只改变数值，不改变 Tensor 形状。"""

    rope = RotaryEmbedding(dim=8)

    # [batch_size, num_heads, sequence_length, head_dim]
    x = torch.randn(2, 4, 16, 8)

    output = rope(x)

    assert output.shape == x.shape
    assert output.dtype == x.dtype


def test_position_zero_does_not_rotate() -> None:
    """位置为0时，旋转角度为0，输出应等于输入。"""

    rope = RotaryEmbedding(dim=8)

    x = torch.randn(1, 1, 1, 8)
    positions = torch.tensor([0])

    output = rope(x, positions)

    torch.testing.assert_close(
        output,
        x,
        rtol=1e-6,
        atol=1e-6,
    )


def test_known_two_dimensional_rotation() -> None:
    """检查一个可以手算的二维旋转。"""

    rope = RotaryEmbedding(dim=2)

    # shape: [batch, heads, sequence, head_dim]
    x = torch.tensor([
        [
            [
                [1.0, 0.0],
            ]
        ]
    ])

    # dim=2 时 inv_freq=[1]。
    # position=1，所以旋转角度是1弧度。
    positions = torch.tensor([1])

    output = rope(x, positions)

    expected = torch.tensor([
        [
            [
                [
                    math.cos(1.0),
                    math.sin(1.0),
                ]
            ]
        ]
    ])

    torch.testing.assert_close(
        output,
        expected,
        rtol=1e-6,
        atol=1e-6,
    )


def test_known_four_dimensional_rotation() -> None:
    """检查dim=4时快慢两个频率的旋转结果。"""

    rope = RotaryEmbedding(dim=4)

    x = torch.tensor([
        [
            [
                [1.0, 0.0, 1.0, 0.0],
            ]
        ]
    ])

    positions = torch.tensor([1])

    output = rope(x, positions)

    # dim=4、base=10000时：
    # 第一对频率为1
    # 第二对频率为0.01
    expected = torch.tensor([
        [
            [
                [
                    math.cos(1.0),
                    math.sin(1.0),
                    math.cos(0.01),
                    math.sin(0.01),
                ]
            ]
        ]
    ])

    torch.testing.assert_close(
        output,
        expected,
        rtol=1e-6,
        atol=1e-6,
    )


def test_rotation_preserves_vector_norm() -> None:
    """二维旋转应保持每个Q/K向量的模长。"""

    torch.manual_seed(0)

    rope = RotaryEmbedding(dim=8)
    x = torch.randn(2, 3, 5, 8)

    output = rope(x)

    input_norm = torch.linalg.vector_norm(
        x,
        dim=-1,
    )

    output_norm = torch.linalg.vector_norm(
        output,
        dim=-1,
    )

    torch.testing.assert_close(
        output_norm,
        input_norm,
        rtol=1e-5,
        atol=1e-6,
    )


def test_relative_position_property() -> None:
    """检查RoPE最重要的相对位置性质。

    (R(m)q)^T R(n)k = q^T R(n-m)k
    """

    torch.manual_seed(1)

    rope = RotaryEmbedding(dim=8)

    q = torch.randn(1, 1, 1, 8)
    k = torch.randn(1, 1, 1, 8)

    position_m = torch.tensor([2])
    position_n = torch.tensor([5])

    rotated_q = rope(q, position_m)
    rotated_k = rope(k, position_n)

    left = (
        rotated_q * rotated_k
    ).sum(dim=-1)

    relative_distance = 5 - 2

    relative_k = rope(
        k,
        torch.tensor([relative_distance]),
    )

    right = (
        q * relative_k
    ).sum(dim=-1)

    torch.testing.assert_close(
        left,
        right,
        rtol=1e-5,
        atol=1e-6,
    )


def test_backward() -> None:
    """检查梯度能否穿过RoPE传回输入。"""

    rope = RotaryEmbedding(dim=8)

    x = torch.randn(
        2,
        4,
        6,
        8,
        requires_grad=True,
    )

    output = rope(x)

    loss = output.square().mean()
    loss.backward()

    assert x.grad is not None
    assert x.grad.shape == x.shape
    assert torch.isfinite(x.grad).all()


def test_inv_freq_is_buffer_not_parameter() -> None:
    """inv_freq应是固定Buffer，而不是可训练参数。"""

    rope = RotaryEmbedding(dim=8)

    buffers = dict(rope.named_buffers())
    parameters = dict(rope.named_parameters())

    assert "inv_freq" in buffers
    assert "inv_freq" not in parameters
    assert rope.inv_freq.requires_grad is False


def test_odd_dimension_is_rejected() -> None:
    """奇数维无法两两分组，应该报错。"""

    with pytest.raises(ValueError):
        RotaryEmbedding(dim=7)


def test_wrong_input_dimension_is_rejected() -> None:
    """输入最后一维必须等于初始化时的dim。"""

    rope = RotaryEmbedding(dim=8)

    x = torch.randn(2, 4, 16, 10)

    with pytest.raises(ValueError):
        rope(x)


def test_wrong_positions_length_is_rejected() -> None:
    """每个token都必须有一个对应的位置。"""

    rope = RotaryEmbedding(dim=8)

    # 序列长度是5
    x = torch.randn(2, 4, 5, 8)

    # 但这里只提供了3个位置
    positions = torch.tensor([0, 1, 2])

    with pytest.raises(ValueError):
        rope(x, positions)


def test_positions_must_be_one_dimensional() -> None:
    """当前简化实现只接受一维positions。"""

    rope = RotaryEmbedding(dim=8)
    x = torch.randn(2, 4, 5, 8)

    positions = torch.tensor([
        [0, 1, 2, 3, 4]
    ])

    with pytest.raises(ValueError):
        rope(x, positions)


def test_integer_input_is_rejected() -> None:
    """RoPE处理的是浮点Q/K，而不是整数token编号。"""

    rope = RotaryEmbedding(dim=8)

    x = torch.ones(
        2,
        4,
        5,
        8,
        dtype=torch.int64,
    )

    with pytest.raises(TypeError):
        rope(x)

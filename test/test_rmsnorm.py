import pytest
import torch

from tinyseek.model.rmsnorm import RMSNorm


def reference_rmsnorm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """直接按照数学公式计算，作为参考答案。"""

    x_float = x.float()

    rms = torch.sqrt(
        x_float.pow(2).mean(dim=-1, keepdim=True) + eps
    )

    output = x_float / rms
    output = output * weight.float()

    return output.to(dtype=x.dtype)


def test_output_shape() -> None:
    """RMSNorm 不应改变输入形状。"""

    model = RMSNorm(hidden_size=4)
    x = torch.randn(2, 3, 4)

    output = model(x)

    assert output.shape == x.shape


def test_matches_reference() -> None:
    """自己的实现应当与直接公式计算一致。"""

    torch.manual_seed(0)

    model = RMSNorm(hidden_size=4, eps=1e-6)
    x = torch.randn(2, 3, 4)

    actual = model(x)

    expected = reference_rmsnorm(
        x=x,
        weight=model.weight,
        eps=model.eps,
    )

    torch.testing.assert_close(
        actual,
        expected,
        rtol=1e-5,
        atol=1e-6,
    )


def test_output_rms_close_to_one() -> None:
    """当 weight 全为 1 时，输出 RMS 应接近 1。"""

    model = RMSNorm(hidden_size=4, eps=1e-8)

    x = torch.tensor([
        [1.0, 2.0, 3.0, 4.0],
        [10.0, 20.0, 30.0, 40.0],
    ])

    output = model(x)

    output_rms = torch.sqrt(
        output.pow(2).mean(dim=-1)
    )

    torch.testing.assert_close(
        output_rms,
        torch.ones_like(output_rms),
        rtol=1e-5,
        atol=1e-5,
    )


def test_backward() -> None:
    """检查输入和可训练参数是否都能获得梯度。"""

    model = RMSNorm(hidden_size=4)

    x = torch.tensor(
        [[1.0, 2.0, 3.0, 4.0]],
        requires_grad=True,
    )

    output = model(x)

    loss = output.sum()
    loss.backward()

    assert x.grad is not None
    assert model.weight.grad is not None

    assert torch.isfinite(x.grad).all()
    assert torch.isfinite(model.weight.grad).all()


def test_wrong_hidden_size() -> None:
    """最后一维不匹配时应主动报错。"""

    model = RMSNorm(hidden_size=4)
    x = torch.randn(2, 3, 8)

    with pytest.raises(ValueError):
        model(x)

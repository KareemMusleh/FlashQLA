import pytest
import torch
import torch.nn.functional as F

from flash_qla.ops.fp8_linear import fp8_linear, quantize_linear_weight

pytestmark = pytest.mark.gpu


def rms(a, b):
    return ((a.double() - b.double()).norm() / b.double().norm().clamp_min(1e-30)).item()


@pytest.mark.parametrize("shape,n", [((33, 128), 48), ((2, 32, 256), 128)])
def test_fp8_forward_and_bf16_backward(shape, n):
    torch.manual_seed(715)
    x = torch.randn(shape, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    w = torch.randn(n, shape[-1], device="cuda", dtype=torch.bfloat16, requires_grad=True)
    prepared = quantize_linear_weight(w)
    y = fp8_linear(x, w, prepared)
    dy = torch.randn_like(y)
    dx, dw = torch.autograd.grad(y, (x, w), dy)
    # Independent quantization formula for the defined surrogate backward.
    scale = w.float().abs().amax(-1, keepdim=True).clamp_min(1e-12) / 448
    dequantized = ((w.float() / scale).to(torch.float8_e4m3fn).float() * scale).bfloat16()
    expected_dx = (dy.reshape(-1, n) @ dequantized).reshape_as(x)
    expected_dw = dy.reshape(-1, n).T @ x.reshape(-1, shape[-1])
    assert y.dtype == dx.dtype == dw.dtype == torch.bfloat16
    assert rms(y, F.linear(x, w)) < 0.05
    torch.testing.assert_close(dx, expected_dx, rtol=0, atol=0)
    torch.testing.assert_close(dw, expected_dw, rtol=0, atol=0)


def test_zero_activation_and_stale_weight():
    x = torch.zeros(17, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    w = torch.zeros(48, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    prepared = quantize_linear_weight(w)
    y = fp8_linear(x, w, prepared)
    dx, dw = torch.autograd.grad(y, (x, w), torch.ones_like(y))
    assert y.norm() == dx.norm() == dw.norm() == 0
    with torch.no_grad():
        w.add_(1)
    with pytest.raises(ValueError, match="stale"):
        fp8_linear(x, w, prepared)

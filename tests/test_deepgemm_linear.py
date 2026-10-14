import os

import pytest
import torch
import torch.nn.functional as F

from flash_qla.ops.deepgemm_linear import deepgemm_linear, load_ydt_blockwise, quantize_deepgemm_weight, quantize_activation_tma

pytestmark = pytest.mark.gpu


@pytest.fixture(scope="module")
def quantizer():
    pytest.importorskip("deep_gemm")
    return load_ydt_blockwise(os.environ.get("YDT_FP8_QUANTIZATION_SOURCE",
        "/home/kareemm/arcadia/ml/torch/ydt/core/ydt_core/fp8/quantization/blockwise.py"))


def independent_dequant(x, bm):
    m, k = x.shape
    padded = F.pad(x.float(), (0, 0, 0, (-m) % bm))
    blocks = padded.reshape(-1, bm, k // 128, 128)
    amax = blocks.abs().amax((1, 3), keepdim=True)
    multiplier = torch.where(amax > 0, torch.exp2(torch.floor(torch.log2(448 / amax))), 1)
    q = (blocks * multiplier).to(torch.float8_e4m3fn)
    return (q.float() / multiplier).reshape(-1, k)[:m]


@pytest.mark.parametrize("shape,n", [((33, 128), 48), ((2, 32, 256), 128)])
def test_deepgemm_forward_and_bf16_backward(quantizer, shape, n):
    torch.manual_seed(818)
    x = torch.randn(shape, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    w = torch.randn(n, shape[-1], device="cuda", dtype=torch.bfloat16, requires_grad=True)
    p = quantize_deepgemm_weight(w, quantizer)
    y = deepgemm_linear(x, w, p)
    dy = torch.randn_like(y)
    dx, dw = torch.autograd.grad(y, (x, w), dy)
    xd = independent_dequant(x.reshape(-1, shape[-1]), 1)
    wd = independent_dequant(w, 128)
    expected = (xd @ wd.T).bfloat16().reshape_as(y)
    error = (y.double()-expected.double()).norm()/expected.double().norm()
    assert error < .005
    assert y.dtype == dx.dtype == dw.dtype == torch.bfloat16
    torch.testing.assert_close(dx, (dy.reshape(-1,n) @ wd.bfloat16()).reshape_as(x), rtol=0, atol=0)
    torch.testing.assert_close(dw, dy.reshape(-1,n).T @ x.reshape(-1,shape[-1]), rtol=0, atol=0)


def test_zero_and_stale_weight(quantizer):
    x = torch.zeros(17,128,device="cuda",dtype=torch.bfloat16,requires_grad=True)
    w = torch.zeros(48,128,device="cuda",dtype=torch.bfloat16,requires_grad=True)
    p = quantize_deepgemm_weight(w, quantizer)
    y = deepgemm_linear(x,w,p)
    dx,dw = torch.autograd.grad(y,(x,w),torch.ones_like(y))
    assert y.norm() == dx.norm() == dw.norm() == 0
    with torch.no_grad():
        w.add_(1)
    with pytest.raises(ValueError,match="stale"):
        deepgemm_linear(x,w,p)


@pytest.mark.parametrize("m,k", [(17,128),(33,256),(256,6144)])
def test_direct_scale_layout_matches_ydt(quantizer,m,k):
    import deep_gemm

    torch.manual_seed(819)
    x = torch.randn(m,k,device="cuda",dtype=torch.bfloat16)
    x[0].zero_()
    q,scale = quantize_activation_tma(x)
    reference_q,reference_scale = quantizer.quant(x,(1,128))
    assert torch.equal(q.view(torch.uint8),reference_q.view(torch.uint8))
    torch.testing.assert_close(scale,reference_scale,rtol=0,atol=0)
    assert scale.stride() == (1, (m+3)//4*4)
    transformed = deep_gemm.transform_sf_into_required_layout(scale,m,k,(1,128,128),is_sfa=True)
    assert transformed.data_ptr() == scale.data_ptr()
    assert transformed.stride() == scale.stride()

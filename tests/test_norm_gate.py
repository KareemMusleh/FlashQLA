"""Numerical and layout coverage for the fused output gate."""
import pytest
import torch
import torch.nn.functional as F
from flash_qla.ops.gated_delta_rule.chunk.hopper.norm_gate import norm_gate_fwd


@pytest.mark.parametrize('dtype', [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize('strided', [False, True])
@pytest.mark.parametrize('eps', [1e-6, 1e-5])
@pytest.mark.parametrize('weight_fp32', [False, True])
@torch.no_grad()
def test_norm_gate(dtype, strided, eps, weight_fp32):
    torch.manual_seed(71)
    o = torch.randn(1, 67, 3, 128, device='cuda', dtype=dtype)
    o[:, 0] = 0
    o[:, 1] *= 1e-3
    z = torch.randn(67, 384 + (96 if strided else 0), device='cuda', dtype=dtype)[:, :384]
    w = torch.randn(128, device='cuda', dtype=torch.float32 if weight_fp32 else dtype)
    actual = norm_gate_fwd(o, z, w, eps)
    # Independent FP64 math, with the original activation-dtype boundaries.
    od = o.double()
    norm = (od * (od.square().mean(-1, keepdim=True) + eps).rsqrt() * w.double()).to(dtype)
    activated = F.silu(z.double()).to(dtype).view_as(o)
    expected = norm * activated
    relative_rms = ((actual.double()-expected.double()).norm()/expected.double().norm()).item()
    assert actual.isfinite().all()
    assert relative_rms < {torch.bfloat16: .004, torch.float16: .0006, torch.float32: 1e-6}[dtype], relative_rms
    torch_expected = F.rms_norm(o, (128,), w, eps).to(dtype) * F.silu(z).view_as(o)
    torch_error = ((actual.double()-torch_expected.double()).norm()/torch_expected.double().norm()).item()
    assert torch_error < {torch.bfloat16: .006, torch.float16: .001, torch.float32: 1e-6}[dtype], torch_error


def test_rejects_autograd():
    o = torch.zeros(1, 2, 3, 128, device='cuda', requires_grad=True)
    with pytest.raises(ValueError, match='forward-only'):
        norm_gate_fwd(o, torch.zeros(2, 384, device='cuda'), torch.ones(128, device='cuda'))


@pytest.mark.parametrize('strided',[False,True])
@pytest.mark.parametrize('weight_fp32',[False,True])
@torch.no_grad()
def test_compiled_norm_gate(strided,weight_fp32):
    from flash_qla.ops.gated_delta_rule.chunk.hopper.norm_gate import norm_gate_fwd_compiled
    torch.manual_seed(31)
    output=torch.randn(1,67,3,128,device='cuda',dtype=torch.bfloat16)
    output[:,0]=0
    gate=torch.randn(67,384+(96 if strided else 0),device='cuda',dtype=output.dtype)[:,:384]
    weight=torch.randn(128,device='cuda',dtype=torch.float32 if weight_fp32 else output.dtype)
    actual=norm_gate_fwd_compiled(output,gate,weight,1e-6)
    normalized=(output.double()*(output.double().square().mean(-1,keepdim=True)+1e-6).rsqrt()*weight.double()).bfloat16()
    expected=normalized*F.silu(gate.double()).bfloat16().view_as(output)
    rms=((actual.double()-expected.double()).norm()/expected.double().norm()).item()
    assert actual.isfinite().all() and rms<.004,rms

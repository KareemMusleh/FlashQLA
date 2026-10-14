import os
import pytest
import torch
import torch.nn.functional as F

from flash_qla.ops.deepgemm_linear import load_ydt_blockwise,quantize_deepgemm_weight,deepgemm_linear,shared_deepgemm_projections
from flash_qla.ops.conv_norm import causal_conv_silu_l2norm
from flash_qla.utils.math import l2norm_fwd

pytestmark=pytest.mark.gpu

def rms(a,b):
    return ((a.double()-b.double()).norm()/b.double().norm().clamp_min(1e-30)).item()

@pytest.mark.parametrize('combined',[False,True])
def test_shared_projections(combined):
    q=load_ydt_blockwise(os.environ.get('YDT_FP8_QUANTIZATION_SOURCE','/home/kareemm/arcadia/ml/torch/ydt/core/ydt_core/fp8/quantization/blockwise.py'))
    torch.manual_seed(905)
    x=torch.randn(1,64,128,device='cuda',dtype=torch.bfloat16,requires_grad=True)
    w=[torch.randn(n,128,device='cuda',dtype=torch.bfloat16,requires_grad=True) for n in (128,256,48)]
    p=[quantize_deepgemm_weight(t,q) for t in w]
    actual=shared_deepgemm_projections(x,w,p,combined)
    reference=[deepgemm_linear(x,wi,pi) for wi,pi in zip(w,p)]
    for a,b in zip(actual,reference):
        torch.testing.assert_close(a,b,rtol=0,atol=0)
    dy=[torch.randn_like(t) for t in actual]
    gradients=torch.autograd.grad(actual,(x,*w),dy)
    packed=torch.cat([t.reshape(64,-1) for t in dy],-1)
    wd=torch.cat([q.dequant(pi.data,pi.scale,(128,128)) for pi in p])
    if combined:
        torch.testing.assert_close(gradients[0],(packed@wd).reshape_as(x),rtol=0,atol=0)
    else:
        old=torch.autograd.grad(reference,(x,*w),dy)
        assert rms(gradients[0],old[0])<.01
    expected_dw=(packed.T@x.reshape(64,128)).split([wi.shape[0] for wi in w])
    for a,b in zip(gradients[1:],expected_dw):
        assert rms(a,b)<.001

@pytest.mark.parametrize('width',[2,3,4])
def test_conv_norm(width):
    from causal_conv1d import causal_conv1d_fn
    torch.manual_seed(906)
    x=torch.randn(1,73,256,device='cuda',dtype=torch.bfloat16,requires_grad=True)
    w=(torch.randn(256,width,device='cuda',dtype=torch.bfloat16)*.5).requires_grad_()
    actual=causal_conv_silu_l2norm(x,w)
    raw=causal_conv1d_fn(x.transpose(1,2),w,activation='silu').transpose(1,2).contiguous()
    expected,_=l2norm_fwd(raw.reshape(1,73,2,128))
    assert rms(actual,expected.reshape_as(actual))<.004
    dy=torch.randn_like(actual)
    ga=torch.autograd.grad(actual,(x,w),dy)
    ge=torch.autograd.grad(expected,(x,w),dy.reshape_as(expected))
    for a,b in zip(ga,ge):
        assert torch.isfinite(a).all() and rms(a,b)<.01

@pytest.mark.parametrize('cp',[False,True])
@pytest.mark.parametrize('v_first',[False,True])
def test_state_workspace_cleanup(cp,v_first):
    from flash_qla.ops.gated_delta_rule.chunk.hopper.prepare_h import fused_gdr_h
    t,h=256,2
    k=torch.zeros(1,t,h,128,device='cuda',dtype=torch.bfloat16)
    v=torch.randn_like(k)
    a=torch.eye(64,device='cuda',dtype=k.dtype)[None,:,None,:].expand(t//64,64,h,64).reshape(1,t,h,64).contiguous()
    b=torch.full((1,t,h),.5,device='cuda')
    g=torch.full_like(b,-.001).reshape(1,4,64,h).cumsum(2).reshape_as(b)
    h0=torch.randn(1,h,128,128,device='cuda')*.01
    cu=torch.tensor([0,t],device='cuda',dtype=torch.int32)
    warm=torch.full((1,h),4,device='cuda',dtype=torch.int32) if cp else None
    states,last,m=fused_gdr_h(k,v,a,g,b,initial_state=h0,cu_seqlens=cu,num_warmup_chunks=warm,
                             output_h=not cp,output_final_state=True,state_v_first=v_first)
    expected=h0*torch.exp(torch.tensor(-.256,device='cuda'))
    assert rms(last,expected.to(last.dtype))<.001
    if cp:
        expected_m=torch.eye(128,device='cuda')*torch.exp(torch.tensor(-.256,device='cuda'))
        assert rms(m,expected_m[None,None].expand_as(m).to(m.dtype))<.001
    else:
        assert m.norm()==0
        for c in range(4):
            expected=h0*torch.exp(torch.tensor(-.064*c,device='cuda'))
            assert rms(states[:,c],expected.bfloat16())<.001

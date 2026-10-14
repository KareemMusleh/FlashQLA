import os
import pytest
import torch
import torch.nn.functional as F

from causal_conv1d import causal_conv1d_fn
from flash_qla import chunk_gated_delta_rule
from flash_qla.ops.deepgemm_linear import load_ydt_blockwise,quantize_deepgemm_weight,deepgemm_linear
from flash_qla.ops.fp8_gdn_layer import fp8_gdn_layer
from flash_qla.ops.gated_delta_rule.chunk.hopper.training_norm_gate import training_norm_gate
from flash_qla.ops.gated_delta_rule.chunk.hopper.fast_fwd import _training_cp_context

pytestmark=pytest.mark.gpu

def rms(a,b):return ((a.double()-b.double()).norm()/b.double().norm().clamp_min(1e-30)).item()

@pytest.mark.parametrize('fused_conv',[False,True])
def test_direct_fp8_full_layer(fused_conv):
    torch.manual_seed(909)
    t,hidden,hk,hv=256,256,2,16
    x=torch.randn(t,hidden,device='cuda',dtype=torch.bfloat16,requires_grad=True)
    weights=[(torch.randn(n,hidden,device='cuda',dtype=torch.bfloat16)*.02).requires_grad_()
             for n in (hk*128,hk*128,hv*128,hv*128,hv,hv)]
    conv=[(torch.randn(n,4,device='cuda',dtype=torch.bfloat16)*.5).requires_grad_() for n in (hk*128,hk*128,hv*128)]
    al=torch.rand(hv,device='cuda',requires_grad=True)
    dt=torch.full((hv,),-6.9,device='cuda',requires_grad=True)
    nw=torch.ones(128,device='cuda',dtype=torch.bfloat16,requires_grad=True)
    wo=(torch.randn(hidden,hv*128,device='cuda',dtype=torch.bfloat16)*.02).requires_grad_()
    params=(*weights,*conv,al,dt,nw,wo)
    quantizer=load_ydt_blockwise(os.environ.get('YDT_FP8_QUANTIZATION_SOURCE','/home/kareemm/arcadia/ml/torch/ydt/core/ydt_core/fp8/quantization/blockwise.py'))
    prepared=[quantize_deepgemm_weight(w,quantizer) for w in (*weights,wo)]
    context=_training_cp_context(x.device,t,2)
    p=[deepgemm_linear(x,w,q) for w,q in zip(weights,prepared)]
    c=[causal_conv1d_fn(z[None].transpose(1,2),w,activation='silu').transpose(1,2).contiguous()
       for z,w in zip(p[:3],conv)]
    g=(-al.exp()*F.softplus(p[4].float()+dt))[None]
    beta=p[5].float().sigmoid()[None]
    o,_=chunk_gated_delta_rule(c[0].reshape(1,t,hk,128),c[1].reshape(1,t,hk,128),c[2].reshape(1,t,hv,128),
        g,beta,use_qk_l2norm_in_kernel=True,cp_context=context,enable_fwd_cp_cache=True)
    gated=training_norm_gate(o,p[3].reshape_as(o),nw)
    expected=deepgemm_linear(gated.reshape(t,-1),wo,prepared[-1])
    actual=fp8_gdn_layer(x,params,prepared,context,fused_conv=fused_conv)
    assert rms(actual,expected)<.02
    if not fused_conv:torch.testing.assert_close(actual,expected,rtol=0,atol=0)
    upstream=torch.randn_like(actual)
    a=torch.autograd.grad(actual,(x,*params),upstream)
    b=torch.autograd.grad(expected,(x,*params),upstream)
    for ga,gb in zip(a,b):
        assert ga.dtype==gb.dtype and torch.isfinite(ga).all()
        assert rms(ga,gb)<.02

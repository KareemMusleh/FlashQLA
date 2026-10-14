"""Aliasing/lifetime coverage for backward-only projection workspace reuse."""
import pytest
import torch

from flash_qla.ops.gdn_layer import direct_gdn_layer, norm_gate_backward_into
from flash_qla.ops.gated_delta_rule.chunk.hopper.kkt_solve import kkt_solve
from test_direct_gdn_layer import fixture, rms


@pytest.mark.parametrize("hk,hv,partitions",[(2,4,2),(2,2,1),(1,8,2)])
def test_layer_workspace(hk,hv,partitions):
    args=fixture(hk=hk,hv=hv)
    x,ws,cs,al,dt,nw,wo=args
    params=(x,*ws,*cs,al,dt,nw,wo)
    saved=[]
    def pack(t):
        saved.append(t.untyped_storage().nbytes())
        return t
    with torch.autograd.graph.saved_tensors_hooks(pack,lambda t:t):
        a=direct_gdn_layer(*args,key_heads=hk,value_heads=hv,partitions=partitions,
                           reuse_gradient_workspace=False)
    old_saved=saved[:];saved.clear()
    with torch.autograd.graph.saved_tensors_hooks(pack,lambda t:t):
        b=direct_gdn_layer(*args,key_heads=hk,value_heads=hv,partitions=partitions)
    assert old_saved==saved
    torch.testing.assert_close(a,b,rtol=0,atol=0)
    dy=torch.randn_like(a)
    ga=torch.autograd.grad(a,params,dy)
    gb=torch.autograd.grad(b,params,dy,retain_graph=True)
    errors=[rms(u,v) for u,v in zip(ga,gb)]
    assert max(errors)<.001,errors
    again=torch.autograd.grad(b,params,dy)
    assert max(rms(u,v) for u,v in zip(gb,again))<.001


@pytest.mark.parametrize("varlen",[False,True])
def test_inverse_writes_only_its_reserved_columns(varlen):
    torch.manual_seed(12)
    k=torch.nn.functional.normalize(torch.randn(1,128,2,128,device="cuda"),dim=-1).bfloat16()
    beta=torch.rand(1,128,4,device="cuda")
    buf=torch.full((128,1544),7.,device="cuda",dtype=k.dtype)
    out=buf[:,:256].view(1,128,4,64)
    cu=torch.tensor([0,64,128],device="cuda",dtype=torch.int32) if varlen else None
    ref=kkt_solve(k,beta,cu_seqlens=cu)
    actual=kkt_solve(k,beta,cu_seqlens=cu,out=out)
    assert actual.data_ptr()==out.data_ptr()
    torch.testing.assert_close(actual,ref,rtol=0,atol=0)
    assert (buf[:,256:]==7).all()


def test_do_and_inverse_regions_are_disjoint():
    t,h,d=67,4,128
    o=torch.randn(1,t,h,d,device="cuda",dtype=torch.bfloat16)
    z=torch.randn(t,h*d,device="cuda",dtype=o.dtype)
    w=torch.randn(d,device="cuda",dtype=o.dtype)
    dy=torch.randn_like(o)
    buf=torch.full((t,1544),7.,device="cuda",dtype=o.dtype)
    do=buf[:,256:768].view_as(o)
    dz=buf[:,1024:1536]
    refdz=torch.empty_like(z)
    refdo,refdw=norm_gate_backward_into(o,z,w,dy,refdz,1e-6)
    actual,dw=norm_gate_backward_into(o,z,w,dy,dz,1e-6,do_out=do)
    torch.testing.assert_close(actual,refdo,rtol=0,atol=0)
    torch.testing.assert_close(dw,refdw,rtol=0,atol=0)
    torch.testing.assert_close(dz,refdz,rtol=0,atol=0)
    assert (buf[:,:256]==7).all() and (buf[:,768:1024]==7).all() and (buf[:,1536:]==7).all()

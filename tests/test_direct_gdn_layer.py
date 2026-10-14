import pytest
import torch
import torch.nn.functional as F
from causal_conv1d import causal_conv1d_fn

from flash_qla import chunk_gated_delta_rule
from flash_qla.ops.gdn_layer import direct_gdn_layer, norm_gate_backward_into, gate_backward_into


def rms(a, b):
    return ((a.double() - b.double()).norm() / b.double().norm().clamp_min(1e-20)).item()


def fixture(t=256, hk=2, hv=4):
    torch.manual_seed(129)
    sizes = (hk*128, hk*128, hv*128, hv*128, hv, hv)
    def bf(shape, scale=1):
        return (torch.randn(shape, device="cuda") * scale).bfloat16().requires_grad_()
    x = bf((t, 64))
    ws = tuple(bf((s, 64), 1/8) for s in sizes)
    cs = tuple(bf((s, 4), .5) for s in sizes[:3])
    al = torch.zeros(hv, device="cuda", requires_grad=True)
    dt = torch.full((hv,), -5., device="cuda", requires_grad=True)
    nw = bf((128,), .25)
    wo = bf((64, hv*128), 1/(hv*128)**.5)
    return x, ws, cs, al, dt, nw, wo


def reference(args, hk=2, hv=4, eps=1e-6, partitions=2):
    x, ws, cs, al, dt, nw, wo = args
    p = [F.linear(x, w) for w in ws]
    c = [causal_conv1d_fn(z[None].transpose(1, 2), w, activation="silu").transpose(1, 2)
         for z, w in zip(p, cs)]
    g = (-al.exp() * F.softplus(p[4].float() + dt))[None]
    beta = p[5].float().sigmoid()[None]
    o = chunk_gated_delta_rule(c[0].view(1,-1,hk,128), c[1].view(1,-1,hk,128),
                              c[2].view(1,-1,hv,128), g, beta,
                              fwd_backend="optimized", bwd_backend="default",
                              enable_fwd_cp_cache=False, fwd_partitions=partitions,
                              fwd_decay_threshold=None, use_qk_l2norm_in_kernel=True)[0]
    gated = F.rms_norm(o, (128,), nw, eps) * F.silu(p[3].view_as(o))
    return F.linear(gated.view(x.shape[0], -1), wo)


@pytest.mark.parametrize("hk,hv,partitions", [(2,4,2), (2,2,1)])
@pytest.mark.parametrize("output_gate_mode", ["eager", "fused"])
def test_full_layer(hk, hv, partitions, output_gate_mode):
    args = fixture(hk=hk, hv=hv)
    x, ws, cs, al, dt, nw, wo = args
    params = (x,*ws,*cs,al,dt,nw,wo)
    saved = []
    def pack(t):
        saved.append((t.untyped_storage().data_ptr(), t.untyped_storage().nbytes()))
        return t
    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t:t):
        a = direct_gdn_layer(*args, key_heads=hk, value_heads=hv, partitions=partitions,
                             output_gate_mode=output_gate_mode)
    direct_storages = dict(saved)
    saved.clear()
    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t:t):
        b = reference(args,hk,hv,partitions=partitions)
    assert sum(direct_storages.values()) <= sum(dict(saved).values())
    if output_gate_mode == "eager":
        torch.testing.assert_close(a,b,rtol=0,atol=0)
    else:
        assert rms(a,b) < .003
    upstream = torch.randn_like(a)
    ga = torch.autograd.grad(a,params,upstream,retain_graph=True)
    gb = torch.autograd.grad(b,params,upstream)
    assert all(torch.isfinite(g).all() for g in ga)
    errors = [rms(u,v) for u,v in zip(ga,gb)]
    assert max(errors) < .02, errors
    # Each backward owns a fresh workspace; retain_graph calls do not share state.
    again = torch.autograd.grad(a,params,upstream)
    for u,v in zip(again,ga):
        torch.testing.assert_close(u,v,rtol=0,atol=0)


@pytest.mark.parametrize("t", [1,67])
def test_norm_gate_strided_destination(t):
    h,d=4,128
    o=torch.randn(1,t,h,d,device="cuda",dtype=torch.bfloat16,requires_grad=True)
    z=torch.randn(t,h*d,device="cuda",dtype=torch.bfloat16,requires_grad=True)
    w=torch.randn(d,device="cuda",dtype=torch.bfloat16,requires_grad=True)
    dy=torch.randn_like(o)
    y=F.rms_norm(o,(d,),w,1e-5)*F.silu(z.view_as(o))
    ref=torch.autograd.grad(y,(o,z,w),dy)
    buf=torch.full((t,h*d+32),123.,device="cuda",dtype=torch.bfloat16)
    dz=buf[:,16:16+h*d]
    do,dw=norm_gate_backward_into(o,z,w,dy,dz,1e-5)
    assert max(rms(a,b) for a,b in zip((do,dz,dw),ref)) < .015
    assert (buf[:,:16]==123).all() and (buf[:,-16:]==123).all()
    # Mathematical FP64 reference independent of fused or eager RMSNorm.
    od,zd,wd=[v.detach().double().requires_grad_() for v in (o,z,w)]
    yd=od * (od.square().mean(-1,keepdim=True)+1e-5).rsqrt() * wd
    yd=yd*(zd.view_as(od).sigmoid()*zd.view_as(od))
    gd=torch.autograd.grad(yd,(od,zd,wd),dy.double())
    assert max(rms(a,b) for a,b in zip((do,dz,dw),gd)) < .015


def test_gate_strided_destination():
    t,h=129,4
    a=torch.randn(t,h,device="cuda",dtype=torch.bfloat16,requires_grad=True)
    b=torch.randn_like(a,requires_grad=True)
    al=torch.randn(h,device="cuda",requires_grad=True)
    dt=torch.randn(h,device="cuda",requires_grad=True)
    with torch.no_grad(): a[0]=torch.tensor([30.,-30.,0.,-1.],device="cuda")
    g=-al.exp()*F.softplus(a.float()+dt)
    beta=b.float().sigmoid()
    dg,db=torch.randn_like(g),torch.randn_like(beta)
    ref=torch.autograd.grad((g,beta),(a,b,al,dt),(dg,db))
    buf=torch.full((t,32),123.,device="cuda",dtype=torch.bfloat16)
    da,dbeta=buf[:,8:12],buf[:,16:20]
    dal,ddt=gate_backward_into(a,b,al,dt,dg,db,da,dbeta)
    assert max(rms(u,v) for u,v in zip((da,dbeta,dal,ddt),ref)) < .005
    assert (buf[:,:8]==123).all() and (buf[:,-12:]==123).all()


def test_destination_offsets_beyond_int32():
    # The 128K Qwen layout exceeds 2**31 BF16 elements. Large destination
    # pitches must use 64-bit pointer arithmetic even with a single norm head.
    t, stride = 131072, 16480
    buf = torch.empty((t,stride),device="cuda",dtype=torch.bfloat16)
    o = torch.ones((1,t,1,128),device="cuda",dtype=torch.bfloat16)
    z = torch.ones((t,128),device="cuda",dtype=torch.bfloat16)
    w = torch.ones(128,device="cuda",dtype=torch.bfloat16)
    dz = buf[:,:128]
    do,dw = norm_gate_backward_into(o,z,w,o,dz,1e-6)
    small_buf = torch.empty((1,128),device="cuda",dtype=torch.bfloat16)
    small_do,_ = norm_gate_backward_into(o[:,:1],z[:1],w,o[:,:1],small_buf,1e-6)
    assert (dz==small_buf).all()
    assert (do==small_do).all() and torch.isfinite(dw).all()
    h = 48
    a = torch.zeros((t,h),device="cuda",dtype=torch.bfloat16)
    al = torch.zeros(h,device="cuda")
    upstream = torch.ones((t,h),device="cuda")
    da,db = buf[:,-96:-48],buf[:,-48:]
    dal,ddt = gate_backward_into(a,a,al,al,upstream,upstream,da,db)
    assert (da==-.5).all() and (db==.25).all()
    torch.testing.assert_close(ddt,al.new_full((h,),-t*.5),rtol=1e-6,atol=0)
    torch.testing.assert_close(dal,al.new_full((h,),-t*torch.log(torch.tensor(2.)).item()),rtol=1e-6,atol=0)


@pytest.mark.parametrize("t",[67,131072])
def test_convolution_strided_destination(t):
    from causal_conv1d.cpp_functions import causal_conv1d_bwd_function
    torch.manual_seed(32)
    x=torch.randn(1,128,t,device="cuda",dtype=torch.bfloat16).transpose(1,2).contiguous().transpose(1,2).requires_grad_()
    w=torch.randn(128,4,device="cuda",dtype=torch.bfloat16,requires_grad=True)
    y=causal_conv1d_fn(x,w,activation="silu")
    dy=torch.randn_like(y)
    ref=torch.autograd.grad(y,(x,w),dy)
    buffer=torch.empty((t,16480),device="cuda",dtype=torch.bfloat16)
    destination=buffer[:,256:384][None].transpose(1,2)
    dx,dw,_,_=causal_conv1d_bwd_function(x,w,None,dy,None,None,None,destination,False,True)
    assert dx.data_ptr()==destination.data_ptr() and dx.stride()==destination.stride()
    assert rms(dx,ref[0]) < .001 and rms(dw,ref[1]) < .001

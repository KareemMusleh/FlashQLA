"""Check fused causal convolution and Q/K normalization against FP64 math."""
import pytest
import torch
import torch.nn.functional as F

from flash_qla.ops.gated_delta_rule.chunk.hopper.prepare_qkv import prepare_qkv_fwd


@pytest.mark.parametrize('dtype', [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize('width', [2, 3, 4])
@pytest.mark.parametrize('packed', [False, True])
@torch.no_grad()
def test_prepare_qkv(dtype, width, packed):
    torch.manual_seed(29)
    t, hk, hv, d = 67, 2, 6, 128
    sizes = (hk*d, hk*d, hv*d)
    if packed:
        source = torch.randn(t, sum(sizes)+96, device='cuda', dtype=dtype)
        projections = source[:, :sum(sizes)].split(sizes, dim=-1)
    else:
        projections = tuple(torch.randn(t, c, device='cuda', dtype=dtype) for c in sizes)
    weights = tuple(torch.randn(c, width, device='cuda', dtype=dtype)/width**.5 for c in sizes)
    # Verify zero-history boundary and tiny/zero normalization vectors.
    for x in projections:
        x[:2, :128] = 0
        x[2:4, :128] *= 1e-4
    actual = prepare_qkv_fwd(projections, weights, hk, hv)
    for index, (x, w, result, heads) in enumerate(zip(projections, weights, actual, (hk,hk,hv))):
        convolved = F.conv1d(x.double().T[None], w.double()[:, None],
                             padding=width-1, groups=x.shape[1])[..., :t]
        reference = F.silu(convolved).transpose(1, 2).to(dtype).reshape(1,t,heads,d)
        if index < 2:
            f = reference.double()
            reference = (f * (f.square().sum(-1, keepdim=True)+1e-6).rsqrt()).to(dtype)
        error = ((result.double()-reference.double()).norm()/reference.double().norm()).item()
        assert result.is_contiguous() and result.isfinite().all()
        assert error < {torch.bfloat16:.004, torch.float16:.0007, torch.float32:2e-6}[dtype], error
        torch.testing.assert_close(result[:, :1, :1], reference[:, :1, :1], atol=0, rtol=0)


def test_rejects_autograd():
    x = torch.zeros(2,128,device='cuda',requires_grad=True)
    with pytest.raises(ValueError, match='forward-only'):
        prepare_qkv_fwd((x,x,x),(torch.ones(128,4,device='cuda'),)*3,1,1)


@pytest.mark.parametrize('dtype',[torch.bfloat16,torch.float16,torch.float32])
@pytest.mark.parametrize('packed',[False,True])
@torch.no_grad()
def test_fused_gates(dtype,packed):
    torch.manual_seed(19)
    t,hk,hv=47,2,6
    sizes=(hk*128,hk*128,hv*128,hv*128,hv,hv)
    if packed:
        source=torch.randn(t,sum(sizes),device='cuda',dtype=dtype)
        p=source.split(sizes,dim=-1)
    else:
        p=tuple(torch.randn(t,c,device='cuda',dtype=dtype) for c in sizes)
    p[4][:6]=torch.tensor([-100,-30,-20,0,30,100],device='cuda',dtype=dtype)[:,None]
    p[5][:6]=p[4][:6]
    w=tuple(torch.randn(c,4,device='cuda',dtype=dtype) for c in sizes[:3])
    al=torch.rand(hv,device='cuda')*2
    dt=torch.linspace(-7,-1,hv,device='cuda')
    qkv,gates=prepare_qkv_fwd(p,w,hk,hv,gate_biases=(al,dt))
    plain=prepare_qkv_fwd(p,w,hk,hv)
    for a,b in zip(qkv,plain):torch.testing.assert_close(a,b,rtol=0,atol=0)
    # FP64 oracle includes rounding of alpha+dt performed by the original FP32 path.
    x=(p[4].float()+dt).double()
    expected_g=(-al.double().exp()*F.softplus(x)).float()[None]
    expected_beta=p[5].double().sigmoid().float()[None]
    for actual,expected in zip(gates,(expected_g,expected_beta)):
        assert actual.dtype==torch.float32 and actual.is_contiguous() and actual.isfinite().all()
        torch.testing.assert_close(actual,expected,rtol=3e-6,atol=1e-38)

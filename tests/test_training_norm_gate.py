import pytest
import torch
import torch.nn.functional as F

from flash_qla.ops.gated_delta_rule.chunk.hopper.training_norm_gate import training_norm_gate

pytestmark = pytest.mark.gpu


def rms(a, b):
    return ((a.double() - b.double()).norm() / b.double().norm().clamp_min(1e-30)).item()


@pytest.mark.parametrize("shape", [(73, 128), (2, 31, 4, 128), (256, 64)])
@pytest.mark.parametrize("weight_dtype", [torch.bfloat16, torch.float32])
def test_training_norm_gate(shape, weight_dtype):
    torch.manual_seed(613)
    inputs = [torch.randn(shape, device="cuda", dtype=torch.bfloat16).requires_grad_() for _ in range(2)]
    weight = torch.randn(shape[-1], device="cuda", dtype=weight_dtype, requires_grad=True)
    inputs.append(weight)
    reference_inputs = [t.detach().clone().requires_grad_() for t in inputs]
    do = torch.randn_like(inputs[0])
    output = training_norm_gate(*inputs)
    expected = F.rms_norm(reference_inputs[0], (shape[-1],), reference_inputs[2], eps=1e-6) * F.silu(reference_inputs[1])
    assert output.dtype == expected.dtype
    assert rms(output, expected) < 0.006
    actual_grad = torch.autograd.grad(output, inputs, do)
    expected_grad = torch.autograd.grad(expected, reference_inputs, do)
    for a, b in zip(actual_grad, expected_grad):
        assert a.dtype == b.dtype and torch.isfinite(a).all()
        assert rms(a, b) < 0.008


def test_zero_weight_and_noncontiguous_upstream():
    x = torch.randn(17, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    z = torch.randn_like(x, requires_grad=True)
    w = torch.zeros(128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    do = torch.randn(128, 17, device="cuda", dtype=torch.bfloat16).T
    out = training_norm_gate(x, z, w)
    dx, dz, dw = torch.autograd.grad(out, (x, z, w), do)
    assert out.norm() == 0 and dx.norm() == 0 and dz.norm() == 0
    assert dw.norm() > 0 and torch.isfinite(dw).all()


def test_independent_fp64_and_large_weight_reduction():
    torch.manual_seed(614)
    inputs = [torch.randn(8193, 128, device="cuda", dtype=torch.bfloat16).requires_grad_() for _ in range(2)]
    inputs.append(torch.randn(128, device="cuda", dtype=torch.bfloat16, requires_grad=True))
    reference = [t.detach().double().requires_grad_() for t in inputs]
    do = torch.randn_like(inputs[0])
    actual = training_norm_gate(*inputs)
    x, z, w = reference
    # Independent formula; autograd differentiates FP64 arithmetic throughout.
    expected = x * (x.square().mean(-1, keepdim=True) + 1e-6).rsqrt() * w * (z * z.sigmoid())
    assert rms(actual, expected) < 0.006
    gradients = torch.autograd.grad(actual, inputs, do)
    expected_gradients = torch.autograd.grad(expected, reference, do.double())
    for a, b in zip(gradients, expected_gradients):
        assert torch.isfinite(a).all()
        assert rms(a, b) < 0.008


def test_small_activations_with_epsilon():
    torch.manual_seed(618)
    x = (torch.randn(129, 128, device="cuda", dtype=torch.bfloat16) * 0.001).requires_grad_()
    z = torch.randn_like(x, requires_grad=True)
    weight = torch.ones(128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    reference = [t.detach().clone().requires_grad_() for t in (x, z, weight)]
    do = torch.randn_like(x)
    actual = training_norm_gate(x, z, weight)
    expected = F.rms_norm(reference[0], (128,), reference[2], eps=1e-6) * F.silu(reference[1])
    assert rms(actual, expected) < 0.006
    for a, b in zip(torch.autograd.grad(actual, (x, z, weight), do),
                    torch.autograd.grad(expected, reference, do)):
        assert torch.isfinite(a).all()
        assert rms(a, b) < 0.008


def test_pitched_gradient_offsets_above_int32():
    from flash_qla.ops.gated_delta_rule.chunk.hopper.training_norm_gate import _backward

    # The 128k packed projection buffer exceeds 2**31 BF16 elements. Use
    # only 17 input rows with a large destination pitch to isolate indexing.
    rows,pitch=17,(1<<27)+128
    x=torch.ones(rows,128,device="cuda",dtype=torch.bfloat16)
    z=torch.zeros_like(x)
    dy=torch.ones_like(x)
    r=torch.ones(rows,device="cuda")
    w=torch.ones(128,device="cuda",dtype=x.dtype)
    dx=torch.empty_like(x)
    storage=torch.empty((rows,pitch),device="cuda",dtype=x.dtype)
    destination=storage[:,:128]
    partial=torch.empty((2,128),device="cuda")
    _backward[(2,)](x,z,w,x,z,r,dy,dx,destination,partial,rows,128,128,16,
                    HEADS=1,DZ_PITCH=pitch,num_warps=4,enable_fp_fusion=False)
    torch.testing.assert_close(destination,torch.full_like(x,.5),rtol=0,atol=0)
    assert dx.norm()==0

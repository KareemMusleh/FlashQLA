import pytest
import torch
import torch.nn.functional as F

from flash_qla.ops.input_projection import combined_input_projection


@pytest.mark.parametrize("shape", [(7, 5), (2, 7, 5)])
@pytest.mark.parametrize("unused", [False, True])
@pytest.mark.parametrize("combine_weight_gradients", [False, True])
def test_projection_gradients_and_saved_tensors(shape, unused, combine_weight_gradients):
    torch.manual_seed(82)
    x = torch.randn(shape, dtype=torch.float64, requires_grad=True)
    weights = [torch.randn(n, 5, dtype=torch.float64, requires_grad=True)
               for n in (3, 3, 7, 7, 2, 2)]
    saved = []
    with torch.autograd.graph.saved_tensors_hooks(lambda t: saved.append(t) or t,
                                                lambda t: t):
        actual = combined_input_projection(x, *weights, combine_weight_gradients=combine_weight_gradients)
    assert [t.data_ptr() for t in saved] == [t.data_ptr() for t in (x, *weights)]
    reference = tuple(F.linear(x, w) for w in weights)
    for a, b in zip(actual, reference):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    selected = slice(0, 1) if unused else slice(None)
    upstream = tuple(torch.randn_like(y) for y in actual[selected])
    ga = torch.autograd.grad(actual[selected], (x, *weights), upstream, allow_unused=True)
    gb = torch.autograd.grad(reference[selected], (x, *weights), upstream, allow_unused=True)
    for i, (a, b) in enumerate(zip(ga, gb)):
        if b is None:
            assert a is None
        else:
            torch.testing.assert_close(a, b, rtol=1e-12, atol=1e-12)


def test_optimizer_updates_original_weights():
    torch.manual_seed(42)
    x = torch.randn(8, 4, dtype=torch.float64, requires_grad=True)
    weights = [torch.nn.Parameter(torch.randn(n, 4, dtype=torch.float64)) for n in (3, 6)]
    refs = [torch.nn.Parameter(w.detach().clone()) for w in weights]
    optimizers = [torch.optim.SGD(ws, lr=.03) for ws in (weights, refs)]
    for _ in range(3):
        for ws, opt, combined in zip((weights, refs), optimizers, (True, False)):
            opt.zero_grad()
            outputs = combined_input_projection(x, *ws) if combined else [F.linear(x, w) for w in ws]
            sum(y.square().mean() for y in outputs).backward()
            opt.step()
        for w, ref in zip(weights, refs):
            torch.testing.assert_close(w, ref, rtol=1e-12, atol=1e-12)


def test_frozen_weights_and_gradcheck():
    x = torch.randn(2, 3, dtype=torch.float64, requires_grad=True)
    weights = [torch.randn(2, 3, dtype=torch.float64, requires_grad=True),
               torch.randn(1, 3, dtype=torch.float64)]
    assert torch.autograd.gradcheck(combined_input_projection, (x, *weights))


@pytest.mark.parametrize("autocast", [False, True])
def test_cuda_low_precision(autocast):
    torch.manual_seed(82)
    dtype = torch.float32 if autocast else torch.bfloat16
    x = torch.randn(64, 128, device="cuda", dtype=dtype, requires_grad=True)
    weights = [torch.randn(n, 128, device="cuda", dtype=dtype, requires_grad=True)
               for n in (128, 128, 384, 384, 3, 3)]
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast):
        a = combined_input_projection(x, *weights)
        b = tuple(F.linear(x, w) for w in weights)
    upstream = tuple(torch.randn_like(y) for y in a)
    ga = torch.autograd.grad(a, (x, *weights), upstream)
    gb = torch.autograd.grad(b, (x, *weights), upstream)
    for aa, bb in zip(a, b):
        torch.testing.assert_close(aa, bb, rtol=0, atol=0)
    # Combined dX rounds after one reduction; independent dX contributions round
    # separately. Compare both against FP64 instead of requiring BF16 equality.
    dx_ref = sum(g.double() @ w.double() for g, w in zip(upstream, weights))
    assert (ga[0].double() - dx_ref).norm() / dx_ref.norm() < .008
    for aa, bb in zip(ga[1:], gb[1:]):
        torch.testing.assert_close(aa, bb, rtol=.01, atol=.02)

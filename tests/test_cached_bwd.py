"""Training reuse of forward preparation, checked against an FP64 recurrence."""
import pytest
import torch

from flash_qla import chunk_gated_delta_rule
from flash_qla.ops.gated_delta_rule.chunk.hopper.local_precompute import local_inverse, local_precompute
from test_implicit_bwd import recurrence, relative_rms

pytestmark = [pytest.mark.gpu, pytest.mark.hopper]


@torch.no_grad()
def test_cached_inverse_preserves_forward():
    torch.manual_seed(171)
    k = torch.nn.functional.normalize(torch.randn(1, 256, 2, 128, device="cuda"), dim=-1).bfloat16()
    beta = torch.rand(1, 256, 4, device="cuda")
    beta[..., 0], beta[..., 1] = 0, 1
    g = -torch.rand_like(beta) / 16
    ab, gc = local_precompute(k, beta, g)
    cached_ab, cached_gc, ar = local_precompute(k, beta, g, return_inverse=True)
    torch.testing.assert_close(ab, cached_ab, rtol=0, atol=0)
    torch.testing.assert_close(gc, cached_gc, rtol=0, atol=0)
    torch.testing.assert_close(ar, local_inverse(k, beta), rtol=0, atol=0)


@pytest.mark.parametrize("parts,v_first,normalize,decay,scale,terminal_only", [
    (1, False, False, 0.0, 0.2, False),
    (2, False, True, 0.01, 0.2, False),
    (4, True, True, 0.08, 0.2, False),
    (2, True, False, 0.0, 0.2, True),
    (4, False, False, 0.04, 0.0, False),
])
def test_cached_backward_against_fp64(monkeypatch, parts, v_first, normalize, decay, scale, terminal_only):
    import flash_qla.ops.gated_delta_rule.chunk as chunk

    torch.manual_seed(172)
    q, k = [torch.nn.functional.normalize(torch.randn(1, 256, 2, 128, device="cuda"), dim=-1).bfloat16()
            for _ in range(2)]
    v = torch.randn(1, 256, 4, 128, device="cuda", dtype=torch.bfloat16)
    g = -torch.rand(1, 256, 4, device="cuda") * decay
    beta = torch.rand_like(g)
    beta[..., 0], beta[..., 1] = 0, 1
    initial = torch.randn(1, 4, 128, 128, device="cuda") * 0.1
    inputs = [x.requires_grad_() for x in (q, k, v, g, beta, initial)]
    kwargs = dict(fwd_backend="optimized", fwd_partitions=parts,
                  auto_cp=parts > 1, force_intra_cp=parts > 1, scale=scale,
                  use_qk_l2norm_in_kernel=normalize, state_v_first=v_first,
                  initial_state=initial, output_final_state=True)
    do = torch.zeros_like(v) if terminal_only else torch.randn_like(v)
    dht = torch.randn_like(initial) * 0.1
    output, final = chunk_gated_delta_rule(*inputs[:5], bwd_backend="optimized", **kwargs)
    with torch.no_grad():
        expected_output, expected_final = chunk_gated_delta_rule(*inputs[:5], **kwargs)
    torch.testing.assert_close(output, expected_output, rtol=0, atol=0)
    torch.testing.assert_close(final, expected_final, rtol=0, atol=0)

    def unexpected_inverse(*args, **kwargs):
        pytest.fail("Cached backward must not rebuild the local inverse")

    monkeypatch.setattr(chunk, "kkt_solve", unexpected_inverse)
    actual = torch.autograd.grad((output, final), inputs, (do, dht))
    ref_inputs = [x.double().detach().requires_grad_() for x in inputs]
    rq, rk = ref_inputs[:2]
    if normalize:
        rq, rk = torch.nn.functional.normalize(rq, dim=-1), torch.nn.functional.normalize(rk, dim=-1)
    ref_output, ref_final = recurrence(rq, rk, *ref_inputs[2:], scale, v_first)
    ref_loss = (ref_output * do.double()).sum() + (ref_final * dht.double()).sum()
    expected = torch.autograd.grad(ref_loss, ref_inputs)
    for name, a, b in zip(("dq", "dk", "dv", "dg", "db", "dh"), actual, expected):
        assert torch.isfinite(a).all(), name
        if b.norm() == 0:
            assert a.norm() == 0, name
        else:
            assert relative_rms(a, b) < 0.035, (name, relative_rms(a, b))
    assert relative_rms(actual[4][..., 0], expected[4][..., 0]) < 0.035


@pytest.mark.parametrize("scale", [None, 0.0])
def test_default_forward_fallback(monkeypatch, scale):
    import flash_qla.ops.gated_delta_rule.chunk as chunk

    torch.manual_seed(173)
    q = torch.randn(1, 64, 1, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True) * 0.03
    k, v = torch.randn_like(q) * 0.03, torch.randn_like(q)
    g = torch.full((1, 64, 1), -0.01, device="cuda")
    beta = torch.full_like(g, 0.5)
    called = []
    original = chunk.kkt_solve

    def inverse(*args, **kwargs):
        called.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(chunk, "kkt_solve", inverse)
    out, _ = chunk_gated_delta_rule(q, k, v, g, beta, auto_cp=False, bwd_backend="optimized", scale=scale)
    grad, = torch.autograd.grad(out, (q,), torch.randn_like(out))
    assert called == [True] and torch.isfinite(grad).all()
    if scale == 0:
        assert out.norm() == 0 and grad.norm() == 0


def test_unknown_backward_backend():
    q = torch.zeros(1, 64, 1, 128, device="cuda", dtype=torch.bfloat16)
    g = torch.zeros(1, 64, 1, device="cuda")
    with pytest.raises(ValueError, match="bwd_backend"):
        chunk_gated_delta_rule(q, q, q, g, g, bwd_backend="unknown")


@pytest.mark.parametrize("cache,threshold,expected_reconstructions", [
    (True, None, 1), (False, None, 2), (True, 5e-5, 2),
])
def test_partition_cache_reuses_matching_boundaries(monkeypatch, cache, threshold, expected_reconstructions):
    import flash_qla.ops.gated_delta_rule.chunk as chunk

    torch.manual_seed(174)
    q, k = [torch.randn(1, 512, 2, 128, device="cuda", dtype=torch.bfloat16).requires_grad_()
            for _ in range(2)]
    v = torch.randn(1, 512, 4, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    g = torch.full((1, 512, 4), -0.001, device="cuda", requires_grad=True)
    beta = torch.full_like(g, 0.5, requires_grad=True)
    calls = []
    original = chunk.fused_gdr_h

    def reconstruct(*args, **kwargs):
        calls.append(kwargs.get("output_h"))
        return original(*args, **kwargs)

    monkeypatch.setattr(chunk, "fused_gdr_h", reconstruct)
    output, _ = chunk_gated_delta_rule(
        q, k, v, g, beta, fwd_backend="optimized", bwd_backend="optimized",
        fwd_partitions=4, force_intra_cp=True, enable_fwd_cp_cache=cache,
        fwd_decay_threshold=threshold, use_qk_l2norm_in_kernel=True,
    )
    if cache and threshold is None:
        context = output.grad_fn.cp_context
        torch.testing.assert_close(context.intra_cp_cu_seqlens,
                                   torch.arange(5, device="cuda", dtype=torch.int32) * 128)
        assert output.grad_fn.cp_cache.h0.shape[0] == 4
        assert output.grad_fn.cp_cache.mt.shape[0] == 4
    else:
        assert output.grad_fn.cp_cache is None
    gradients = torch.autograd.grad(output, (q, k, v, g, beta), torch.randn_like(output))
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    assert len(calls) == expected_reconstructions
    assert calls[-1] is True  # Chunk states still need reconstruction.

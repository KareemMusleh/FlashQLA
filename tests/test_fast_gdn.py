"""Independent checks for the optimized inference path and its local solve."""
import pytest
import torch

from flash_qla.ops.gated_delta_rule.chunk.hopper.local_precompute import local_inverse, local_precompute
from flash_qla.ops.gated_delta_rule.chunk.hopper.fast_fwd import chunk_gated_delta_rule_hopper
from flash_qla import chunk_gated_delta_rule

pytestmark = [pytest.mark.gpu, pytest.mark.hopper]


def rms(actual, expected):
    return ((actual.double() - expected.double()).norm()
            / expected.double().norm().clamp_min(1e-30)).item()


@torch.no_grad()
def test_local_inverse_and_prefix():
    torch.manual_seed(12)
    t, hk, hv = 256, 2, 4
    k = torch.nn.functional.normalize(torch.randn(1, t, hk, 128, device="cuda"), dim=-1).bfloat16()
    beta = torch.rand(1, t, hv, device="cuda")
    beta[:, :, 0] = 0
    beta[:, :, -1] = 1
    g = -torch.rand_like(beta) / 16
    a = local_inverse(k, beta)
    ab, gc = local_precompute(k, beta, g)
    kr = k[0].double().reshape(t // 64, 64, hk, 128).permute(0, 2, 1, 3).repeat_interleave(hv // hk, dim=1)
    br = beta[0].double().reshape(t // 64, 64, hv).permute(0, 2, 1)
    eye = torch.eye(64, device="cuda", dtype=torch.float64).expand(t // 64, hv, 64, 64)
    system = eye + (br.unsqueeze(-1) * (kr @ kr.transpose(-1, -2))).tril(-1)
    ref = torch.linalg.solve_triangular(system, eye, upper=False)
    ar = a[0].reshape(t // 64, 64, hv, 64).permute(0, 2, 1, 3)
    abr = ab[0].reshape(t // 64, 64, hv, 64).permute(0, 2, 1, 3)
    assert rms(ar, ref) < 0.003
    assert rms(abr, ref * br.unsqueeze(-2)) < 0.003
    expected_g = g.double().reshape(1, t // 64, 64, hv).cumsum(2).reshape_as(g)
    torch.testing.assert_close(gc.double(), expected_g, rtol=2e-6, atol=2e-6)


@pytest.mark.parametrize("decay,parts,v_first,nonzero,scale", [
    (0.0, 1, False, True, None),
    (0.0001, 2, False, True, 0.2),
    (0.04, 4, False, False, None),
    (0.0, 2, True, True, None),
    (0.04, 4, True, True, 0.2),
    (0.04, 1, False, True, 0.0),
])
@torch.no_grad()
def test_forward_and_final_state(decay, parts, v_first, nonzero, scale):
    torch.manual_seed(13)
    t, hk, hv, d = 256, 2, 4, 128
    q, k = [torch.nn.functional.normalize(torch.randn(1, t, hk, d, device="cuda"), dim=-1).bfloat16() for _ in range(2)]
    v = torch.randn(1, t, hv, d, device="cuda", dtype=torch.bfloat16)
    beta = torch.rand(1, t, hv, device="cuda")
    beta[:, :, 0] = 0
    beta[:, :, -1] = 1
    g = -torch.rand_like(beta) * decay
    s0 = torch.randn(1, hv, d, d, device="cuda") * 0.1 if nonzero else None
    initial = s0.transpose(-1, -2).contiguous() if s0 is not None and v_first else s0
    output, final = chunk_gated_delta_rule_hopper(
        q, k, v, g, beta, partitions=parts, initial_state=initial,
        state_v_first=v_first, scale=scale, output_final_state=True)
    state = s0[0].double().clone() if s0 is not None else torch.zeros(hv, d, d, device="cuda", dtype=torch.float64)
    qr = q[0].double().repeat_interleave(hv // hk, dim=1)
    kr = k[0].double().repeat_interleave(hv // hk, dim=1)
    vr, br, gr = v[0].double(), beta[0].double(), g[0].double()
    expected = []
    factor = scale if scale is not None else d**-0.5
    for i in range(t):
        state *= gr[i].exp().unsqueeze(-1).unsqueeze(-1)
        residual = vr[i] - (kr[i].unsqueeze(-2) @ state).squeeze(-2)
        state += kr[i].unsqueeze(-1) * (br[i].unsqueeze(-1) * residual).unsqueeze(-2)
        expected.append((qr[i].unsqueeze(-2) @ state).squeeze(-2) * factor)
    assert torch.isfinite(output).all() and torch.isfinite(final).all()
    assert rms(output[0], torch.stack(expected)) < 0.012
    actual_state = final[0].transpose(-1, -2) if v_first else final[0]
    assert rms(actual_state, state) < 0.012


def test_autograd_is_not_silently_disabled():
    q = torch.zeros(1, 64, 1, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    g = torch.zeros(1, 64, 1, device="cuda")
    with pytest.raises(ValueError, match="forward-only"):
        chunk_gated_delta_rule_hopper(q, q, q, g, g, partitions=1)


@torch.no_grad()
def test_public_backend_and_normalization():
    torch.manual_seed(14)
    q, k = [torch.randn(1, 128, 2, 128, device="cuda", dtype=torch.bfloat16) for _ in range(2)]
    v = torch.randn(1, 128, 4, 128, device="cuda", dtype=torch.bfloat16)
    g = -torch.rand(1, 128, 4, device="cuda") / 16
    beta = torch.rand_like(g)
    expected = chunk_gated_delta_rule_hopper(q, k, v, g, beta, partitions=2,
                                             use_qk_l2norm_in_kernel=True)[0]
    actual = chunk_gated_delta_rule(q, k, v, g, beta, fwd_backend="optimized",
                                    fwd_partitions=2, use_qk_l2norm_in_kernel=True)[0]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("parts,v_first,normalize,loss_kind,threshold", [
    (1, False, False, "output", None),
    (2, False, True, "both", None),
    (4, True, True, "both", None),
    (2, True, False, "final", None),
    (2, False, False, "both", 5e-5),
])
def test_optimized_training_uses_default_backward(
    monkeypatch, parts, v_first, normalize, loss_kind, threshold,
):
    import flash_qla.ops.gated_delta_rule.chunk as chunk
    import flash_qla.ops.gated_delta_rule.chunk.hopper.fast_fwd as fast

    calls = []
    original_forward = fast.chunk_gated_delta_rule_hopper
    original_backward = chunk.chunk_gated_delta_rule_bwd

    def forward(*args, **kwargs):
        calls.append("optimized_forward")
        return original_forward(*args, **kwargs)

    def backward(*args, **kwargs):
        calls.append("default_backward")
        return original_backward(*args, **kwargs)

    monkeypatch.setattr(fast, "chunk_gated_delta_rule_hopper", forward)
    monkeypatch.setattr(chunk, "chunk_gated_delta_rule_bwd", backward)
    torch.manual_seed(15)
    q, k = [torch.nn.functional.normalize(
        torch.randn(1, 256, 2, 128, device="cuda"), dim=-1,
    ).bfloat16() for _ in range(2)]
    v = torch.randn(1, 256, 4, 128, device="cuda", dtype=torch.bfloat16)
    g = torch.full((1, 256, 4), -0.2 if threshold else -0.01, device="cuda")
    beta = torch.rand_like(g)
    initial = torch.randn(1, 4, 128, 128, device="cuda") * 0.1
    inputs = [x.requires_grad_() for x in (q, k, v, g, beta, initial)]
    reference_inputs = [x.detach().clone().requires_grad_() for x in inputs]
    kwargs = dict(
        output_final_state=True, state_v_first=v_first, scale=0.2,
        use_qk_l2norm_in_kernel=normalize, auto_cp=parts > 1,
        force_intra_cp=parts > 1, enable_fwd_cp_cache=False,
    )
    output, final = chunk_gated_delta_rule(
        *inputs[:5], initial_state=inputs[5], fwd_backend="optimized",
        fwd_partitions=parts, fwd_decay_threshold=threshold, **kwargs,
    )
    assert calls == ["optimized_forward"]
    with torch.no_grad():
        expected, expected_final = chunk_gated_delta_rule(
            *inputs[:5], initial_state=inputs[5], fwd_backend="optimized",
            fwd_partitions=parts, fwd_decay_threshold=threshold, **kwargs,
        )
    torch.testing.assert_close(output, expected, rtol=0, atol=0)
    torch.testing.assert_close(final, expected_final, rtol=0, atol=0)
    calls.clear()
    ref_output, ref_final = chunk_gated_delta_rule(
        *reference_inputs[:5], initial_state=reference_inputs[5], **kwargs,
    )
    do, dht = torch.randn_like(output), torch.randn_like(final)
    # Fixed upstream gradients isolate backward from forward rounding differences.
    if loss_kind == "output":
        output.backward(do)
        ref_output.backward(do)
    elif loss_kind == "final":
        final.backward(dht)
        ref_final.backward(dht)
    else:
        torch.autograd.backward((output, final), (do, dht))
        torch.autograd.backward((ref_output, ref_final), (do, dht))
    assert calls == ["default_backward", "default_backward"]
    for actual, reference in zip(inputs, reference_inputs):
        assert actual.grad is not None and torch.isfinite(actual.grad).all()
        torch.testing.assert_close(actual.grad, reference.grad, rtol=0, atol=0)


def test_optimized_training_falls_back_for_unsupported_dtype(monkeypatch):
    import flash_qla.ops.gated_delta_rule.chunk.hopper.fast_fwd as fast

    def unexpected_forward(*args, **kwargs):
        pytest.fail("FP16 inputs should use the default forward")

    monkeypatch.setattr(fast, "chunk_gated_delta_rule_hopper", unexpected_forward)
    q = (torch.randn(1, 64, 1, 128, device="cuda", dtype=torch.float16) * 0.03).requires_grad_()
    k, v = torch.randn_like(q) * 0.03, torch.randn_like(q)
    g = torch.full((1, 64, 1), -0.01, device="cuda")
    beta = torch.full_like(g, 0.5)
    output, _ = chunk_gated_delta_rule(
        q, k, v, g, beta, fwd_backend="optimized", auto_cp=False,
    )
    output.float().square().mean().backward()
    assert torch.isfinite(q.grad).all() and q.grad.abs().sum() > 0


@torch.no_grad()
def test_optional_decay_truncation():
    torch.manual_seed(16)
    t, hv = 512, 3
    q, k = [torch.nn.functional.normalize(torch.randn(1, t, 1, 128, device="cuda"), dim=-1).bfloat16()
            for _ in range(2)]
    v = torch.randn(1, t, hv, 128, device="cuda", dtype=torch.bfloat16)
    beta = torch.rand(1, t, hv, device="cuda")
    g = torch.empty_like(beta)
    g[..., 0], g[..., 1], g[..., 2] = -0.2, -0.0001, 0.0
    initial = torch.randn(1, hv, 128, 128, device="cuda") * 0.1
    kwargs = dict(fwd_backend="optimized", fwd_partitions=2, initial_state=initial,
                  output_final_state=True)
    full, hf = chunk_gated_delta_rule(q, k, v, g, beta, **kwargs)
    explicit, he = chunk_gated_delta_rule(q, k, v, g, beta, fwd_decay_threshold=None, **kwargs)
    torch.testing.assert_close(full, explicit, rtol=0, atol=0)
    torch.testing.assert_close(hf, he, rtol=0, atol=0)
    truncated, ht = chunk_gated_delta_rule(q, k, v, g, beta, fwd_decay_threshold=5e-5, **kwargs)
    assert rms(truncated, full) < 0.001
    assert rms(ht, hf) < 0.001
    torch.testing.assert_close(truncated[..., 1:, :], full[..., 1:, :], rtol=0, atol=0)
    torch.testing.assert_close(ht[:, 1:], hf[:, 1:], rtol=0, atol=0)
    # No partition summaries are needed with one partition.
    single = dict(fwd_backend="optimized", auto_cp=False)
    a = chunk_gated_delta_rule(q, k, v, g, beta, **single)[0]
    b = chunk_gated_delta_rule(q, k, v, g, beta, fwd_decay_threshold=5e-5, **single)[0]
    torch.testing.assert_close(a, b, rtol=0, atol=0)


@pytest.mark.parametrize("threshold", [0.0, -1.0, 1.0, float("nan"), float("inf")])
def test_invalid_decay_threshold(threshold):
    q = torch.zeros(1, 64, 1, 128, dtype=torch.bfloat16)
    g = torch.zeros(1, 64, 1)
    with pytest.raises(ValueError, match="fwd_decay_threshold"):
        chunk_gated_delta_rule(q, q, q, g, g, fwd_decay_threshold=threshold)

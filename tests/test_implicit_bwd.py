"""Independent recurrence checks for the experimental implicit adjoint."""
import pytest
import torch

from flash_qla.ops.gated_delta_rule.chunk import chunk_gated_delta_rule_fwd
from flash_qla.ops.gated_delta_rule.chunk.hopper.implicit_bwd import implicit_gdr_bwd

pytestmark = [pytest.mark.gpu, pytest.mark.hopper]


def relative_rms(actual, expected):
    return ((actual.double() - expected.double()).norm()
            / expected.double().norm().clamp_min(1e-30)).item()


def recurrence(q, k, v, gates, beta, initial, scale, v_first):
    """Token-by-token FP64 recurrence, independently differentiated by PyTorch."""
    hv, hk = v.shape[2], k.shape[2]
    qr = q[0].repeat_interleave(hv // hk, dim=1)
    kr = k[0].repeat_interleave(hv // hk, dim=1)
    state = initial[0].transpose(-1, -2) if v_first else initial[0]
    outputs = []
    for i in range(q.shape[1]):
        state = state * gates[0, i].exp()[:, None, None]
        residual = v[0, i] - (kr[i, :, None, :] @ state).squeeze(-2)
        state = state + kr[i, :, :, None] * (beta[0, i, :, None] * residual)[:, None, :]
        outputs.append((qr[i, :, None, :] @ state).squeeze(-2) * scale)
    final = state.transpose(-1, -2) if v_first else state
    return torch.stack(outputs).unsqueeze(0), final.unsqueeze(0)


@pytest.mark.parametrize("length,decay,v_first,terminal,scale", [
    (64, 0.0, False, False, 0.2),
    (128, 0.01, False, True, 0.2),
    (128, 0.08, True, True, 0.2),
    (256, 0.0, True, False, 0.2),
    (64, 0.04, False, True, 0.0),
])
def test_implicit_adjoint_against_fp64(length, decay, v_first, terminal, scale):
    torch.manual_seed(123)
    hk, hv = 1, 4
    q, k = [torch.nn.functional.normalize(
        torch.randn(1, length, hk, 128, device="cuda"), dim=-1,
    ).bfloat16() for _ in range(2)]
    v = torch.randn(1, length, hv, 128, device="cuda", dtype=torch.bfloat16)
    g = -torch.rand(1, length, hv, device="cuda") * decay
    beta = torch.rand_like(g)
    beta[..., 0] = 0
    beta[..., 1] = 1
    initial = torch.randn(1, hv, 128, 128, device="cuda") * 0.1
    do = torch.randn_like(v)
    dht = torch.randn_like(initial) * 0.1 if terminal else None
    with torch.no_grad():
        gc, a, _, h, _, _ = chunk_gated_delta_rule_fwd(
            q, k, v, g, beta, scale=scale, initial_state=initial,
            auto_cp=False, output_h=True, state_v_first=v_first,
        )
        dq, dk, dv, dg, db, dh = implicit_gdr_bwd(
            q, k, v, a, gc, beta, do, dht, h, scale=scale,
            state_v_first=v_first,
        )
        dq = dq.reshape(1, length, hk, hv // hk, 128).float().sum(3)
        dk = dk.reshape(1, length, hk, hv // hk, 128).float().sum(3)
        dg = dg.reshape(1, length // 64, 64, hv).flip(2).cumsum(2).flip(2).reshape_as(g)
    inputs = [x.double().detach().requires_grad_() for x in (q, k, v, g, beta, initial)]
    output, final = recurrence(*inputs, scale, v_first)
    loss = (output * do.double()).sum()
    if terminal:
        loss = loss + (final * dht.double()).sum()
    expected = torch.autograd.grad(loss, inputs)
    # Check the zero-beta head separately so other heads cannot hide its error.
    assert expected[4][..., 0].norm() > 0
    assert relative_rms(db[..., 0], expected[4][..., 0]) < 0.035
    for name, actual, ref in zip(("dq", "dk", "dv", "dg", "db", "dh"),
                                 (dq, dk, dv, dg, db, dh), expected):
        assert torch.isfinite(actual).all(), name
        if ref.norm() == 0:
            assert actual.norm() == 0, name
        else:
            assert relative_rms(actual, ref) < 0.035, (name, relative_rms(actual, ref))


@pytest.mark.parametrize("v_first,normalize", [(False, False), (True, True)])
def test_training_driver_with_implicit_kernel(monkeypatch, v_first, normalize):
    import flash_qla.ops.gated_delta_rule.chunk as chunk

    torch.manual_seed(124)
    q, k = [torch.nn.functional.normalize(
        torch.randn(1, 256, 2, 128, device="cuda"), dim=-1,
    ).bfloat16().requires_grad_() for _ in range(2)]
    v = torch.randn(1, 256, 4, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    g = torch.full((1, 256, 4), -0.01, device="cuda", requires_grad=True)
    beta = torch.full_like(g, 0.5, requires_grad=True)
    initial = (torch.randn(1, 4, 128, 128, device="cuda") * 0.1).requires_grad_()
    inputs = (q, k, v, g, beta, initial)
    kwargs = dict(initial_state=initial, fwd_backend="optimized", fwd_partitions=2,
                  force_intra_cp=True, state_v_first=v_first,
                  use_qk_l2norm_in_kernel=normalize, output_final_state=True)
    do, dht = torch.randn_like(v), torch.randn_like(initial)
    output, final = chunk.chunk_gated_delta_rule(*inputs[:5], **kwargs)
    expected = torch.autograd.grad((output, final), inputs, (do, dht))
    monkeypatch.setattr(chunk, "fused_gdr_bwd", implicit_gdr_bwd)
    output, final = chunk.chunk_gated_delta_rule(*inputs[:5], **kwargs)
    actual = torch.autograd.grad((output, final), inputs, (do, dht))
    for name, a, b in zip(("dq", "dk", "dv", "dg", "db", "dh"), actual, expected):
        assert torch.isfinite(a).all(), name
        assert relative_rms(a, b) < 0.04, (name, relative_rms(a, b))

"""Gradient postprocessing must preserve grouped reduction rounding."""

import importlib

import pytest
import torch

from flash_qla.utils.math import group_reduce_l2norm_bwd, l2norm_bwd, l2norm_fwd


@pytest.mark.gpu
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("group", [1, 2, 3, 4])
@pytest.mark.parametrize("dim", [63, 128])
def test_group_reduce_l2norm_bwd(dtype, group, dim):
    torch.manual_seed(42)
    # Partial final block exercises masking; batch > 1 tests flattening.
    x = torch.randn(2, 37, 3, dim, device="cuda", dtype=dtype)
    y, rstd = l2norm_fwd(x)
    dy = torch.randn(2, 37, 3 * group, dim, device="cuda", dtype=dtype)
    grouped = dy.float().reshape(2, 37, 3, group, dim)
    reduced = torch.zeros_like(grouped[..., 0, :])
    for i in range(group):
        reduced = reduced + grouped[..., i, :]
    reduced = reduced.to(dtype).float()
    yf = y.float()
    expected = (
        (reduced - (reduced * yf).sum(-1, keepdim=True) * yf) * rstd.unsqueeze(-1)
    ).to(dtype)
    actual = group_reduce_l2norm_bwd(y, rstd, dy)
    assert actual.dtype == dtype
    torch.testing.assert_close(
        actual, expected, rtol=0, atol=2e-3 if dtype == torch.bfloat16 else 2e-4
    )


@pytest.mark.gpu
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_l2norm_bwd_output_dtype(dtype):
    torch.manual_seed(42)
    x = torch.randn(1, 73, 4, 128, device="cuda", dtype=dtype)
    y, rstd = l2norm_fwd(x)
    dy = torch.randn_like(y)
    reference = l2norm_bwd(y, rstd, dy)
    assert reference.dtype == torch.float32
    actual = l2norm_bwd(y, rstd, dy, output_dtype=dtype)
    assert actual.dtype == dtype
    torch.testing.assert_close(actual, reference.to(dtype), rtol=0, atol=0)


@pytest.mark.gpu
@pytest.mark.needs_bwd
@pytest.mark.parametrize("group", [1, 3])
@pytest.mark.parametrize("normalize", [False, True])
def test_gdn_gradient_postprocessing(monkeypatch, group, normalize):
    chunk = importlib.import_module("flash_qla.ops.gated_delta_rule.chunk")
    torch.manual_seed(42)
    q, k = [
        torch.randn(
            1, 128, 2, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True
        )
        for _ in range(2)
    ]
    if not normalize:
        q, k = [
            torch.nn.functional.normalize(t.detach().float(), dim=-1)
            .to(t.dtype)
            .requires_grad_()
            for t in (q, k)
        ]
    v = torch.randn(
        1, 128, 2 * group, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    g = torch.full((1, 128, 2 * group), -0.1, device="cuda", requires_grad=True)
    beta = torch.rand_like(g).requires_grad_()
    h0 = torch.randn(1, 2 * group, 128, 128, device="cuda", requires_grad=True)
    do, dht = torch.randn_like(v), torch.randn_like(h0)
    inputs = q, k, v, g, beta, h0

    def run():
        o, ht = chunk.chunk_gated_delta_rule(
            q,
            k,
            v,
            g,
            beta,
            initial_state=h0,
            output_final_state=True,
            auto_cp=False,
            use_qk_l2norm_in_kernel=normalize,
        )
        return torch.autograd.grad((o, ht), inputs, (do, dht))

    actual = run()
    optimized = chunk.chunk_gated_delta_rule_bwd

    def previous(*args, q_rstd=None, k_rstd=None, **kwargs):
        dq, dk, dv, db, dg, dh0 = optimized(*args, **kwargs)
        if q_rstd is not None:
            dq = l2norm_bwd(kwargs["q"], q_rstd, dq)
            dk = l2norm_bwd(kwargs["k"], k_rstd, dk)
        return dq, dk, dv, db, dg, dh0

    monkeypatch.setattr(chunk, "chunk_gated_delta_rule_bwd", previous)
    expected = run()
    for a, b in zip(actual, expected):
        assert torch.isfinite(a).all()
        torch.testing.assert_close(a, b, rtol=0.01, atol=0.001)

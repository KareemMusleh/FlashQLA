"""Check partition summaries against the token recurrence, including live M."""

import pytest
import torch

from flash_qla.ops.gated_delta_rule.chunk import kkt_solve
from flash_qla.ops.gated_delta_rule.chunk.hopper.prepare_mn import prepare_mn

pytestmark = [pytest.mark.gpu, pytest.mark.hopper]


def relative_rms(actual, expected):
    return ((actual.float() - expected.float()).norm()
            / expected.float().norm().clamp_min(1e-12)).item()


@pytest.mark.parametrize("decay", [0.0, 0.0001, 0.04])
@pytest.mark.parametrize("state_v_first", [False, True])
@pytest.mark.parametrize("partition_size", [64, 256])
@torch.no_grad()
def test_partition_affine_map(decay, state_v_first, partition_size,
                              overlap_mn=False, lifetime_control=False):
    torch.manual_seed(42)
    p, hk, hv, d = 2, 1, 3, 128
    tokens = p * partition_size
    k = torch.nn.functional.normalize(
        torch.randn(1, tokens, hk, d, device="cuda"), dim=-1).bfloat16()
    v = torch.randn(1, tokens, hv, d, device="cuda", dtype=torch.bfloat16)
    beta = torch.rand(1, tokens, hv, device="cuda")
    logg = -torch.full_like(beta, decay)
    g = logg.reshape(1, tokens // 64, 64, hv).cumsum(2).reshape_as(logg)
    a = kkt_solve(k, beta, chunk_size=64)
    n, m = prepare_mn(k, v, a, g, beta, partition_size,
                      state_v_first=state_v_first, output_dtype=torch.float32,
                      overlap_mn=overlap_mn, lifetime_control=lifetime_control)
    if state_v_first:
        n = n.transpose(-1, -2)

    # Independent FP64 reference. The left block evolves identity into M;
    # the right block evolves zero into N under the token-level GDN update.
    kr = k.double().repeat_interleave(hv // hk, dim=2).reshape(p, partition_size, hv, d)
    vr = v.double().reshape(p, partition_size, hv, d)
    br = beta.double().reshape(p, partition_size, hv)
    mr = torch.eye(d, device="cuda", dtype=torch.float64).expand(p, hv, d, d).clone()
    nr = torch.zeros_like(mr)
    alpha = torch.exp(torch.tensor(-decay, device="cuda", dtype=torch.float64))
    for t in range(partition_size):
        kt, vt, bt = kr[:, t], vr[:, t], br[:, t]
        mr *= alpha
        nr *= alpha
        projected_m = (kt.unsqueeze(-2) @ mr).squeeze(-2)
        mr -= kt.unsqueeze(-1) * (bt.unsqueeze(-1) * projected_m).unsqueeze(-2)
        residual = vt - (kt.unsqueeze(-2) @ nr).squeeze(-2)
        nr += kt.unsqueeze(-1) * (bt.unsqueeze(-1) * residual).unsqueeze(-2)
    assert relative_rms(m, mr) < 0.012
    assert relative_rms(n, nr) < 0.012

    # Exercise a nonzero incoming state, where an incorrect transfer matrix
    # cannot be hidden by the zero-state N result.
    h0 = torch.randn_like(mr)
    actual = m.double() @ h0 + n.double()
    expected = mr @ h0 + nr
    assert relative_rms(actual, expected) < 0.012


def test_long_weak_decay_partition():
    # At 2k tokens M is still significant with this decay, unlike the benchmark
    # gates whose long-partition transfer matrix underflows to zero.
    test_partition_affine_map(0.0001, False, 2048)


@pytest.mark.parametrize("state_v_first", [False, True])
@pytest.mark.parametrize("lifetime_control", [False, True])
def test_overlapped_affine_map(state_v_first, lifetime_control):
    test_partition_affine_map(0.0001, state_v_first, 256,
                              overlap_mn=True, lifetime_control=lifetime_control)


def test_overlapped_long_weak_decay():
    test_partition_affine_map(0.0001, False, 2048, overlap_mn=True)


@pytest.mark.parametrize("split_n_only", [False, True])
def test_overlapped_mixed_transfers(monkeypatch, split_n_only):
    from functools import partial

    monkeypatch.setitem(globals(), "prepare_mn", partial(
        prepare_mn, overlap_mn=True, lifetime_control=True))
    test_truncated_summary_matches_retained_suffix(True, split_n_only)


@torch.no_grad()
def test_skip_last_and_bf16_output():
    torch.manual_seed(43)
    k = torch.nn.functional.normalize(torch.randn(1, 128, 1, 128, device="cuda"), dim=-1).bfloat16()
    v = torch.randn_like(k)
    beta = torch.rand(1, 128, 1, device="cuda")
    g = torch.zeros_like(beta)
    a = kkt_solve(k, beta, chunk_size=64)
    n, m = prepare_mn(k, v, a, g, beta, 64, skip_last=True)
    nf, mf = prepare_mn(k, v, a, g, beta, 64)
    assert n.dtype == m.dtype == torch.bfloat16
    torch.testing.assert_close(n[0], nf[0], rtol=0, atol=0)
    torch.testing.assert_close(m[0], mf[0], rtol=0, atol=0)
    assert torch.count_nonzero(n[1]) == 0
    torch.testing.assert_close(m[1, 0], torch.eye(128, device="cuda", dtype=torch.bfloat16))


@pytest.mark.parametrize("state_v_first", [False, True])
@pytest.mark.parametrize("split_n_only", [False, True])
@torch.no_grad()
def test_truncated_summary_matches_retained_suffix(state_v_first, split_n_only):
    torch.manual_seed(44)
    p, length, hv, d = 2, 256, 3, 128
    k = torch.nn.functional.normalize(torch.randn(1, p * length, 1, d, device="cuda"), dim=-1).bfloat16()
    v = torch.randn(1, p * length, hv, d, device="cuda", dtype=torch.bfloat16)
    beta = torch.rand(1, p * length, hv, device="cuda")
    logg = -torch.rand_like(beta) / 16
    g = logg.reshape(1, p * length // 64, 64, hv).cumsum(2).reshape_as(logg)
    a = kkt_solve(k, beta, chunk_size=64)
    counts = torch.tensor([[1, 2, 4], [3, 4, 1]], device="cuda", dtype=torch.int32)
    n, m = prepare_mn(k, v, a, g, beta, length, output_dtype=torch.float32,
                      state_v_first=state_v_first, num_warmup_chunks=counts,
                      split_n_only=split_n_only, lifetime_control=True)
    kr = k.double().expand(1, p * length, hv, d).reshape(p, length, hv, d)
    vr = v.double().reshape(p, length, hv, d)
    br = beta.double().reshape(p, length, hv)
    gr = logg.double().reshape(p, length, hv)
    nr = torch.zeros(p, hv, d, d, device="cuda", dtype=torch.float64)
    mr = torch.eye(d, device="cuda", dtype=torch.float64).expand_as(nr).clone()
    for t in range(length):
        active = (t >= length - counts * 64)[..., None, None]
        kt = kr[:, t]
        alpha = gr[:, t].exp()[..., None, None]
        decayed_n, decayed_m = nr * alpha, mr * alpha
        residual = vr[:, t] - (kt.unsqueeze(-2) @ decayed_n).squeeze(-2)
        nn = decayed_n + kt.unsqueeze(-1) * (br[:, t, :, None] * residual).unsqueeze(-2)
        mm = decayed_m - kt.unsqueeze(-1) * (
            br[:, t, :, None] * (kt.unsqueeze(-2) @ decayed_m).squeeze(-2)).unsqueeze(-2)
        nr, mr = torch.where(active, nn, nr), torch.where(active, mm, mr)
    mr[counts < length // 64] = 0
    assert relative_rms(n.transpose(-1, -2) if state_v_first else n, nr) < 0.012
    assert relative_rms(m, mr) < 0.012
    assert torch.count_nonzero(m[counts < length // 64]) == 0

    # A head may cross the cutoff only at the oldest chunk: keep every chunk
    # for N but discard the incoming-state transfer, so M is still unnecessary.
    transfer_mask = counts == length // 64
    transfer_mask[0, 2] = False
    nm, mm = prepare_mn(k, v, a, g, beta, length, output_dtype=torch.float32,
                        state_v_first=state_v_first, num_warmup_chunks=counts,
                        transfer_mask=transfer_mask, split_n_only=split_n_only,
                        lifetime_control=True)
    torch.testing.assert_close(nm, n, rtol=2e-5, atol=2e-6)
    expected_m = m.clone()
    expected_m[~transfer_mask] = 0
    torch.testing.assert_close(mm, expected_m, rtol=2e-5, atol=2e-6)

    full_counts = torch.full_like(counts, length // 64)
    nf, mf = prepare_mn(k, v, a, g, beta, length, state_v_first=state_v_first)
    nc, mc = prepare_mn(k, v, a, g, beta, length, state_v_first=state_v_first,
                        num_warmup_chunks=full_counts)
    torch.testing.assert_close(nc, nf, rtol=0, atol=0)
    torch.testing.assert_close(mc, mf, rtol=0, atol=0)


@pytest.mark.parametrize("state_v_first", [False, True])
@pytest.mark.parametrize("skip_last", [False, True])
@torch.no_grad()
def test_split_n_only_folded_bf16(state_v_first, skip_last):
    from flash_qla.ops.gated_delta_rule.chunk.hopper.local_precompute import local_precompute
    torch.manual_seed(45)
    p, length, hv = 2, 256, 3
    k = torch.nn.functional.normalize(torch.randn(1,p*length,1,128,device="cuda"),dim=-1).bfloat16()
    v = torch.randn(1,p*length,hv,128,device="cuda",dtype=torch.bfloat16)
    beta = torch.rand(1,p*length,hv,device="cuda")
    g = -torch.rand_like(beta)*.0001
    a,gc = local_precompute(k,beta,g)
    counts = torch.tensor([[1,4,4],[4,3,2]],device="cuda",dtype=torch.int32)
    mask = torch.tensor([[False,True,False],[True,False,False]],device="cuda")
    kw = dict(state_v_first=state_v_first,skip_last=skip_last,beta_folded=True,
              num_warmup_chunks=counts,transfer_mask=mask)
    reference = prepare_mn(k,v,a,gc,beta,length,**kw)
    candidate = prepare_mn(k,v,a,gc,beta,length,split_n_only=True,lifetime_control=True,**kw)
    for actual,expected in zip(candidate,reference):
        torch.testing.assert_close(actual,expected,rtol=0,atol=0)

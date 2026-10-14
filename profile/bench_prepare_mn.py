"""Compare the readable M/N kernel with full-history FlashQLA on an H100.

CUDA_VISIBLE_DEVICES=3 /opt/ydt/bin/python profile/bench_prepare_mn.py \
    --flashinfer /tmp/flashinfer-gdn-compare
Preparation timings process every partition and write FP32 summaries. Both local
matrices are prepared outside stage timing. Forward timings use normalized Q/K
and CUDA graphs; the last outgoing summary is skipped when it is not needed.
"""

import argparse
import importlib
import json
import statistics
import sys
from functools import partial
from pathlib import Path
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flash_qla import chunk_gated_delta_rule
from flash_qla.ops.gated_delta_rule.chunk.cp.context import _build_intra_cp_context
from flash_qla.ops.gated_delta_rule.chunk.hopper.prepare_mn import prepare_mn, _prepare_mn

chunk = importlib.import_module("flash_qla.ops.gated_delta_rule.chunk")


def error(x, y):
    return ((x.float() - y.float()).norm() / y.float().norm().clamp_min(1e-30)).item()


def benchmark(functions, inner_repeats=1):
    graphs, outputs = {}, {}
    for name, fn in functions.items():
        for _ in range(3):
            fn()
        torch.cuda.synchronize()
        graphs[name] = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graphs[name]):
            for _ in range(inner_repeats):
                outputs[name] = fn()
    for _ in range(10):
        for graph in graphs.values():
            graph.replay()
    samples = {name: [] for name in functions}
    names = list(functions)
    for cycle in range(9):
        for name in names[cycle % len(names):] + names[:cycle % len(names)]:
            start, end = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
            start.record()
            for _ in range(30):
                graphs[name].replay()
            end.record()
            end.synchronize()
            samples[name].append(start.elapsed_time(end) / (30 * inner_repeats))
    return {name: {"median_ms": statistics.median(times), "samples_ms": times}
            for name, times in samples.items()}


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seqlen", type=int, default=16384)
    parser.add_argument("--partitions", type=int, nargs="+", default=[2, 8])
    parser.add_argument("--output", type=Path, default=Path("traces/prepare_mn_16k"))
    parser.add_argument("--flashinfer", type=Path, default=None)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(42)
    torch.set_num_threads(4)
    torch.cuda.set_stream(torch.cuda.Stream())
    t, hk, hv, d = args.seqlen, 16, 48, 128
    q, k = [torch.nn.functional.normalize(
        torch.randn(1, t, hk, d, device="cuda"), dim=-1).bfloat16() for _ in range(2)]
    v = torch.randn(1, t, hv, d, device="cuda", dtype=torch.bfloat16)
    logg = torch.nn.functional.logsigmoid(torch.randn(1, t, hv, device="cuda")) / 16
    beta = torch.randn_like(logg).sigmoid()
    cu = torch.tensor([0, t], device="cuda", dtype=torch.int32)
    g = chunk.chunk_local_cumsum(logg, chunk_size=64)
    a = chunk.kkt_solve(k, beta, chunk_size=64)
    if args.flashinfer:
        sys.path.insert(0, str(args.flashinfer))
        from flashinfer.gdn_kernels.delta_rule_dsl.delta_rule_cp_sm90 import (
            cp_delta_rule_t_precompute_dsl_sm90 as fi_t,
            cp_delta_rule_mn_precompute_dsl_sm90 as fi_mn,
        )
        alpha = logg[0].exp()
        fi_local = fi_t(k[0], beta[0], cu, t, max_seqlen=t)
    full_warmup = partial(chunk.get_warmup_chunks, threshold=float("-inf"))
    results = {"gpu": torch.cuda.get_device_name(), "tokens": t,
               "shape": [1, t, hk, hv, d], "scope": "forward; pre-normalized Q/K; full history",
               "timing": "9 rotating-order CUDA graph samples, 30 replays each",
               "stage_scope": "all partitions, FP32 outputs; local matrices precomputed outside timing",
               "implementation": "384 threads: TMA producer plus 2 ordered math groups; STMatrix X stores; no decay truncation",
               "partitions": {}}
    ref = chunk_gated_delta_rule(q, k, v, logg, beta, auto_cp=False,
                                 use_qk_l2norm_in_kernel=False, output_final_state=False)[0]
    for p in args.partitions:
        assert t % (p * 64) == 0
        length = t // p
        ctx = _build_intra_cp_context(cu, 64, [t // 64], length // 64)
        warmup = torch.full((p, hv), length // 64, device="cuda", dtype=torch.int32)

        def old_stage():
            return chunk.fused_gdr_h(k, v, a, g, beta, output_h=False,
                                     cu_seqlens=ctx.intra_cp_cu_seqlens,
                                     num_warmup_chunks=warmup)[1:]

        def new_stage():
            return prepare_mn(k, v, a, g, beta, length, output_dtype=torch.float32)

        def adapter(k, v, a, g, b, initial_state=None, output_final_state=True,
                    output_h=False, cu_seqlens=None, num_warmup_chunks=None,
                    state_v_first=False):
            assert initial_state is None and not output_h and output_final_state
            n, m = prepare_mn(k, v, a, g, b, length, state_v_first=state_v_first,
                              skip_last=True)
            return None, n, m

        def forward(use_new):
            with patch.object(chunk, "get_warmup_chunks", full_warmup):
                with patch.object(chunk, "fused_gdr_h", adapter if use_new else original_prepare):
                    return chunk_gated_delta_rule(
                        q, k, v, logg, beta, use_qk_l2norm_in_kernel=False,
                        output_final_state=False, cp_context=ctx)[0]

        original_prepare = chunk.fused_gdr_h
        print(f"Checking and compiling {p} partitions", flush=True)
        no, mo = old_stage()
        nn, mn = new_stage()
        check = {"N_rms_vs_old": error(nn, no),
                 "M_rms_vs_old": error(mn, mo)}
        assert torch.isfinite(nn).all() and torch.isfinite(mn).all()
        assert check["N_rms_vs_old"] < 0.015, check
        for name, use_new in [("old", False), ("new", True)]:
            output = forward(use_new)
            check[f"{name}_output_rms_vs_no_cp"] = error(output, ref)
            assert torch.isfinite(output).all() and check[f"{name}_output_rms_vs_no_cp"] < 0.01, check
        functions = {"old_prepare": old_stage, "new_prepare": new_stage,
                     "old_forward": partial(forward, False), "new_forward": partial(forward, True)}
        if args.flashinfer:
            functions["flashinfer_prepare"] = lambda: fi_mn(
                k[0], v[0], fi_local, alpha, cu, t, cp_chunk_len=length, max_seqlen=t)
        measurements = benchmark(functions)
        results["partitions"][p] = {"correctness": check, **measurements}
        print(json.dumps(results["partitions"][p], indent=2), flush=True)
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                torch.profiler.ProfilerActivity.CUDA]) as prof:
            for name, fn in functions.items():
                with torch.profiler.record_function(name):
                    fn()
            torch.cuda.synchronize()
        prof.export_chrome_trace(str(args.output / f"p{p}.json"))
        source = _prepare_mn(t, hk, hv, length, False, False, torch.float32).get_kernel_source()
        (args.output / f"p{p}_kernel.cu").write_text(source)
        (args.output / "results.json").write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()

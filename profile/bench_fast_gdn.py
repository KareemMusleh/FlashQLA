"""Matched-partition H100 forward comparison, including all GPU work."""
import argparse
import importlib
import importlib.util
import json
import math
import sys
from functools import partial
from pathlib import Path
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bench_prepare_mn import benchmark, error
from flash_qla import chunk_gated_delta_rule as qla
from flash_qla.ops.gated_delta_rule.chunk.cp.context import _build_intra_cp_context
from flash_qla.ops.gated_delta_rule.chunk.hopper.cp_fwd import get_warmup_chunks
chunk = importlib.import_module("flash_qla.ops.gated_delta_rule.chunk")


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seqlen", type=int, default=16384)
    parser.add_argument("--partitions", type=int, nargs="+", default=[2, 8])
    parser.add_argument("--flashinfer", type=Path, default=Path("/tmp/flashinfer-gdn-compare"))
    parser.add_argument("--skip-flashinfer", action="store_true",
                        help="Sweep FlashQLA alone without requiring a particular FlashInfer checkout")
    parser.add_argument("--output", type=Path, default=Path("traces/gdn_fast_16k"))
    parser.add_argument("--inner-repeats", type=int, default=None)
    parser.add_argument("--optimized-prepare-file", type=Path,
                        help="Optional frozen M/N module for controlled comparisons")
    parser.add_argument("--optimized-output-file", type=Path,
                        help="Optional frozen output module for controlled comparisons")
    parser.add_argument("--decay-threshold", type=float, default=None,
                        help="Also measure optional truncation separately; the FlashInfer comparison stays full-history")
    args = parser.parse_args()
    if args.optimized_prepare_file or args.optimized_output_file:
        fast = importlib.import_module("flash_qla.ops.gated_delta_rule.chunk.hopper.fast_fwd")
        for name, path, attr in [('prepare', args.optimized_prepare_file, 'prepare_mn'),
                                 ('output', args.optimized_output_file, 'output_fwd')]:
            if path is not None:
                spec = importlib.util.spec_from_file_location('frozen_' + name, path)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                setattr(fast, attr, getattr(module, attr))
    sys.path.insert(0, str(args.flashinfer))
    if not args.skip_flashinfer:
        from flashinfer.gdn_prefill import chunk_gated_delta_rule as fi
    torch.manual_seed(42)
    torch.set_num_threads(4)
    torch.cuda.set_stream(torch.cuda.Stream())
    t = args.seqlen
    inner_repeats = args.inner_repeats or (10 if t <= 16384 else 1)
    q, k = [torch.nn.functional.normalize(torch.randn(1, t, 16, 128, device="cuda"), dim=-1).bfloat16() for _ in range(2)]
    v = torch.randn(1, t, 48, 128, device="cuda", dtype=torch.bfloat16)
    g = torch.nn.functional.logsigmoid(torch.randn(1, t, 48, device="cuda")) / 16
    beta = torch.randn_like(g).sigmoid()
    alpha = g[0].exp()
    cu = torch.tensor([0, t], device="cuda", dtype=torch.int32)
    reference = qla(q, k, v, g, beta, auto_cp=False, output_final_state=False)[0]
    full_warmup = partial(chunk.get_warmup_chunks, threshold=float("-inf"))
    results = {"gpu": torch.cuda.get_device_name(), "shape": [1, t, 16, 48, 128],
               "scope": "forward; normalized Q/K; full history; no initial/final state",
               "comparison_decay_threshold": None,
               "optional_truncated_decay_threshold": args.decay_threshold,
               "optimized_prepare_file": str(args.optimized_prepare_file) if args.optimized_prepare_file else None,
               "optimized_output_file": str(args.optimized_output_file) if args.optimized_output_file else None,
               "timing": f"9 rotating-order samples, 30 graph replays with {inner_repeats} forwards per graph", "partitions": {}}
    args.output.mkdir(parents=True, exist_ok=True)
    for parts in args.partitions:
        ctx = _build_intra_cp_context(cu, 64, [t // 64], t // parts // 64)
        def baseline():
            with patch.object(chunk, "get_warmup_chunks", full_warmup):
                return qla(q, k, v, g, beta, cp_context=ctx, output_final_state=False)[0]
        functions = {
            "flashqla_full": baseline,
            "optimized": lambda: qla(q, k, v, g, beta, fwd_backend="optimized",
                                       fwd_partitions=parts, fwd_decay_threshold=None,
                                       output_final_state=False)[0],
        }
        if not args.skip_flashinfer:
            functions["flashinfer"] = lambda: fi(q[0], k[0], v[0], alpha, beta[0],
                cu_seqlens=cu, use_cp=True, _cp_chunk_len=t // parts,
                output_final_state=False, use_qk_l2norm_in_kernel=False)[None]
        if args.decay_threshold is not None:
            functions["optimized_truncated"] = lambda: qla(
                q, k, v, g, beta, fwd_backend="optimized", fwd_partitions=parts,
                fwd_decay_threshold=args.decay_threshold, output_final_state=False)[0]
        checks = {}
        for name, fn in functions.items():
            print("CHECK", parts, name, flush=True)
            output = fn()
            checks[name] = {"relative_rms": error(output, reference), "finite": bool(output.isfinite().all())}
            assert checks[name]["finite"] and checks[name]["relative_rms"] < 0.012, checks
        timings = benchmark(functions, inner_repeats=inner_repeats)
        results["partitions"][parts] = {"correctness": checks, **timings}
        if args.decay_threshold is not None:
            gc = g.reshape(1, t // 64, 64, 48).cumsum(2).reshape_as(g)
            boundaries = torch.arange(parts + 1, device="cuda", dtype=torch.int32) * (t // parts)
            counts, _ = get_warmup_chunks(gc, boundaries, torch.zeros(parts, device="cuda", dtype=torch.bool),
                                          threshold=math.log(args.decay_threshold))
            retained = counts[:-1].float()
            results["partitions"][parts]["truncation"] = {
                "full_chunks_per_partition": t // parts // 64,
                "mean_retained_chunks": retained.mean().item() if parts > 1 else None,
                "max_retained_chunks": retained.max().item() if parts > 1 else None,
                "relative_rms_vs_optimized_full": error(functions["optimized_truncated"](), functions["optimized"]()),
            }
        print(json.dumps(results["partitions"][parts], indent=2), flush=True)
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                torch.profiler.ProfilerActivity.CUDA]) as prof:
            for name, fn in functions.items():
                with torch.profiler.record_function(name):
                    fn()
            torch.cuda.synchronize()
        prof.export_chrome_trace(str(args.output / f"p{parts}.json"))
        (args.output / "results.json").write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()

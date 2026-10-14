"""Before/after measurements for M skipping and combined state projections."""
import argparse
import importlib
import importlib.util
import json
import math
import sys
from pathlib import Path
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bench_prepare_mn import benchmark, error
from flash_qla import chunk_gated_delta_rule
from flash_qla.ops.gated_delta_rule.chunk.hopper.local_precompute import local_precompute
from flash_qla.ops.gated_delta_rule.chunk.hopper.cp_fwd import get_warmup_chunks

fast = importlib.import_module("flash_qla.ops.gated_delta_rule.chunk.hopper.fast_fwd")


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seqlen", type=int, default=16384)
    parser.add_argument("--baseline-prepare", type=Path, required=True)
    parser.add_argument("--baseline-output", type=Path, required=True)
    parser.add_argument("--candidate-output", type=Path,
                        help="Measure an experimental output module without changing the active kernel")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    old_prepare = load("before_prepare", args.baseline_prepare).prepare_mn
    old_output = load("before_output", args.baseline_output).output_fwd
    new_prepare, new_output = fast.prepare_mn, fast.output_fwd
    if args.candidate_output:
        new_output = load("candidate_output", args.candidate_output).output_fwd

    def prepare_before(*a, **kw):
        kw.pop("transfer_mask", None)
        return old_prepare(*a, **kw)

    torch.manual_seed(42)
    torch.set_num_threads(4)
    torch.cuda.set_stream(torch.cuda.Stream())
    t, parts, hv = args.seqlen, 8, 48
    length = t // parts
    q, k = [torch.nn.functional.normalize(torch.randn(1, t, 16, 128, device="cuda"), dim=-1).bfloat16()
            for _ in range(2)]
    v = torch.randn(1, t, hv, 128, device="cuda", dtype=torch.bfloat16)
    g = torch.nn.functional.logsigmoid(torch.randn(1, t, hv, device="cuda")) / 16
    beta = torch.randn_like(g).sigmoid()
    ab, gc = local_precompute(k, beta, g)
    cu = torch.arange(parts + 1, device="cuda", dtype=torch.int32) * length
    counts, mask = get_warmup_chunks(gc, cu, torch.zeros(parts, device="cuda", dtype=torch.bool),
                                      threshold=math.log(5e-5))
    h0 = torch.randn(parts, hv, 128, 128, device="cuda") * 0.1
    repeats = 10 if t <= 16384 else 2
    stages = {
        "prepare": {
            "before": lambda: old_prepare(k, v, ab, gc, beta, length, skip_last=True,
                                           beta_folded=True, num_warmup_chunks=counts),
            "after": lambda: new_prepare(k, v, ab, gc, beta, length, skip_last=True,
                                          beta_folded=True, num_warmup_chunks=counts, transfer_mask=mask),
        },
        "output": {
            "before": lambda: old_output(q, k, v, ab, gc, h0, length)[0],
            "after": lambda: new_output(q, k, v, ab, gc, h0, length)[0],
        },
    }
    results = {"shape": [1, t, 16, hv, 128], "partitions": parts,
               "scope": "before/after; no FlashInfer comparison; full history unless labeled truncated"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for stage, functions in stages.items():
        print("STAGE", stage, flush=True)
        before, after = functions["before"](), functions["after"]()
        difference = error(after[0], before[0]) if stage == "prepare" else error(after, before)
        assert difference < 0.001, difference
        results[stage] = {"relative_rms_vs_before": difference, **benchmark(functions, inner_repeats=repeats)}
        args.output.write_text(json.dumps(results, indent=2) + "\n")

    def forward(prepare, output, threshold):
        with patch.object(fast, "prepare_mn", prepare), patch.object(fast, "output_fwd", output):
            return chunk_gated_delta_rule(q, k, v, g, beta, fwd_backend="optimized",
                                          fwd_partitions=parts, fwd_decay_threshold=threshold)[0]

    for scope, threshold in [("full", None), ("truncated", 5e-5)]:
        functions = {
            "before": lambda: forward(prepare_before, old_output, threshold),
            "skip_m_only": lambda: forward(new_prepare, old_output, threshold),
            "both": lambda: forward(new_prepare, new_output, threshold),
        }
        print("FORWARD", scope, flush=True)
        reference = functions["before"]()
        checks = {name: error(fn(), reference) for name, fn in functions.items()}
        assert max(checks.values()) < 0.001, checks
        results[scope] = {"relative_rms_vs_before": checks, **benchmark(functions, inner_repeats=repeats)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n")
    summary = {stage: {name: value["median_ms"] for name, value in results[stage].items()
                       if isinstance(value, dict) and "median_ms" in value}
               for stage in ("prepare", "output", "full", "truncated")}
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()

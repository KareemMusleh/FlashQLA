"""Matched before/after M/N preparation and full forward, with optional truncation."""
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
from flash_qla import chunk_gated_delta_rule
from flash_qla.ops.gated_delta_rule.chunk.hopper.local_precompute import local_precompute
from flash_qla.ops.gated_delta_rule.chunk.hopper.cp_fwd import get_warmup_chunks

fast = importlib.import_module("flash_qla.ops.gated_delta_rule.chunk.hopper.fast_fwd")


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seqlen", type=int, default=16384)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location("mn_before_overlap", args.baseline)
    old = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old)
    torch.manual_seed(42)
    torch.set_num_threads(4)
    torch.cuda.set_stream(torch.cuda.Stream())
    t, parts = args.seqlen, 8
    length = t // parts
    q, k = [torch.nn.functional.normalize(torch.randn(1, t, 16, 128, device="cuda"), dim=-1).bfloat16()
            for _ in range(2)]
    v = torch.randn(1, t, 48, 128, device="cuda", dtype=torch.bfloat16)
    g = torch.nn.functional.logsigmoid(torch.randn(1, t, 48, device="cuda")) / 16
    beta = torch.randn_like(g).sigmoid()
    a, gc = local_precompute(k, beta, g)
    cu = torch.arange(parts + 1, device="cuda", dtype=torch.int32) * length
    counts, mask = get_warmup_chunks(gc, cu, torch.zeros(parts, device="cuda", dtype=torch.bool),
                                      threshold=math.log(5e-5))
    candidate_prepare = fast.prepare_mn
    if args.candidate:
        spec = importlib.util.spec_from_file_location("mn_overlap_candidate", args.candidate)
        new = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(new)
        candidate_prepare = new.prepare_mn
    candidate = partial(candidate_prepare, overlap_mn=True)
    results = {"shape": [1, t, 16, 48, 128], "partitions": parts,
               "scope": "same output/local kernels; full history unless labeled truncated; no FlashInfer comparison"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    repeats = 10 if t <= 16384 else 2
    for scope, extra, threshold in [("full", {}, None),
                                    ("truncated", {"num_warmup_chunks": counts, "transfer_mask": mask}, 5e-5)]:
        def stage(fn):
            return fn(k, v, a, gc, beta, length, skip_last=True, beta_folded=True, **extra)

        def forward(fn):
            with patch.object(fast, "prepare_mn", fn):
                return chunk_gated_delta_rule(q, k, v, g, beta, fwd_backend="optimized",
                                              fwd_partitions=parts, fwd_decay_threshold=threshold)[0]

        functions = {"prepare_before": lambda: stage(old.prepare_mn),
                     "prepare_after": lambda: stage(candidate),
                     "forward_before": lambda: forward(old.prepare_mn),
                     "forward_after": lambda: forward(candidate)}
        n, m = functions["prepare_before"]()
        nn, mm = functions["prepare_after"]()
        checks = {"N_rms": error(nn, n), "M_rms": error(mm, m),
                  "forward_rms": error(functions["forward_after"](), functions["forward_before"]())}
        assert max(checks.values()) < 0.001, checks
        print("TIMING", scope, checks, flush=True)
        results[scope] = {"checks": checks, **benchmark(functions, inner_repeats=repeats)}
        args.output.write_text(json.dumps(results, indent=2) + "\n")
        print(scope, {name: value["median_ms"] for name, value in results[scope].items()
                      if "median_ms" in value}, flush=True)


if __name__ == "__main__":
    main()

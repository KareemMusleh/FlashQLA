"""Matched training timings for output RMSNorm/SiLU gating and all gradients."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bench_implicit_bwd import interleaved, rms
from flash_qla.ops.gated_delta_rule.chunk.hopper.training_norm_gate import training_norm_gate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seqlens", type=int, nargs="+", default=[16384, 131072])
    parser.add_argument("--output", type=Path, default=Path("traces/gdn_backward_pipeline/norm_gate_stage/results.json"))
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.cuda.set_stream(torch.cuda.Stream())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    source = ROOT / "flash_qla/ops/gated_delta_rule/chunk/hopper/training_norm_gate.py"
    results = dict(gpu=torch.cuda.get_device_name(), torch=torch.__version__,
                   visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"), source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                   scope="B=1, Hv=48, D=128, BF16 activation/weight. Same saved intermediates and fixed upstream gradients. Includes dx, dz and normalization-weight gradient.",
                   method="CUDA graphs; nine alternating-order samples after 50 alternating warmup rounds")
    for length in args.seqlens:
        torch.manual_seed(617)
        x, z = [torch.randn(1, length, 48, 128, device="cuda", dtype=torch.bfloat16).requires_grad_() for _ in range(2)]
        weight = torch.randn(128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        inputs = (x, z, weight)
        do = torch.randn_like(x)
        functions = dict(torch=lambda: F.rms_norm(x, (128,), weight, eps=1e-6) * F.silu(z),
                         fused=lambda: training_norm_gate(x, z, weight, eps=1e-6))
        outputs = {name: fn() for name, fn in functions.items()}
        reference = torch.autograd.grad(outputs["torch"], inputs, do, retain_graph=True)
        actual = torch.autograd.grad(outputs["fused"], inputs, do, retain_graph=True)
        checks = {name: rms(a, b) for name, a, b in zip(("dx", "dz", "dw"), actual, reference)}
        assert max(checks.values()) < 0.01, checks
        assert all(torch.isfinite(t).all() for t in actual)
        entry = results[str(length)] = dict(output_relative_rms=rms(outputs["fused"], outputs["torch"]), gradient_relative_rms=checks)
        del reference, actual
        repeats = 10 if length <= 16384 else 2
        entry["forward"] = interleaved(functions, repeats)
        entry["backward"] = interleaved(
            {name: lambda name=name: torch.autograd.grad(outputs[name], inputs, do, retain_graph=True)
             for name in functions}, repeats)
        entry["forward_backward"] = interleaved(
            {name: lambda fn=fn: torch.autograd.grad(fn(), inputs, do) for name, fn in functions.items()}, repeats)
        args.output.write_text(json.dumps(results, indent=2) + "\n")
        print(length, {scope: {name: round(e["median_ms"], 4) for name, e in entry[scope].items()}
                       for scope in ("forward", "backward", "forward_backward")}, flush=True)
        del x, z, weight, inputs, do, functions, outputs
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

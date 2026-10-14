"""Measure reuse of the optimized forward inverse in the default backward."""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flash_qla import chunk_gated_delta_rule
from bench_implicit_bwd import interleaved, rms


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seqlens", type=int, nargs="+", default=[16384, 131072])
    parser.add_argument("--output", type=Path, default=Path("traces/gdn_cached_bwd/results.json"))
    parser.add_argument("--trace", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.cuda.set_stream(torch.cuda.Stream())
    root = Path(__file__).resolve().parents[1]
    results = {"gpu": torch.cuda.get_device_name(), "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
               "scope": "B=1,Hk=16,Hv=48,D=128,BF16. Optimized forward, full history; Q/K normalization and default CP/backward kernels.",
               "method": "CUDA graphs; 50 alternating warmups and nine alternating-order samples",
               "source_sha256": {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                                  for name in ["flash_qla/ops/gated_delta_rule/chunk/__init__.py",
                                               "flash_qla/ops/gated_delta_rule/chunk/hopper/local_precompute.py",
                                               "flash_qla/ops/gated_delta_rule/chunk/hopper/fast_fwd.py"]}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for length in args.seqlens:
        torch.manual_seed(42)
        q, k = [torch.nn.functional.normalize(torch.randn(1, length, 16, 128, device="cuda"), dim=-1).bfloat16().requires_grad_()
                for _ in range(2)]
        v = torch.randn(1, length, 48, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        g = (torch.nn.functional.logsigmoid(torch.randn(1, length, 48, device="cuda")) / 16).requires_grad_()
        beta = torch.randn_like(g).sigmoid().requires_grad_()
        inputs = (q, k, v, g, beta)
        do = torch.randn_like(v)
        variants = {"default": "default", "cached": "optimized"}

        def forward(backend, inputs=inputs):
            return chunk_gated_delta_rule(*inputs, fwd_backend="optimized", fwd_partitions=8,
                                         bwd_backend=backend, use_qk_l2norm_in_kernel=True)[0]

        outputs = {name: forward(backend) for name, backend in variants.items()}
        torch.testing.assert_close(outputs["cached"], outputs["default"], rtol=0, atol=0)

        def backward(name, outputs=outputs, inputs=inputs, do=do):
            return torch.autograd.grad(outputs[name], inputs, do, retain_graph=True)

        def step(backend, forward=forward, inputs=inputs, do=do):
            return torch.autograd.grad(forward(backend), inputs, do)

        reference, actual = backward("default"), backward("cached")
        checks = {name: rms(a, b) for name, a, b in zip(("dq", "dk", "dv", "dg", "db"), actual, reference)}
        assert all(torch.isfinite(a).all() for a in actual)
        assert max(checks.values()) < 0.02, checks
        entry = results[str(length)] = {"gradient_rms_vs_default": checks, "output_bitwise_equal": True,
                                        "retained_inverse_bytes": outputs["cached"].grad_fn.saved_tensors[7].numel() * 2}
        print("BENCH", length, checks, flush=True)
        repeats = 10 if length <= 16384 else 2
        entry["forward"] = interleaved({name: lambda backend=backend, forward=forward: forward(backend)
                                         for name, backend in variants.items()}, repeats)
        entry["backward"] = interleaved({name: lambda name=name, backward=backward: backward(name)
                                          for name in variants}, repeats)
        entry["forward_backward"] = interleaved({name: lambda backend=backend, step=step: step(backend)
                                                  for name, backend in variants.items()}, repeats)
        if args.trace:
            for name, backend in variants.items():
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as prof:
                    step(backend)
                    torch.cuda.synchronize()
                prof.export_chrome_trace(str(args.output.parent / f"{length}_{name}_trace.json"))
        args.output.write_text(json.dumps(results, indent=2) + "\n")
        print(json.dumps(entry, indent=2), flush=True)
        del reference, actual, outputs, inputs, q, k, v, g, beta, do
        del forward, backward, step
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

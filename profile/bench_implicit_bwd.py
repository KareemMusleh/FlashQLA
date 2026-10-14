"""Compare implicit and default backward, including all reconstruction work."""
import argparse
import hashlib
import importlib
import json
import os
import statistics
import sys
from functools import partial
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flash_qla import chunk_gated_delta_rule
from flash_qla.ops.gated_delta_rule.chunk.hopper.implicit_bwd import implicit_gdr_bwd, _implicit_bwd

chunk = importlib.import_module("flash_qla.ops.gated_delta_rule.chunk")


def rms(actual, ref):
    return ((actual.double() - ref.double()).norm() / ref.double().norm().clamp_min(1e-30)).item()


def interleaved(functions, repeats):
    graphs, outputs = {}, {}
    for name, fn in functions.items():
        for _ in range(2):
            fn()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            outputs[name] = fn()
        graphs[name] = graph
    for _ in range(50):
        for graph in graphs.values():
            graph.replay()
    samples = {name: [] for name in graphs}
    names = list(graphs)
    for cycle in range(9):
        for name in names[::1 if cycle % 2 else -1]:
            begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            begin.record()
            for _ in range(repeats):
                graphs[name].replay()
            end.record()
            end.synchronize()
            samples[name].append(begin.elapsed_time(end) / repeats)
    return {name: dict(median_ms=statistics.median(times), samples_ms=times)
            for name, times in samples.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seqlens", type=int, nargs="+", default=[16384, 131072])
    parser.add_argument("--output", type=Path, default=Path("traces/gdn_implicit_bwd/results.json"))
    parser.add_argument("--trace", action="store_true")
    parser.add_argument("--variant", choices=["implicit", "tma", "parallel", "queued"], default="implicit")
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.cuda.set_stream(torch.cuda.Stream())
    baseline = chunk.fused_gdr_bwd
    candidate = implicit_gdr_bwd
    if args.variant != "implicit":
        archive = {"tma": "traces/gdn_bwd_tma/fused_bwd_tma.py",
                   "parallel": "traces/gdn_bwd_parallel/parallel_bwd.py",
                   "queued": "traces/gdn_bwd_queued/fused_bwd_queued.py"}[args.variant]
        source_path = Path(__file__).resolve().parents[1] / archive
        module_name = "flash_qla.ops.gated_delta_rule.chunk.hopper._backward_experiment"
        spec = importlib.util.spec_from_file_location(module_name, source_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        candidate = (partial(module.fused_gdr_bwd, use_tma_stores=True)
                     if args.variant == "tma" else module.parallel_gdr_bwd
                     if args.variant == "parallel" else module.fused_gdr_bwd)
    results = {"gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
               "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
               "scope": "Optimized forward, full-history; B=1,Hk=16,Hv=48,D=128,BF16. Default CP/state reconstruction and gradient postprocessing retained.",
               "method": "CUDA graphs, nine alternating-order samples; fixed upstream gradients"}
    if args.variant == "implicit":
        source_path = Path(__file__).resolve().parents[1] / "flash_qla/ops/gated_delta_rule/chunk/hopper/implicit_bwd.py"
    results["candidate_source_sha256"] = hashlib.sha256(source_path.read_bytes()).hexdigest()
    results["variant"] = args.variant
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for length in args.seqlens:
        torch.manual_seed(42)
        q, k = [torch.nn.functional.normalize(
            torch.randn(1, length, 16, 128, device="cuda"), dim=-1,
        ).bfloat16().requires_grad_() for _ in range(2)]
        v = torch.randn(1, length, 48, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        g = (torch.nn.functional.logsigmoid(torch.randn(1, length, 48, device="cuda")) / 16).requires_grad_()
        beta = torch.randn_like(g).sigmoid().requires_grad_()
        inputs = (q, k, v, g, beta)
        do = torch.randn_like(v)

        def forward(inputs=inputs):
            return chunk_gated_delta_rule(*inputs, fwd_backend="optimized", fwd_partitions=8,
                                         use_qk_l2norm_in_kernel=True)[0]

        output = forward()

        def backward(kernel, output=output, inputs=inputs, do=do):
            chunk.fused_gdr_bwd = kernel
            return torch.autograd.grad(output, inputs, do, retain_graph=True)

        def step(kernel, forward=forward, inputs=inputs, do=do):
            chunk.fused_gdr_bwd = kernel
            return torch.autograd.grad(forward(), inputs, do)

        reference = backward(baseline)
        actual = backward(candidate)
        checks = {name: rms(a, b) for name, a, b in zip(
            ("dq", "dk", "dv", "dg", "db"), actual, reference)}
        assert max(checks.values()) < 0.04, checks
        assert all(torch.isfinite(t).all() for t in actual)
        entry = results[str(length)] = {"gradient_rms_vs_default": checks}

        # Capture the exact main-kernel inputs produced by the existing driver.
        fixture = {}

        def collect(fixture=fixture, **kwargs):
            fixture.update(kwargs)
            return baseline(**kwargs)

        backward(collect)
        entry["partition_count"] = (fixture["cu_seqlens"].numel() - 1
                                      if fixture["cu_seqlens"] is not None else 1)
        compiled = None
        if args.variant == "implicit":
            compiled = _implicit_bwd(
                length, 16, 48, entry["partition_count"], fixture["scale"], False,
                fixture["dht"] is not None,
                str(fixture["cu_seqlens"].dtype).split('.')[-1] if fixture["cu_seqlens"] is not None else "int32",
            )
            compiled.export_sources(kernel_path=str(args.output.parent / f"{length}_implicit.cu"))
        functions = {"default": baseline, args.variant: candidate}
        print("BENCH", length, checks, flush=True)
        entry["main_kernel"] = interleaved(
            {name: lambda fn=fn, fixture=fixture: fn(**fixture) for name, fn in functions.items()},
            10 if length <= 16384 else 2,
        )
        entry["backward"] = interleaved(
            {name: lambda fn=fn, backward=backward: backward(fn) for name, fn in functions.items()},
            10 if length <= 16384 else 2,
        )
        entry["forward_backward"] = interleaved(
            {name: lambda fn=fn, step=step: step(fn) for name, fn in functions.items()},
            10 if length <= 16384 else 2,
        )
        if args.trace:
            for name, fn in functions.items():
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                        torch.profiler.ProfilerActivity.CUDA]) as prof:
                    step(fn)
                    torch.cuda.synchronize()
                prof.export_chrome_trace(str(args.output.parent / f"{length}_{name}_trace.json"))
        args.output.write_text(json.dumps(results, indent=2) + "\n")
        print(json.dumps(entry, indent=2), flush=True)
        del reference, actual, output, fixture, inputs, q, k, v, g, beta, do
        del forward, backward, step, collect, compiled, functions
        torch.cuda.empty_cache()
    chunk.fused_gdr_bwd = baseline


if __name__ == "__main__":
    main()

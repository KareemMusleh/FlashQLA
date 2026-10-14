"""Compare original, cast-fused, and group/norm-fused GDN backward paths."""

import argparse
import importlib
import json
import os
import statistics
import sys
from functools import partial
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from flash_qla import chunk_gated_delta_rule
from flash_qla.utils.math import l2norm_bwd

chunk = importlib.import_module("flash_qla.ops.gated_delta_rule.chunk")
optimized_bwd = chunk.chunk_gated_delta_rule_bwd


def select(mode):
    if mode == "fused":
        chunk.chunk_gated_delta_rule_bwd = optimized_bwd
        return

    def separate(*args, q_rstd=None, k_rstd=None, **kwargs):
        dq, dk, dv, db, dg, dh = optimized_bwd(*args, **kwargs)
        q, k = kwargs["q"], kwargs["k"]
        if q_rstd is not None:
            dtype = q.dtype if mode == "cast" else None
            dq = l2norm_bwd(q, q_rstd, dq, output_dtype=dtype)
            dk = l2norm_bwd(k, k_rstd, dk, output_dtype=dtype)
        return dq, dk, dv, db, dg, dh

    chunk.chunk_gated_delta_rule_bwd = separate


def benchmark(fn):
    for _ in range(3):
        _result = fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        _result = fn()
    for _ in range(5):
        graph.replay()
    times = []
    for _ in range(7):
        start, end = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
        start.record()
        for _ in range(10):
            graph.replay()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end) / 10)
    return {"median_ms": statistics.median(times), "samples_ms": times}


def difference(a, b):
    af, bf = a.float(), b.float()
    return {
        "relative_rms": (
            (af - bf).square().mean() / af.square().mean().clamp_min(1e-30)
        )
        .sqrt()
        .item(),
        "max_abs": (af - bf).abs().max().item(),
        "finite": bool(b.isfinite().all()),
    }


def benchmark_interleaved(fn):
    """Rotate graph replay order to reduce bias from changing GPU conditions."""
    modes = ["original", "cast", "fused"]
    graphs, outputs = {}, {}
    for mode in modes:
        select(mode)
        for _ in range(3):
            fn()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            outputs[mode] = fn()
        graphs[mode] = graph
    for _ in range(20):
        for graph in graphs.values():
            graph.replay()
    samples = {mode: [] for mode in modes}
    for cycle in range(9):
        order = modes[cycle % 3 :] + modes[: cycle % 3]
        if cycle % 2:
            order = order[::-1]
        for mode in order:
            start, end = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
            start.record()
            for _ in range(10):
                graphs[mode].replay()
            end.record()
            end.synchronize()
            samples[mode].append(start.elapsed_time(end) / 10)
    return {
        mode: {"median_ms": statistics.median(times), "samples_ms": times}
        for mode, times in samples.items()
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seqlens", type=int, nargs="+", default=[16384, 131072])
    parser.add_argument("--interleaved-only", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.cuda.set_stream(torch.cuda.Stream())
    results = {
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "scope": "GDN operator including Q/K normalization; B=1,Hk=16,Hv=48,D=128,BF16",
        "method": "CUDA graphs, seven samples of ten replays, opposite-order sweeps",
    }
    for length in args.seqlens:
        torch.manual_seed(42)
        q, k = [
            torch.randn(
                1,
                length,
                16,
                128,
                device="cuda",
                dtype=torch.bfloat16,
                requires_grad=True,
            )
            for _ in range(2)
        ]
        v = torch.randn(
            1, length, 48, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True
        )
        g = (
            torch.nn.functional.logsigmoid(torch.randn(1, length, 48, device="cuda"))
            / 16
        ).requires_grad_()
        beta = torch.randn_like(g).sigmoid().requires_grad_()
        inputs = (q, k, v, g, beta)
        do = torch.randn_like(v)

        def forward(inputs=inputs):
            return chunk_gated_delta_rule(*inputs, use_qk_l2norm_in_kernel=True)[0]

        def step(forward=forward, inputs=inputs, do=do):
            return torch.autograd.grad(forward(), inputs, do)

        select("original")
        ref_output = forward()
        reference = tuple(
            t.detach().clone() for t in torch.autograd.grad(ref_output, inputs, do)
        )
        ref_output = ref_output.detach()
        entry = results[str(length)] = {}
        sweeps = [["original", "cast", "fused"]]
        if not args.interleaved_only:
            sweeps.append(["fused", "cast", "original"])
        for sweep, modes in enumerate(sweeps):
            for mode in modes:
                select(mode)
                print("RUN", length, sweep, mode, flush=True)
                y = forward()
                grads = torch.autograd.grad(y, inputs, do, retain_graph=True)
                case = entry.setdefault(mode, {"runs": []})
                if sweep == 0:
                    checks = {
                        "output": difference(ref_output, y),
                        **{
                            n: difference(a, b)
                            for n, a, b in zip(
                                ["dq", "dk", "dv", "dg", "db"], reference, grads
                            )
                        },
                    }
                    assert all(
                        c["finite"] and c["relative_rms"] < 0.002
                        for c in checks.values()
                    ), checks
                    case["correctness"] = checks
                del grads
                if not args.interleaved_only:
                    run = {
                        "backward": benchmark(
                            partial(
                                torch.autograd.grad, y, inputs, do, retain_graph=True
                            )
                        ),
                        "forward_backward": benchmark(step),
                    }
                    case["runs"].append(run)
                    print(length, mode, run, flush=True)
                if sweep == 0:
                    with torch.profiler.profile(
                        activities=[
                            torch.profiler.ProfilerActivity.CPU,
                            torch.profiler.ProfilerActivity.CUDA,
                        ]
                    ) as prof:
                        for _ in range(3):
                            with torch.profiler.record_function(
                                f"GDN/{mode}/forward_backward"
                            ):
                                step()
                        torch.cuda.synchronize()
                    prof.export_chrome_trace(
                        str(args.output / f"{length}_{mode}_trace.json")
                    )
                del y
                (args.output / "results.json").write_text(
                    json.dumps(results, indent=2) + "\n"
                )
        del reference, ref_output
        print("INTERLEAVED", length, flush=True)
        entry["interleaved_forward_backward"] = benchmark_interleaved(step)
        select("fused")
        y = forward()
        entry["interleaved_backward"] = benchmark_interleaved(
            partial(torch.autograd.grad, y, inputs, do, retain_graph=True)
        )
        print(
            length,
            entry["interleaved_forward_backward"],
            entry["interleaved_backward"],
            flush=True,
        )
        (args.output / "results.json").write_text(json.dumps(results, indent=2) + "\n")
        del y, inputs, q, k, v, g, beta, do, forward, step
        torch.cuda.empty_cache()
    select("fused")


if __name__ == "__main__":
    main()

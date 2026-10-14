"""Whole-layer direct-buffer backward versus eager and matched packing ablation."""
import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import statistics

import torch

from bench_gdn_projection_training import TrainingLayer, saved_tensors
from bench_implicit_bwd import rms
from flash_qla.ops.gdn_layer import direct_gdn_layer


class Layer(TrainingLayer):
    def training_forward(self, mode):
        if mode == "separate":
            return super().training_forward(mode)
        direct = mode in ("direct", "direct_reuse")
        return direct_gdn_layer(self.x, self.weights, self.conv_weights,
                                self.a_log, self.dt_bias, self.norm_weight, self.wo,
                                key_heads=self.hk, value_heads=self.hv,
                                eps=self.args.norm_eps, partitions=self.args.partitions,
                                direct_gradients=direct,
                                reuse_gradient_workspace=mode == "direct_reuse")


def paired_graphs(functions, samples=21):
    """Capture both execution orders; per-stage external events avoid clock bias.

    Calls return results which are discarded between variants. One graph pool
    per order therefore reuses workspace rather than retaining every variant's
    forward graph and backward workspace simultaneously at long sequences.
    """
    names = list(functions)
    for fn in functions.values():
        fn()
    torch.cuda.synchronize()
    graphs = []
    for order in (names, names[::-1]):
        graph = torch.cuda.CUDAGraph()
        events = {n: (torch.cuda.Event(enable_timing=True, external=True),
                      torch.cuda.Event(enable_timing=True, external=True)) for n in names}
        with torch.cuda.graph(graph):
            for name in order:
                start, end = events[name]
                start.record()
                result = functions[name]()
                end.record()
                del result
        graphs.append((graph, events))
    for _ in range(5):
        for graph, _ in graphs:
            graph.replay()
    times = {n: [] for n in names}
    for i in range(samples):
        graph, events = graphs[i % 2]
        graph.replay()
        torch.cuda.synchronize()
        for name, (start, end) in events.items():
            times[name].append(start.elapsed_time(end))
    return {name: {"median_ms": statistics.median(t), "samples_ms": t}
            for name, t in times.items()}


def compare(layer, trace_dir, memory_only=False, modes=("separate", "packed", "direct")):
    params = layer.parameters_for("separate", True)
    upstream = torch.randn_like(layer.x)
    def forward(mode):
        return layer.training_forward(mode)
    def step(mode):
        return torch.autograd.grad(forward(mode), params, upstream)
    checks = {}
    reference_mode = modes[0]
    output_ref = forward(reference_mode).detach()
    reference = step(reference_mode)
    for mode in modes:
        gc.collect()
        torch.cuda.synchronize()
        # Drain delayed frees from autograd's CUDA streams before taking the
        # baseline; otherwise memory_allocated includes dead previous graphs.
        torch.cuda.empty_cache()
        live_before_forward = torch.cuda.memory_allocated()
        # Native allocator history observes custom backward threads without
        # changing operator dispatch or RMSNorm's autograd implementation.
        torch.cuda.memory._record_memory_history(
            enabled="all", context=None, max_entries=100000, clear_history=True,
        )
        try:
            output = forward(mode)
            torch.cuda.synchronize()
            allocated = torch.cuda.memory_allocated()
            grad = torch.autograd.grad(output, params, upstream)
            torch.cuda.synchronize()
            trace = torch.cuda.memory._snapshot()["device_traces"][torch.cuda.current_device()]
        finally:
            torch.cuda.memory._record_memory_history(enabled=None)
        live, peak = 0, 0
        for event in trace:
            if event["action"] == "alloc":
                live += event["size"]
                peak = max(peak, live)
            elif event["action"] == "free_requested":
                live -= event["size"]
        assert len(trace) < 100000, "allocation history was truncated"
        observed = torch.cuda.memory_allocated() - live_before_forward
        # Trace sizes are requested bytes; live allocation counters include
        # allocator rounding of small bias/norm gradients and metadata.
        assert abs(live - observed) < 2 * 1024 * 1024, (
            "allocation trace baseline mismatch", live, observed,
        )
        checks[mode] = {
            "output_rms": rms(output, output_ref),
            "gradient_rms": [rms(a,b) for a,b in zip(grad,reference)],
            "finite": all(torch.isfinite(g).all().item() for g in grad),
            "allocation_before_backward_bytes": allocated,
            "whole_step_peak_increment_bytes": peak,
            "trace_delta_rounding_bytes": observed - live,
            "saved": saved_tensors(layer,mode,True),
        }
        assert checks[mode]["finite"] and max(checks[mode]["gradient_rms"]) < .02, checks[mode]
        if mode == reference_mode:
            assert max(checks[mode]["gradient_rms"]) < .001, "instrumentation changed reference gradients"
        del grad, output
    assert all(c["saved"]["unique_activation_storage_bytes"] <=
               checks[reference_mode]["saved"]["unique_activation_storage_bytes"]
               for c in checks.values())
    del reference, output_ref
    if memory_only:
        return {"checks":checks,"timings":{}}
    timings = {}
    for scope, fn in (("forward",forward), ("forward_backward",step)):
        torch.cuda.empty_cache()
        timings[scope] = paired_graphs({m: lambda m=m,fn=fn: fn(m) for m in modes})
        print(scope,{m:round(v["median_ms"],4) for m,v in timings[scope].items()},flush=True)
    if trace_dir is not None:
        for mode in modes:
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                    torch.profiler.ProfilerActivity.CUDA],
                                        record_shapes=True) as prof:
                step(mode)
                torch.cuda.synchronize()
            prof.export_chrome_trace(str(trace_dir / f"{layer.args.seqlen}_{mode}.json"))
    return {"checks":checks,"timings":timings}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seqlens",nargs="+",type=int,default=[16384])
    parser.add_argument("--trace",action="store_true")
    parser.add_argument("--memory-only",action="store_true")
    parser.add_argument("--output",type=Path,default=Path("traces/gdn_direct_training/results.json"))
    args=parser.parse_args()
    torch.set_num_threads(4)
    torch.cuda.set_stream(torch.cuda.Stream())
    root=Path(__file__).resolve().parents[1]
    paths=[Path(__file__),root/"flash_qla/ops/gdn_layer.py",root/"profile/bench_gdn_projection_training.py",
           root/"flash_qla/ops/gated_delta_rule/chunk/__init__.py"]
    results={"gpu":torch.cuda.get_device_name(),"torch":torch.__version__,
             "causal_conv1d":importlib.metadata.version("causal-conv1d"),
             "visible_devices":os.environ.get("CUDA_VISIBLE_DEVICES"),
             "scope":"Synthetic full YDT layer, single unpadded sequence, hidden2560 Hk16 Hv48 D128 conv4; "
                     "BF16/FP32 gates; full-history optimized forward and default uncached GDR backward; "
                     "dt_max=.01 fixture; no optimizer, distributed CP, state cache, or activation-stat logging.",
             "method":"21 paired CUDA graph samples, alternating execution order, external per-variant events",
             "memory_method":"Native allocator alloc/free_requested event history (requested bytes) for eager forward/backward; resident input/parameters/reference gradients excluded; event delta checked against live allocation counters allowing <2MiB CUDA allocator rounding; no dispatch mode",
             "sources":{str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (*paths, root/"flash_qla/ops/gated_delta_rule/chunk/hopper/kkt_solve.py")}}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    for length in args.seqlens:
        torch.manual_seed(42)
        config=argparse.Namespace(seqlen=length,hidden_size=2560,key_heads=16,value_heads=48,
                                  conv_size=4,norm_eps=1e-6,dt_max=.01,partitions=8,
                                  conv_backend="dao",projection_mode="separate")
        layer=Layer(config)
        print("LAYER",length,flush=True)
        results[str(length)]=compare(
            layer, args.output.parent if args.trace else None, args.memory_only,
            modes=("separate", "direct", "direct_reuse"),
        )
        args.output.write_text(json.dumps(results,indent=2)+"\n")
        del layer
        torch.cuda.empty_cache()


if __name__=="__main__":
    main()

"""Matched projection and complete GDN training experiment, without extra caches.

Separate parameter candidate includes runtime packing in backward. Packed-parameter
candidate is a distinct parameter layout, not a stale/prepacked weight cache.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

import torch
import torch.nn.functional as F

from bench_gdn_layer import Layer
from bench_implicit_bwd import interleaved, rms
from flash_qla import chunk_gated_delta_rule
from flash_qla.ops.input_projection import combined_input_projection


class TrainingLayer(Layer):
    def __init__(self, args):
        super().__init__(args)
        self.x.requires_grad_()
        for t in (*self.weights, *self.conv_weights, self.wo, self.norm_weight,
                  self.a_log, self.dt_bias):
            t.requires_grad_()
        # Authoritative parameter for the packed variant; never used by the
        # separate-parameter variant and not a forward/backward cache.
        self.packed_weight.requires_grad_()

    def project_training(self, mode):
        if mode == "combined_backward":
            return combined_input_projection(self.x, *self.weights)
        if mode == "combined_dx":
            return combined_input_projection(self.x, *self.weights, combine_weight_gradients=False)
        return self.project("packed" if mode == "packed_parameter" else "separate")

    def parameters_for(self, mode, full):
        ws = (self.packed_weight,) if mode == "packed_parameter" else tuple(self.weights)
        params = (self.x, *ws)
        if full:
            params += (*self.conv_weights, self.a_log, self.dt_bias, self.norm_weight, self.wo)
        return params

    def training_forward(self, mode):
        p = self.project_training(mode)
        q, k, v = self.conv(p, "dao")
        q = q.view(1, -1, self.hk, self.d)
        k = k.view_as(q)
        v = v.view(1, -1, self.hv, self.d)
        g, beta = self.gates(p)
        o = chunk_gated_delta_rule(
            q, k, v, g, beta, fwd_backend="optimized", bwd_backend="default",
            enable_fwd_cp_cache=False, fwd_partitions=self.args.partitions,
            fwd_decay_threshold=None, use_qk_l2norm_in_kernel=True,
            output_final_state=False,
        )[0]
        z = p[3].view_as(o)
        gated = F.rms_norm(o, (self.d,), self.norm_weight, self.args.norm_eps) * F.silu(z)
        return self.output_project(gated)


def saved_tensors(layer, mode, full):
    params = layer.parameters_for(mode, full)
    parameter_storages = {t.untyped_storage().data_ptr() for t in params[1:]}
    saved = []
    def pack(t):
        saved.append((t.untyped_storage().data_ptr(), t.untyped_storage().nbytes()))
        return t
    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        output = layer.training_forward(mode) if full else layer.project_training(mode)
    storages = {ptr: n for ptr, n in saved if ptr not in parameter_storages}
    return {"saved_tensor_count": len(saved), "unique_activation_storage_bytes": sum(storages.values())}


def compare(layer, full, trace_dir=None):
    modes = ("separate", "combined_backward", "combined_dx", "packed_parameter")
    upstream = (torch.randn_like(layer.x) if full else
                tuple(torch.randn_like(y) for y in layer.project_training("separate")))
    def forward(mode):
        return layer.training_forward(mode) if full else layer.project_training(mode)
    def step(mode):
        return torch.autograd.grad(forward(mode), layer.parameters_for(mode, full), upstream)
    outputs = {mode: forward(mode) for mode in modes}
    checks, reference = {}, None
    for mode in modes:
        grad = torch.autograd.grad(outputs[mode], layer.parameters_for(mode, full), upstream,
                                   retain_graph=True)
        # Convert packed dW into the same six conceptual parameter gradients.
        if mode == "packed_parameter":
            grad = (grad[0], *grad[1].split(layer.projection_sizes), *grad[2:])
        if reference is None:
            reference = grad
        output = (outputs[mode],) if full else outputs[mode]
        ref_output = (outputs["separate"],) if full else outputs["separate"]
        checks[mode] = dict(
            output_rms=max(rms(a, b) for a, b in zip(output, ref_output)),
            gradient_rms=[rms(a, b) for a, b in zip(grad, reference)],
            finite=all(torch.isfinite(t).all().item() for t in grad),
            saved=saved_tensors(layer, mode, full),
        )
        assert checks[mode]["finite"] and max(checks[mode]["gradient_rms"]) < .02, checks[mode]
    assert (checks["combined_backward"]["saved"]["unique_activation_storage_bytes"] <=
            checks["separate"]["saved"]["unique_activation_storage_bytes"])
    del grad, reference
    workspace = {}
    for mode in modes:
        torch.cuda.synchronize()
        allocated = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        gradients = torch.autograd.grad(outputs[mode], layer.parameters_for(mode, full),
                                        upstream, retain_graph=True)
        torch.cuda.synchronize()
        workspace[mode] = torch.cuda.max_memory_allocated() - allocated
        del gradients
    repeats = 5 if layer.args.seqlen <= 16384 else 2
    timings = {}
    for scope, op in (
        ("forward", forward),
        ("backward", lambda mode: torch.autograd.grad(
            outputs[mode], layer.parameters_for(mode, full), upstream, retain_graph=True)),
        ("forward_backward", step),
    ):
        timings[scope] = interleaved({mode: lambda mode=mode, op=op: op(mode)
                                    for mode in modes}, repeats)
        print(scope, {m: round(t["median_ms"], 4) for m, t in timings[scope].items()}, flush=True)
    if trace_dir is not None:
        for mode in modes:
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                    torch.profiler.ProfilerActivity.CUDA],
                                        record_shapes=True) as prof:
                step(mode)
                torch.cuda.synchronize()
            prof.export_chrome_trace(str(trace_dir / f"{layer.args.seqlen}_{'layer' if full else 'projection'}_{mode}.json"))
    return dict(checks=checks, timings=timings, backward_peak_increment_bytes=workspace)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seqlens", type=int, nargs="+", default=[16384])
    parser.add_argument("--projection-only", action="store_true")
    parser.add_argument("--trace", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("traces/gdn_projection_training/results.json"))
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.cuda.set_stream(torch.cuda.Stream())
    root = Path(__file__).resolve().parents[1]
    results = dict(gpu=torch.cuda.get_device_name(), torch=torch.__version__,
                   visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
                   scope="Synthetic YDT operation order; Hk=16,Hv=48,D=128,hidden=2560,conv=4. "
                         "BF16 activations/weights, FP32 gate parameters; fixed upstream gradients; "
                         "full-history optimized forward, default uncached backward; "
                         "single sequence, no distributed CP, optimizer or instrumentation timing. "
                         "dt_max=.01 follows the existing forward fixture, not original YDT's .1.",
                   gradient_order=["input", "wq", "wk", "wv", "wz", "wa", "wb",
                                   "conv_q", "conv_k", "conv_v", "a_log", "dt_bias", "norm_weight", "wo"],
                   sources={str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                            for p in (Path(__file__), root / "flash_qla/ops/input_projection.py",
                                      root / "profile/bench_gdn_layer.py",
                                      root / "flash_qla/ops/gated_delta_rule/chunk/__init__.py")})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for length in args.seqlens:
        torch.manual_seed(42)
        config = argparse.Namespace(seqlen=length, hidden_size=2560, key_heads=16, value_heads=48,
                                    conv_size=4, norm_eps=1e-6, dt_max=.01, partitions=8,
                                    conv_backend="dao", projection_mode="separate")
        layer = TrainingLayer(config)
        entry = results[str(length)] = {}
        print("PROJECTIONS", length, flush=True)
        trace_dir = args.output.parent if args.trace else None
        entry["projections"] = compare(layer, False, trace_dir)
        if not args.projection_only:
            print("COMPLETE LAYER", length, flush=True)
            entry["layer"] = compare(layer, True, trace_dir)
        args.output.write_text(json.dumps(results, indent=2) + "\n")
        del layer
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

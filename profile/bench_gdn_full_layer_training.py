"""Full GDN-layer training comparison with an identical forward and cache policy.

Only GDN backward postprocessing differs. Includes input/parameter gradients,
projections, causal convolution/SiLU, gate preparation, output RMSNorm/gating.
"""
import argparse
import hashlib
from contextlib import contextmanager
import json
import os
from pathlib import Path
import sys

import torch
import torch.nn.functional as F
from causal_conv1d import causal_conv1d_fn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bench_fair_gdn_backward import reference, chunk, truncation
from bench_implicit_bwd import interleaved, rms
from flash_qla import chunk_gated_delta_rule
from flash_qla.ops.gated_delta_rule.chunk.hopper.fast_fwd import _training_cp_context
from flash_qla.ops.gated_delta_rule.chunk.hopper.training_norm_gate import training_norm_gate


@contextmanager
def backward_driver(driver):
    previous = chunk.chunk_gated_delta_rule_bwd
    chunk.chunk_gated_delta_rule_bwd = driver
    try:
        yield
    finally:
        chunk.chunk_gated_delta_rule_bwd = previous


class Layer(torch.nn.Module):
    def __init__(self, hidden, dt, context):
        super().__init__()
        self.context = context
        self.fused_norm_gate = False
        self.qk_normalized = False
        self.q = torch.nn.Linear(hidden, 2048, bias=False, device="cuda", dtype=torch.bfloat16)
        self.k = torch.nn.Linear(hidden, 2048, bias=False, device="cuda", dtype=torch.bfloat16)
        self.v = torch.nn.Linear(hidden, 6144, bias=False, device="cuda", dtype=torch.bfloat16)
        self.z = torch.nn.Linear(hidden, 6144, bias=False, device="cuda", dtype=torch.bfloat16)
        self.a = torch.nn.Linear(hidden, 48, bias=False, device="cuda", dtype=torch.bfloat16)
        self.b = torch.nn.Linear(hidden, 48, bias=False, device="cuda", dtype=torch.bfloat16)
        self.out = torch.nn.Linear(6144, hidden, bias=False, device="cuda", dtype=torch.bfloat16)
        self.conv = torch.nn.ParameterList([
            torch.nn.Parameter(torch.randn(channels, 4, device="cuda", dtype=torch.bfloat16) * 0.5)
            for channels in (2048, 2048, 6144)])
        self.norm = torch.nn.Parameter(torch.ones(128, device="cuda", dtype=torch.bfloat16))
        self.A_log = torch.nn.Parameter(torch.rand(48, device="cuda") * 2)
        self.dt_bias = torch.nn.Parameter(torch.full((48,), dt, device="cuda").expm1().log())

    def input_projections(self, x):
        return [fn(x) for fn in (self.q, self.k, self.v, self.z, self.a, self.b)]

    def convolution_outputs(self, q, k, v):
        return [causal_conv1d_fn(t.transpose(1, 2), w, activation="silu").transpose(1, 2).contiguous()
                for t, w in zip((q, k, v), self.conv)]

    def forward(self, x):
        with torch.profiler.record_function("layer/input_projections"):
            q, k, v, z, a, b = self.input_projections(x)
        with torch.profiler.record_function("layer/causal_convolution_silu"):
            q, k, v = self.convolution_outputs(q,k,v)
        with torch.profiler.record_function("layer/decay_beta"):
            g = (-self.A_log.exp() * F.softplus(a.float() + self.dt_bias)).contiguous()
            beta = b.float().sigmoid().contiguous()
        with torch.profiler.record_function("layer/gdn_including_qk_normalization"):
            o, _ = chunk_gated_delta_rule(
                q.reshape(1, -1, 16, 128), k.reshape(1, -1, 16, 128), v.reshape(1, -1, 48, 128),
                g, beta, use_qk_l2norm_in_kernel=not self.qk_normalized, cp_context=self.context,
                enable_fwd_cp_cache=True, fwd_backend="default",
            )
        with torch.profiler.record_function("layer/output_rmsnorm_silu_gate"):
            if self.fused_norm_gate:
                gated = training_norm_gate(o, z.reshape_as(o), self.norm, eps=1e-6)
            else:
                normed = F.rms_norm(o, (128,), self.norm, eps=1e-6)
                gated = normed * F.silu(z.reshape_as(o))
        with torch.profiler.record_function("layer/output_projection"):
            return self.out(gated.flatten(2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seqlens", type=int, nargs="+", default=[16384, 131072])
    parser.add_argument("--hidden-size", type=int, default=2560)
    parser.add_argument("--dt", type=float, default=0.001)
    parser.add_argument("--partitions", type=int, default=8)
    parser.add_argument("--trace", action="store_true")
    parser.add_argument("--compare-norm-gate", action="store_true",
                        help="Compare fused norm/gate training with identical current GDN forward/backward")
    parser.add_argument("--output", type=Path, default=Path("traces/gdn_backward_pipeline/full_layer/results.json"))
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.cuda.set_stream(torch.cuda.Stream())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    baseline, _, hashes = reference("HEAD", args.output.parent)
    current = chunk.chunk_gated_delta_rule_bwd
    results = dict(gpu=torch.cuda.get_device_name(), visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
                   hidden_size=args.hidden_size, key_heads=16, value_heads=48, head_dim=128,
                   dt=args.dt, partitions=args.partitions, reference_source_sha256=hashes,
                   compare_norm_gate=args.compare_norm_gate,
                   source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                  for p in [Path(__file__), ROOT / "flash_qla/ops/gated_delta_rule/chunk/hopper/training_norm_gate.py"]},
                   scope="Full layer forward/backward including all projections, Dao causal-conv1d/SiLU, decay/beta preparation, GDN with Q/K normalization, RMSNorm/output SiLU gate and output projection. Input and parameter gradients. Loss and optimizer excluded. Same standard forward, caches, partitions, precision; GDN backward postprocessing is the only variant.")
    if args.compare_norm_gate:
        results["scope"] = "Full layer forward/backward including projections and parameter/input gradients. Both variants use current GDN backward and identical standard GDN forward/cache/partitions. Only output RMSNorm/SiLU gate implementation differs; both retain normalized output, SiLU output and FP32 statistics. BF16 intermediate rounding is preserved. Loss and optimizer excluded."
    with truncation(False):
        for length in args.seqlens:
            torch.manual_seed(442)
            x = torch.randn(1, length, args.hidden_size, device="cuda", dtype=torch.bfloat16, requires_grad=True)
            layer = Layer(args.hidden_size, args.dt, _training_cp_context(x.device, length, args.partitions))
            inputs = (x, *layer.parameters())
            upstream = torch.randn_like(x)
            print("BENCH", length, flush=True)

            def step(driver, fused_norm_gate=False):
                layer.fused_norm_gate = fused_norm_gate
                with backward_driver(driver):
                    output = layer(x)
                    return torch.autograd.grad(output, inputs, upstream)

            variants = (dict(torch_norm_gate=(current, False), fused_norm_gate=(current, True))
                        if args.compare_norm_gate else
                        dict(flashqla_committed=(baseline, False), optimized_postprocessing=(current, False)))
            first, second = variants.values()
            with torch.no_grad():
                layer.fused_norm_gate = first[1]
                first_output = layer(x)
                layer.fused_norm_gate = second[1]
                second_output = layer(x)
                output_error = rms(second_output, first_output)
                assert output_error < 0.01, output_error
                del first_output, second_output
            expected, actual = step(*first), step(*second)
            names = ["input", *[name for name, _ in layer.named_parameters()]]
            checks = {name: rms(a, b) for name, a, b in zip(names, actual, expected)}
            assert all(torch.isfinite(t).all() for t in actual)
            assert max(checks.values()) < 0.01, checks
            del actual, expected
            functions = {name: lambda settings=settings: step(*settings) for name, settings in variants.items()}
            entry = results[str(length)] = dict(output_relative_rms=output_error, gradient_rms=checks,
                forward_backward=interleaved(functions, 5 if length <= 16384 else 2))
            if args.trace:
                for name, settings in variants.items():
                    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                            torch.profiler.ProfilerActivity.CUDA]) as prof:
                        step(*settings)
                        torch.cuda.synchronize()
                    prof.export_chrome_trace(str(args.output.parent / f"{length}_{name}.json"))
            args.output.write_text(json.dumps(results, indent=2) + "\n")
            print({name: e["median_ms"] for name, e in entry["forward_backward"].items()}, flush=True)
            del layer, inputs, x, upstream, step, functions
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

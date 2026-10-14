"""Compare public training backends and a synthetic GDN layer's optimization trajectory.

The trajectory is a regression smoke test, not a language-model convergence test.
"""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flash_qla import chunk_gated_delta_rule
from bench_implicit_bwd import interleaved, rms


BACKENDS = {
    "flashqla": dict(fwd_backend="default", bwd_backend="default"),
    "optimized_forward": dict(fwd_backend="optimized", bwd_backend="default"),
    "optimized_inverse_only": dict(fwd_backend="optimized", bwd_backend="optimized", enable_fwd_cp_cache=False),
    "optimized_cached": dict(fwd_backend="optimized", bwd_backend="optimized"),
}


def call(inputs, backend, partitions=8):
    return chunk_gated_delta_rule(
        *inputs, **BACKENDS[backend], fwd_partitions=partitions,
        fwd_decay_threshold=None, use_qk_l2norm_in_kernel=True,
    )[0]


def benchmark(length, trace_dir, decay):
    torch.manual_seed(42)
    q, k = [F.normalize(torch.randn(1, length, 16, 128, device="cuda"), dim=-1)
            .bfloat16().requires_grad_() for _ in range(2)]
    v = torch.randn(1, length, 48, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    g = (F.logsigmoid(torch.randn(1, length, 48, device="cuda")) * decay).requires_grad_()
    beta = torch.randn_like(g).sigmoid().requires_grad_()
    inputs = (q, k, v, g, beta)
    upstream = torch.randn_like(v)
    outputs = {name: call(inputs, name) for name in BACKENDS}
    gradients = {name: torch.autograd.grad(o, inputs, upstream, retain_graph=True)
                 for name, o in outputs.items()}
    reference = gradients["flashqla"]
    checks = {}
    for name in BACKENDS:
        grad_errors = {key: rms(a, b) for key, a, b in
                       zip(("dq", "dk", "dv", "dg", "db"), gradients[name], reference)}
        assert all(torch.isfinite(a).all() for a in gradients[name]), name
        assert max(grad_errors.values()) < 0.02, (name, grad_errors)
        checks[name] = dict(output_rms=rms(outputs[name], outputs["flashqla"]),
                            fixed_upstream_gradient_rms=grad_errors,
                            training_cp_partitions=outputs[name].grad_fn.cp_context.intra_cp_cu_seqlens.numel() - 1
                            if outputs[name].grad_fn.cp_context.is_intra else 1)
        cache = outputs[name].grad_fn.cp_cache
        checks[name]["cached_partition_bytes"] = (sum(
            t.numel() * t.element_size() for t in vars(cache).values() if isinstance(t, torch.Tensor)
        ) if cache is not None else 0)
    # Include each forward's own upstream gradient from an actual scalar loss.
    for name, o in outputs.items():
        grads = torch.autograd.grad(o.float().square().mean(), inputs, retain_graph=True)
        if name == "flashqla":
            loss_reference = grads
        checks[name]["mse_loss_gradient_rms"] = {
            key: rms(a, b) for key, a, b in zip(("dq", "dk", "dv", "dg", "db"), grads, loss_reference)
        }
    del gradients, reference, loss_reference, grads

    def backward(name):
        return torch.autograd.grad(outputs[name], inputs, upstream, retain_graph=True)

    def step(name):
        return torch.autograd.grad(call(inputs, name), inputs, upstream)

    repeats = 10 if length <= 16384 else 2
    measurements = dict(checks=checks)
    for scope, operation in (("forward", lambda name: call(inputs, name)),
                             ("backward", backward), ("forward_backward", step)):
        measurements[scope] = interleaved(
            {name: lambda name=name, op=operation: op(name) for name in BACKENDS}, repeats)
    if trace_dir is not None:
        for name in BACKENDS:
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                    torch.profiler.ProfilerActivity.CUDA]) as prof:
                step(name)
                torch.cuda.synchronize()
            prof.export_chrome_trace(str(trace_dir / f"{length}_{name}_trace.json"))
    return measurements


class RegressionLayer(torch.nn.Module):
    """Trainable Q/K/V/g/beta projections and output projection, without convolution."""
    def __init__(self):
        super().__init__()
        self.projection = torch.nn.Linear(64, 2 * 2 * 128 + 4 * 128 + 2 * 4, device="cuda")
        self.output = torch.nn.Linear(4 * 128, 64, device="cuda")

    def forward(self, x, backend, decay):
        projected = self.projection(x)
        q, k, v, g, beta = projected.split([256, 256, 512, 4, 4], dim=-1)
        inputs = (q.reshape(1, -1, 2, 128).bfloat16(),
                  k.reshape(1, -1, 2, 128).bfloat16(),
                  v.reshape(1, -1, 4, 128).bfloat16(),
                  -F.softplus(g) * decay, beta.sigmoid())
        result = call(inputs, backend, partitions=2)
        return self.output(result.flatten(2).float())


def trajectory(steps, decay):
    torch.manual_seed(194)
    x = torch.randn(1, 256, 64, device="cuda")
    teacher = RegressionLayer()
    with torch.no_grad():
        target = teacher(x, "flashqla", decay).detach()
    initial = RegressionLayer().state_dict()
    results, final, first_gradients = {}, {}, {}
    for name in BACKENDS:
        layer = RegressionLayer()
        layer.load_state_dict(initial)
        optimizer = torch.optim.AdamW(layer.parameters(), lr=0.001)
        losses = []
        for iteration in range(steps):
            optimizer.zero_grad(set_to_none=True)
            loss = F.mse_loss(layer(x, name, decay), target)
            loss.backward()
            assert torch.isfinite(loss), (name, iteration)
            assert all(torch.isfinite(p.grad).all() for p in layer.parameters()), (name, iteration)
            if iteration == 0:
                first_gradients[name] = torch.cat([p.grad.flatten() for p in layer.parameters()]).detach()
            optimizer.step()
            losses.append(loss.item())
        with torch.no_grad():
            final_loss = F.mse_loss(layer(x, name, decay), target).item()
        assert final_loss < losses[0], (name, losses[0], final_loss)
        final[name] = torch.cat([p.detach().flatten() for p in layer.parameters()])
        results[name] = dict(losses=losses, final_loss=final_loss,
                             loss_reduction=losses[0] / max(final_loss, 1e-30))
    for name in BACKENDS:
        results[name]["first_parameter_gradient_rms_vs_flashqla"] = rms(first_gradients[name], first_gradients["flashqla"])
        results[name]["final_parameter_rms_vs_flashqla"] = rms(final[name], final["flashqla"])
    return dict(scope="Synthetic fixed-input regression, T=256, Hk=2, Hv=4, D=128, hidden=64. "
                      "FP32 trainable projections; BF16 GDN activations; AdamW lr=0.001. "
                      "No convolution, norm/gating output block, language model, or held-out data.",
                steps=steps, gate_softplus_multiplier=decay, backends=results)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seqlens", nargs="*", type=int, default=[16384, 131072])
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--decay", type=float, default=1 / 16,
                        help="Multiplier for logsigmoid gate samples; use 0.001 for weaker forgetting")
    parser.add_argument("--trace", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("traces/gdn_training/results.json"))
    args = parser.parse_args()
    assert args.steps >= 2
    assert args.decay >= 0
    torch.set_num_threads(4)
    torch.cuda.set_stream(torch.cuda.Stream())
    root = Path(__file__).resolve().parents[1]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    results = dict(gpu=torch.cuda.get_device_name(), torch=torch.__version__,
                   visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
                   gate_logsigmoid_multiplier=args.decay,
                   scope="GDN public API, B=1,Hk=16,Hv=48,D=128,BF16; normalization; "
                         "full-history optimized forward; default training CP policy for backward. "
                         "Timings exclude loss, optimizer, and other model layers.",
                   source_sha256={str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                                  for p in [Path(__file__), root / "flash_qla/ops/gated_delta_rule/chunk/__init__.py",
                                            root / "flash_qla/ops/gated_delta_rule/chunk/hopper/fused_bwd.py"]})
    for length in args.seqlens:
        print("BENCH", length, flush=True)
        results[str(length)] = benchmark(length, args.output.parent if args.trace else None, args.decay)
        args.output.write_text(json.dumps(results, indent=2) + "\n")
        print({name: data["median_ms"] for name, data in results[str(length)]["forward_backward"].items()}, flush=True)
        torch.cuda.empty_cache()
    results["trajectory"] = trajectory(args.steps, args.decay)
    args.output.write_text(json.dumps(results, indent=2) + "\n")
    print({name: data["final_loss"] for name, data in results["trajectory"]["backends"].items()}, flush=True)


if __name__ == "__main__":
    main()

"""Backward comparison with identical inputs, preparation, partitions, and cache policy.

Reference driver/math/main kernel are loaded from a specified git revision.
Forward preparation is shared, outside timing; no inverse recomputation advantage.
"""
import argparse
import ast
from contextlib import contextmanager
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bench_implicit_bwd import interleaved, rms
from flash_qla.ops.gated_delta_rule.chunk.hopper.fast_fwd import _training_cp_context
from flash_qla.ops.gated_delta_rule.chunk.hopper.implicit_bwd import implicit_gdr_bwd
from flash_qla.utils import l2norm_fwd

chunk = importlib.import_module("flash_qla.ops.gated_delta_rule.chunk")


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def reference(revision, folder):
    sources = {}
    for name, relative in {
        "driver": "flash_qla/ops/gated_delta_rule/chunk/__init__.py",
        "math": "flash_qla/utils/math.py",
        "main": "flash_qla/ops/gated_delta_rule/chunk/hopper/fused_bwd.py",
    }.items():
        source = subprocess.check_output(["git", "show", f"{revision}:{relative}"], cwd=ROOT, text=True)
        path = folder / f"reference_{name}.py"
        path.write_text(source)
        sources[name] = path
    math = load_module(sources["math"], "flash_qla.utils._fair_reference_math")
    main = load_module(sources["main"], "flash_qla.ops.gated_delta_rule.chunk.hopper._fair_reference_main")
    node = next(n for n in ast.parse(sources["driver"].read_text()).body
                if isinstance(n, ast.FunctionDef) and n.name == "chunk_gated_delta_rule_bwd")
    namespace = dict(vars(chunk))
    namespace["fused_gdr_bwd"] = main.fused_gdr_bwd
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(sources["driver"]), "exec"), namespace)
    driver = namespace["chunk_gated_delta_rule_bwd"]

    def backward(**kwargs):
        kwargs = dict(kwargs)
        qr, kr = kwargs.pop("q_rstd"), kwargs.pop("k_rstd")
        dq, dk, dv, db, dg, dh = driver(**kwargs)
        dq = math.l2norm_bwd(kwargs["q"], qr, dq).to(kwargs["q"].dtype)
        dk = math.l2norm_bwd(kwargs["k"], kr, dk).to(kwargs["k"].dtype)
        return dq, dk, dv, db, dg, dh

    return backward, main.fused_gdr_bwd, {n: hashlib.sha256(p.read_bytes()).hexdigest() for n, p in sources.items()}


@contextmanager
def truncation(enabled):
    original = chunk.get_warmup_chunks_bidi

    def full_history(*args, **kwargs):
        return original(*args, **dict(kwargs, threshold=float("-inf")))

    if not enabled:
        chunk.get_warmup_chunks_bidi = full_history
    try:
        yield
    finally:
        chunk.get_warmup_chunks_bidi = original


@contextmanager
def main_kernel(kernel):
    original = chunk.fused_gdr_bwd
    chunk.fused_gdr_bwd = kernel
    try:
        yield
    finally:
        chunk.fused_gdr_bwd = original


def unique_tensor_bytes(values):
    storages = {}
    for t in values:
        if isinstance(t, torch.Tensor):
            storage = t.untyped_storage()
            storages[storage.data_ptr()] = storage.nbytes()
    return sum(storages.values())


def check(actual, expected, tolerance, names=("dq", "dk", "dv", "db", "dg", "dh0")):
    result = {}
    for name, a, b in zip(names, actual, expected):
        if b is None:
            assert a is None, name
            continue
        assert a.dtype == b.dtype and torch.isfinite(a).all(), name
        error = rms(a, b)
        assert error < tolerance, (name, error)
        result[name] = dict(relative_rms=error, bitwise_equal=torch.equal(a, b))
    return result


def peak_memory(fn):
    # Called before any graph capture. Fixture remains allocated for every variant.
    out = fn()
    torch.cuda.synchronize()
    del out
    torch.cuda.empty_cache()
    before = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    out = fn()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()
    del out
    return dict(allocated_before_bytes=before, peak_allocated_bytes=peak,
                peak_increment_bytes=peak - before)


@torch.no_grad()
def benchmark(length, decay, parts, policies, baseline, baseline_main, include_implicit, trace_dir, candidate_tolerance):
    assert length % (parts * 64) == 0
    torch.manual_seed(42)
    raw_q, raw_k = [F.normalize(torch.randn(1, length, 16, 128, device="cuda"), dim=-1).bfloat16()
                    for _ in range(2)]
    q, qr = l2norm_fwd(raw_q)
    k, kr = l2norm_fwd(raw_k)
    v = torch.randn(1, length, 48, 128, device="cuda", dtype=torch.bfloat16)
    raw_g = F.logsigmoid(torch.randn(1, length, 48, device="cuda")) * decay
    beta = torch.randn_like(raw_g).sigmoid()
    upstream = torch.randn_like(v)
    context = _training_cp_context(k.device, length, parts)
    # One original-precision inverse and one cache; all variants share these tensors.
    gc, inverse, output, _, final, cache = chunk.chunk_gated_delta_rule_fwd(
        q, k, v, raw_g, beta, scale=128**-0.5, output_final_state=False,
        cp_context=context, enable_fwd_cp_cache=True, is_train=True,
    )
    del output, final, raw_q, raw_k, raw_g
    shared = dict(q=q, k=k, v=v, g=gc, beta=beta, A=inverse, do=upstream,
                  scale=128**-0.5, cp_context=context, q_rstd=qr, k_rstd=kr)
    results = dict(partitions=parts, boundaries=context.intra_cp_cu_seqlens.tolist(),
                   common_inverse_dtype=str(inverse.dtype),
                   common_inverse_bytes=inverse.numel() * inverse.element_size(),
                   common_partition_cache_bytes=unique_tensor_bytes(vars(cache).values()),
                   shared_fixture_bytes=unique_tensor_bytes([*shared.values(), *vars(cache).values()]))
    current_main = chunk.fused_gdr_bwd
    for policy in policies:
        kwargs = dict(shared, cp_cache=cache if policy == "cached" else None)
        functions = {
            "flashqla_committed": lambda kwargs=kwargs: baseline(**kwargs),
            "optimized_postprocessing": lambda kwargs=kwargs: chunk.chunk_gated_delta_rule_bwd(**kwargs),
        }
        if include_implicit:
            def implicit(kwargs=kwargs):
                with main_kernel(implicit_gdr_bwd):
                    return chunk.chunk_gated_delta_rule_bwd(**kwargs)
            functions["implicit_experimental"] = implicit
        expected = functions["flashqla_committed"]()
        checks = {}
        for name, fn in functions.items():
            actual = fn()
            checks[name] = check(actual, expected, 0.04 if name == "implicit_experimental" else candidate_tolerance)
            del actual
        del expected
        entry = results[policy] = dict(gradient_checks=checks,
                                       memory={name: peak_memory(fn) for name, fn in functions.items()})
        fixture = {}

        def collect(**args):
            fixture.update(args)
            return current_main(**args)

        with main_kernel(collect):
            out = chunk.chunk_gated_delta_rule_bwd(**kwargs)
        del out
        # Exact same tensor objects, including reconstructed h and corrected dht.
        kernels = dict(flashqla_committed=baseline_main, current_main=current_main)
        if include_implicit:
            kernels["implicit_experimental"] = implicit_gdr_bwd
        expected = baseline_main(**fixture)
        entry["main_gradient_checks"] = {}
        for name, kernel in kernels.items():
            actual = kernel(**fixture)
            entry["main_gradient_checks"][name] = check(
                actual, expected, 0.04 if name == "implicit_experimental" else candidate_tolerance,
                names=("dq", "dk", "dv", "dg", "db", "dh0"))
            del actual
        del expected
        entry["main_self_repeat_checks"] = {}
        for name, kernel in kernels.items():
            first = kernel(**fixture)
            errors = []
            for _ in range(3):
                repeated = kernel(**fixture)
                errors.append(check(repeated, first, 0.04, names=("dq", "dk", "dv", "dg", "db", "dh0")))
                del repeated
            entry["main_self_repeat_checks"][name] = errors
            del first
        repeats = 10 if length <= 16384 else 2
        entry["main_kernel"] = interleaved({name: lambda kernel=kernel: kernel(**fixture)
                                             for name, kernel in kernels.items()}, repeats)
        del fixture
        torch.cuda.empty_cache()
        entry["complete_backward"] = interleaved(functions, repeats)
        if trace_dir is not None:
            for name, fn in functions.items():
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                        torch.profiler.ProfilerActivity.CUDA]) as prof:
                    fn()
                    torch.cuda.synchronize()
                path = trace_dir / f"{length}_decay{decay}_{policy}_{name}.json"
                prof.export_chrome_trace(str(path))
        print(length, decay, policy, {n: round(e["median_ms"], 4) for n, e in entry["complete_backward"].items()}, flush=True)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seqlens", type=int, nargs="+", default=[16384, 131072])
    parser.add_argument("--decays", type=float, nargs="+", default=[0.001, 0.0625])
    parser.add_argument("--partitions", type=int, default=8)
    parser.add_argument("--cache-policies", nargs="+", choices=["cached", "recomputed"], default=["cached", "recomputed"])
    parser.add_argument("--reference-rev", default="HEAD")
    parser.add_argument("--main-source", type=Path, help="Experimental current main kernel source; reference unchanged")
    parser.add_argument("--candidate-tolerance", type=float, default=0.003,
                        help="Explicit RMS limit for candidate checks; factored-algebra experiments need a separate precision assessment")
    parser.add_argument("--truncation", choices=["off", "default"], default="off")
    parser.add_argument("--implicit", action="store_true")
    parser.add_argument("--trace", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("traces/gdn_fair_backward/results.json"))
    args = parser.parse_args()
    assert 0 < args.candidate_tolerance <= 0.04
    torch.set_num_threads(4)
    torch.cuda.set_stream(torch.cuda.Stream())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    revision = subprocess.check_output(["git", "rev-parse", args.reference_rev], cwd=ROOT, text=True).strip()
    baseline, baseline_main, hashes = reference(revision, args.output.parent)
    if args.main_source:
        module = load_module(args.main_source, "flash_qla.ops.gated_delta_rule.chunk.hopper._fair_candidate_main")
        chunk.fused_gdr_bwd = module.fused_gdr_bwd
    results = dict(gpu=torch.cuda.get_device_name(), torch=torch.__version__,
                   visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"), reference_revision=revision,
                   reference_source_sha256=hashes, truncation=args.truncation,
                   candidate_gradient_rms_limit=args.candidate_tolerance,
                   scope="Backward only, B=1 Hk=16 Hv=48 D=128 BF16; same tensor objects for inverse, cumulative gates, CP cache, upstream gradients; same equal partitions. No optimized-forward intermediates.",
                   method="Nine alternating CUDA graph samples; 50 alternating warmups; peak memory measured before graph capture",
                   current_source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                         for p in [Path(__file__), ROOT / "flash_qla/utils/math.py",
                                                   ROOT / "flash_qla/ops/gated_delta_rule/chunk/__init__.py",
                                                   ROOT / "flash_qla/ops/gated_delta_rule/chunk/hopper/fused_bwd.py"]})
    if args.main_source:
        results["experimental_main"] = dict(path=str(args.main_source), sha256=hashlib.sha256(args.main_source.read_bytes()).hexdigest())
    with truncation(args.truncation == "default"):
        for decay in args.decays:
            for length in args.seqlens:
                print("BENCH", length, decay, flush=True)
                results[f"{length}_decay{decay}"] = benchmark(
                    length, decay, args.partitions, args.cache_policies, baseline, baseline_main,
                    args.implicit, args.output.parent if args.trace else None, args.candidate_tolerance)
                args.output.write_text(json.dumps(results, indent=2) + "\n")
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

"""FP8-forward/BF16-backward projections, including online quant/dequant costs."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import sys

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bench_gdn_full_layer_training import Layer
from bench_fair_gdn_backward import truncation
from bench_implicit_bwd import interleaved, rms
from flash_qla.ops.fp8_linear import fp8_linear, quantize_linear_weight
from flash_qla.ops.gated_delta_rule.chunk.hopper.fast_fwd import _training_cp_context


class PreparedLinear(torch.nn.Module):
    def __init__(self, linear, quantizer=None, compare_rowwise=False, direct_scales=True):
        super().__init__()
        self.weight = linear.weight
        if quantizer is None:
            self.prepared = quantize_linear_weight(self.weight)
            self.operation = fp8_linear
        else:
            from flash_qla.ops.deepgemm_linear import deepgemm_linear, quantize_deepgemm_weight
            self.prepared = quantize_deepgemm_weight(self.weight, quantizer, direct_scales)
            self.operation = deepgemm_linear
        self.enabled = False
        self.direct_scales = direct_scales
        self.rowwise = quantize_linear_weight(self.weight) if compare_rowwise else None

    def forward(self, x):
        if self.enabled == "rowwise":
            return fp8_linear(x, self.weight, self.rowwise)
        if self.operation is not fp8_linear:
            self.prepared.direct_scales = self.direct_scales if self.enabled != "legacy_scales" else False
        return (self.operation(x, self.weight, self.prepared) if self.enabled
                else F.linear(x, self.weight))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seqlens", type=int, nargs="+", default=[16384, 131072])
    parser.add_argument("--scopes", nargs="+", choices=["projections", "full_layer"],
                        default=["projections", "full_layer"])
    parser.add_argument("--trace", action="store_true")
    parser.add_argument("--fp8-backend", choices=["rowwise", "deepgemm"], default="rowwise")
    parser.add_argument("--compare-rowwise", action="store_true",
                        help="Add rowwise FP8 as a third interleaved full-layer variant")
    parser.add_argument("--legacy-scales", action="store_true", help="Use original YDT activation scale layout")
    parser.add_argument("--compare-scale-layout", action="store_true",
                        help="Add original-layout DeepGEMM as an interleaved full-layer variant")
    parser.add_argument("--ydt-quantization-source", type=Path,
                        default=Path("/home/kareemm/arcadia/ml/torch/ydt/core/ydt_core/fp8/quantization/blockwise.py"))
    parser.add_argument("--fp8-projections", nargs="+", choices=["q", "k", "v", "z", "a", "b", "out"],
                        default=["q", "k", "v", "z", "a", "b", "out"])
    parser.add_argument("--output", type=Path, default=Path("traces/gdn_fp8_training/results.json"))
    args = parser.parse_args()
    if args.compare_rowwise and args.fp8_backend != "deepgemm":
        parser.error("--compare-rowwise requires --fp8-backend deepgemm")
    if args.compare_scale_layout and (args.fp8_backend != "deepgemm" or args.legacy_scales):
        parser.error("--compare-scale-layout requires DeepGEMM with direct scales")
    torch.set_num_threads(4)
    torch.cuda.set_stream(torch.cuda.Stream())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sources = [Path(__file__), ROOT / "flash_qla/ops/fp8_linear.py",
               ROOT / "profile/bench_gdn_full_layer_training.py"]
    quantizer = None
    if args.fp8_backend == "deepgemm":
        import deep_gemm
        from flash_qla.ops.deepgemm_linear import load_ydt_blockwise
        quantizer = load_ydt_blockwise(args.ydt_quantization_source)
        sources.append(ROOT / "flash_qla/ops/deepgemm_linear.py")
    results = dict(gpu=torch.cuda.get_device_name(), torch=torch.__version__,
        visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
        fp8_projections=args.fp8_projections,
        fp8_backend=args.fp8_backend,
        direct_scales=not args.legacy_scales,
        compare_rowwise=args.compare_rowwise,
        compare_scale_layout=args.compare_scale_layout,
        shape=dict(batch=1, hidden=2560, key_heads=16, value_heads=48, head_dim=128, partitions=8),
        precision="Per-row E4M3 activation/per-output-channel E4M3 weight, FP32 scales, BF16 output; BF16 input and parameter gradients using BF16 backward GEMMs. Straight-through dW uses original BF16 X; dX uses runtime-dequantized FP8 W.",
        timing="Weight quantization is excluded (explicit preparation between optimizer steps). Activation quantization, weight dequantization, BF16 dX and dW GEMMs included. No cached dequantized weight. CUDA graphs: 50 alternating warmups, nine alternating-order samples.",
        full_layer="Identical standard GDN forward/backward, saved intermediates, P8 and decay truncation off; fused training norm/gate in both variants. Includes all parameter/input gradients; excludes loss/optimizer.",
        projections="Each of seven projections measured with BF16 fixed inputs/upstream gradients and both dX/dW; input-gradient summation for shared input is excluded in this isolated scope.",
        source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
    if quantizer is not None:
        results["precision"] = "YDT blockwise E4M3: activation 1x128, weight 128x128; power-of-two FP32 scales; DeepGEMM fp8_gemm_nt with BF16 output. BF16 dX/dW GEMMs; dX uses runtime-dequantized FP8 W and straight-through dW uses original BF16 X. Activation scales are written directly as TMA-aligned MN-major FP32 unless --legacy-scales is selected."
        results["deepgemm_version"] = deep_gemm.__version__
        results["deepgemm_num_sms"] = deep_gemm.get_num_sms()
        results["quantization_source"] = str(args.ydt_quantization_source.resolve())
        results["quantization_sha256"] = hashlib.sha256(args.ydt_quantization_source.read_bytes()).hexdigest()
        results["timing"] += " DeepGEMM scale-layout conversion is timed; no scale layout or dequantized weight is cached."
    for length in args.seqlens:
        print("SEQLEN", length, flush=True)
        torch.manual_seed(442)
        x = torch.randn(1, length, 2560, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        layer = Layer(2560, .001, _training_cp_context(x.device, length, 8))
        layer.fused_norm_gate = True
        linears = {}
        # Weight quantization occurs here, before every measured scope.
        for name in ("q", "k", "v", "z", "a", "b", "out"):
            module = PreparedLinear(getattr(layer, name), quantizer, args.compare_rowwise, not args.legacy_scales)
            setattr(layer, name, module)
            linears[name] = module
        entry = results[str(length)] = {}
        if "projections" in args.scopes:
            isolated = entry["projections"] = {}
            for name, module in linears.items():
                k = module.weight.shape[1]
                activation = x if k == 2560 else torch.randn(1, length, k, device="cuda", dtype=torch.bfloat16, requires_grad=True)
                do = torch.randn(1, length, module.weight.shape[0], device="cuda", dtype=torch.bfloat16)
                def forward(enabled):
                    module.enabled = enabled if name in args.fp8_projections else False
                    return module(activation)
                def step(enabled):
                    return torch.autograd.grad(forward(enabled), (activation, module.weight), do)
                with torch.no_grad():
                    first, second = forward(False), forward(True)
                    output_error = rms(second, first)
                    del first, second
                expected, actual = step(False), step(True)
                checks = {n:rms(a,b) for n,a,b in zip(("dx","dw"),actual,expected)}
                assert all(t.dtype == torch.bfloat16 and torch.isfinite(t).all() for t in actual)
                del expected, actual
                forward_times = interleaved(dict(bf16=lambda:forward(False), fp8=lambda:forward(True)), 5 if length==16384 else 2)
                times = interleaved(dict(bf16=lambda:step(False), fp8=lambda:step(True)), 5 if length==16384 else 2)
                isolated[name] = dict(output_relative_rms=output_error,gradient_relative_rms=checks,
                                      forward=forward_times,forward_backward=times)
                print(name,{n:t["median_ms"] for n,t in times.items()},flush=True)
                del activation, do, forward, step
                torch.cuda.empty_cache()
            args.output.write_text(json.dumps(results,indent=2)+"\n")
        if "full_layer" in args.scopes:
            inputs = (x, *layer.parameters())
            upstream = torch.randn_like(x)
            def forward(enabled):
                for name, module in linears.items():
                    module.enabled = enabled if name in args.fp8_projections else False
                with truncation(False):
                    return layer(x)
            def step(enabled):
                return torch.autograd.grad(forward(enabled), inputs, upstream)
            with torch.no_grad():
                expected, actual = forward(False),forward(True)
                output_error = rms(actual,expected)
                assert torch.isfinite(actual).all()
                del expected, actual
            expected,actual = step(False),step(True)
            checks = {n:rms(a,b) for n,a,b in zip(["input",*[n for n,_ in layer.named_parameters()]],actual,expected)}
            assert all(torch.isfinite(t).all() for t in actual)
            assert all(a.dtype==b.dtype for a,b in zip(actual,expected))
            del actual,expected
            # Quantization changes the numerical algorithm: record errors, do
            # not use the former <1% parity limit as a convergence claim.
            functions = dict(bf16=lambda:step(False),fp8=lambda:step(True))
            if args.compare_rowwise:
                functions["rowwise"] = lambda:step("rowwise")
            if args.compare_scale_layout:
                functions["legacy_scales"] = lambda:step("legacy_scales")
            timing = interleaved(functions,5 if length==16384 else 2)
            entry["full_layer"] = dict(output_relative_rms=output_error,gradient_relative_rms=checks,forward_backward=timing)
            print("full_layer",{n:t["median_ms"] for n,t in timing.items()},"output_rms",output_error,"max_gradient_rms",max(checks.values()),flush=True)
            if args.trace:
                gc.collect()
                torch.cuda.empty_cache()
                trace_variants = [("bf16",False),("fp8",True)]
                if args.compare_rowwise:
                    trace_variants.append(("rowwise","rowwise"))
                if args.compare_scale_layout:
                    trace_variants.append(("legacy_scales","legacy_scales"))
                for name,enabled in trace_variants:
                    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
                        step(enabled)
                        torch.cuda.synchronize()
                    prof.export_chrome_trace(str(args.output.parent/f"{length}_{name}.json"))
            del inputs,upstream,forward,step,functions
        args.output.write_text(json.dumps(results,indent=2)+"\n")
        del x,layer,linears,module
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

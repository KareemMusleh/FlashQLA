"""Experimental DeepGEMM forward with YDT block quantization and BF16 backward.

The quantization implementation is loaded explicitly from the caller's YDT
checkout, avoiding unrelated YDT package initialization. No files there change.
"""
from dataclasses import dataclass
import importlib.util
from pathlib import Path
import sys

import torch
import triton
import triton.language as tl


@triton.jit
def _quantize_activation_tma(X, Q, S, M: tl.constexpr, K: tl.constexpr,
                             SCALE_LD: tl.constexpr, BM: tl.constexpr):
    row = tl.program_id(0).to(tl.int64) * BM + tl.arange(0, BM)
    kb = tl.program_id(1)
    col = kb * 128 + tl.arange(0, 128)
    off = row[:, None] * K + col[None, :]
    x = tl.load(X + off, row[:, None] < M, 0.0).to(tl.float32)
    amax = tl.max(tl.abs(x), axis=1)
    # YDT's exact power-of-two multiplier, including zero/tiny-value handling.
    mult = tl.where(amax == 0, 1.0, 448.0 / amax)
    mult = tl.where(mult.to(tl.int32, bitcast=True) == 0x7F800000,
                    3.4028234663852886e38, mult)
    mult = (mult.to(tl.uint32, bitcast=True) & 0xFF800000).to(tl.float32, bitcast=True)
    tl.store(Q + off, (x * mult[:, None]).to(tl.float8e4nv), row[:, None] < M)
    tl.store(S + row + kb * SCALE_LD, 1.0 / mult, row < SCALE_LD)


def quantize_activation_tma(x):
    """Write FP32 activation scales directly as TMA-aligned MN-major data."""
    m, k = x.shape
    if x.dtype != torch.bfloat16 or not x.is_cuda or not x.is_contiguous() or k % 128:
        raise ValueError("expected a contiguous BF16 CUDA matrix with K divisible by 128")
    ld = triton.cdiv(m, 4) * 4  # 16-byte TMA alignment for FP32 scales.
    data = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    storage = torch.empty((k // 128, ld), device=x.device, dtype=torch.float32)
    scale = storage.T[:m]
    _quantize_activation_tma[(triton.cdiv(m, 128), k // 128)](
        x, data, scale, m, k, ld, 128, num_warps=4)
    return data, scale


def load_ydt_blockwise(source):
    source = Path(source).resolve()
    name = "_flashqla_ydt_blockwise_quantization"
    if name in sys.modules:
        module = sys.modules[name]
        if Path(module.__file__).resolve() != source:
            raise ValueError("a different YDT quantization source is already loaded")
        return module
    spec = importlib.util.spec_from_file_location(name, source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        del sys.modules[name]
        raise
    return module


@dataclass
class DeepGEMMWeight:
    data: torch.Tensor
    scale: torch.Tensor
    parameter: torch.Tensor
    version: int
    quantizer: object
    direct_scales: bool = True


@torch.no_grad()
def quantize_deepgemm_weight(weight, quantizer, direct_scales=True):
    if weight.ndim != 2 or weight.dtype != torch.bfloat16 or not weight.is_cuda:
        raise ValueError("weight must be a BF16 CUDA matrix")
    if not weight.is_contiguous() or weight.shape[1] % 128 or weight.shape[0] % 16:
        raise ValueError("weight requires contiguous storage, K divisible by 128, N divisible by 16")
    data, scale = quantizer.quant(weight, (128, 128))
    return DeepGEMMWeight(data, scale, weight, weight._version, quantizer, direct_scales)


class _DeepGEMMLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, prepared):
        import deep_gemm

        matrix = x.reshape(-1, weight.shape[1])
        with torch.profiler.record_function("deepgemm_linear/activation_quantization"):
            if prepared.direct_scales:
                data, scale = quantize_activation_tma(matrix)
            else:
                data, scale = prepared.quantizer.quant(matrix, (1, 128))
        out = torch.empty((matrix.shape[0], weight.shape[0]), device=x.device, dtype=torch.bfloat16)
        with torch.profiler.record_function("deepgemm_linear/fp8_gemm_nt"):
            # SM90 accepts the direct MN-major activation scales unchanged;
            # 128x128 weight-block scales already use its required layout.
            deep_gemm.fp8_gemm_nt((data, scale), (prepared.data, prepared.scale), out)
        ctx.save_for_backward(x, weight)
        ctx.prepared = prepared
        return out.reshape(*x.shape[:-1], weight.shape[0])

    @staticmethod
    def backward(ctx, dy):
        x, weight = ctx.saved_tensors
        p = ctx.prepared
        dy = dy.reshape(-1, weight.shape[0]).contiguous()
        dx = dw = None
        if ctx.needs_input_grad[0]:
            with torch.profiler.record_function("deepgemm_linear/weight_dequantization"):
                w = p.quantizer.dequant(p.data, p.scale, (128, 128), dtype=torch.bfloat16)
            with torch.profiler.record_function("deepgemm_linear/bf16_input_gradient"):
                dx = (dy @ w).reshape_as(x)
        if ctx.needs_input_grad[1]:
            with torch.profiler.record_function("deepgemm_linear/bf16_weight_gradient"):
                dw = dy.T @ x.reshape(-1, weight.shape[1])
        return dx, dw, None


def deepgemm_linear(x, weight, prepared):
    """DeepGEMM FP8 forward, runtime weight dequantization and BF16 dX/dW."""
    if prepared.parameter is not weight or prepared.version != weight._version:
        raise ValueError("stale FP8 weight: re-quantize after the optimizer update")
    if x.dtype != torch.bfloat16 or x.device != weight.device or not x.is_contiguous():
        raise ValueError("input must be contiguous BF16 on the weight's CUDA device")
    if x.shape[-1] != weight.shape[1]:
        raise ValueError("input and weight feature dimensions differ")
    return _DeepGEMMLinear.apply(x, weight, prepared)


class _SharedDeepGEMMProjections(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, prepared, combined_backward, *weights):
        import deep_gemm

        matrix = x.reshape(-1, x.shape[-1])
        data, scale = quantize_activation_tma(matrix)
        outputs = []
        for w, p in zip(weights, prepared):
            out = torch.empty((matrix.shape[0], w.shape[0]), device=x.device, dtype=x.dtype)
            deep_gemm.fp8_gemm_nt((data, scale), (p.data, p.scale), out)
            outputs.append(out.reshape(*x.shape[:-1], w.shape[0]))
        # Quantized X is not retained: BF16 backward needs the original X.
        ctx.save_for_backward(x, *weights)
        ctx.prepared = prepared
        ctx.combined_backward = combined_backward
        ctx.set_materialize_grads(False)
        return tuple(outputs)

    @staticmethod
    def backward(ctx, *grad_outputs):
        x, *weights = ctx.saved_tensors
        rows, k = x.numel() // x.shape[-1], x.shape[-1]
        gradients = [g.reshape(rows, w.shape[0]).contiguous() if g is not None
                     else x.new_zeros(rows, w.shape[0]) for g, w in zip(grad_outputs, weights)]
        if ctx.combined_backward:
            packed = torch.cat(gradients, dim=-1)
            dequantized = [p.quantizer.dequant(p.data, p.scale, (128,128), dtype=x.dtype)
                           for p in ctx.prepared]
            dx = (packed @ torch.cat(dequantized, dim=0)).reshape_as(x)
            dw = (packed.T @ x.reshape(rows,k)).split([w.shape[0] for w in weights])
        else:
            dx, dw = None, []
            # Reverse branch order matches the usual separate-linear graph.
            for g, w, p in reversed(list(zip(gradients, weights, ctx.prepared))):
                decoded = p.quantizer.dequant(p.data,p.scale,(128,128),dtype=x.dtype)
                term = g @ decoded
                dx = term if dx is None else dx + term
                dw.append(g.T @ x.reshape(rows,k))
            dx, dw = dx.reshape_as(x), dw[::-1]
        dw = [g if original is not None else None for g, original in zip(dw, grad_outputs)]
        return dx, None, None, *dw


def shared_deepgemm_projections(x, weights, prepared, combined_backward=False):
    """One online X quantization; optional combined BF16 dX/dW GEMMs.

    All weight dequantization and gradient/weight packing occurs in backward.
    No quantized activation or additional forward cache is retained.
    """
    if not weights or len(weights) != len(prepared):
        raise ValueError("expected matching nonempty weights and prepared weights")
    for w,p in zip(weights,prepared):
        if p.parameter is not w or p.version != w._version:
            raise ValueError("stale FP8 weight: re-quantize after the optimizer update")
        if w.shape[1] != x.shape[-1] or w.device != x.device:
            raise ValueError("input and weights must have matching features/devices")
    if x.dtype != torch.bfloat16 or not x.is_contiguous():
        raise ValueError("expected contiguous BF16 input")
    return _SharedDeepGEMMProjections.apply(x, tuple(prepared), combined_backward, *weights)

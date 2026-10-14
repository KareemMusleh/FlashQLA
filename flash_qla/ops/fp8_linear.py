"""FP8 forward linear with timed weight dequantization and BF16 backward.

Weights are explicitly quantized between optimizer steps, outside forward.
Backward uses a straight-through weight gradient: dW = dY.T @ original X.
"""
from dataclasses import dataclass

import torch
import triton
import triton.language as tl


@triton.jit
def _quantize_rows(X, Q, S, ROWS: tl.constexpr, K: tl.constexpr,
                   BK: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, BK)
    x = tl.load(X + row * K + col, (row < ROWS) & (col < K), 0).to(tl.float32)
    scale = tl.maximum(tl.max(tl.abs(x), axis=0), 1e-12) / 448.0
    tl.store(Q + row * K + col, x / scale, col < K)
    tl.store(S + row, scale)


@triton.jit
def _dequantize_weight(Q, S, W, N: tl.constexpr, K: tl.constexpr,
                       BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    q = tl.load(Q + i, i < N * K, 0.0).to(tl.float32)
    scale = tl.load(S + i // K, i < N * K, 0)
    tl.store(W + i, q * scale, i < N * K)


def _quantize(x):
    rows, k = x.shape
    padded = triton.cdiv(rows, 16) * 16
    q = torch.empty((padded, k), device=x.device, dtype=torch.float8_e4m3fn)
    scale = torch.empty((padded, 1), device=x.device, dtype=torch.float32)
    _quantize_rows[(padded,)](x, q, scale, rows, k, triton.next_power_of_2(k), num_warps=4)
    return q, scale


@dataclass
class FP8Weight:
    data: torch.Tensor
    scale: torch.Tensor
    parameter: torch.Tensor
    version: int


@torch.no_grad()
def quantize_linear_weight(weight):
    """Prepare a per-output-channel E4M3 weight after each optimizer update."""
    if weight.ndim != 2 or weight.dtype != torch.bfloat16 or not weight.is_cuda:
        raise ValueError("weight must be a BF16 CUDA matrix")
    if not weight.is_contiguous() or any(d % 16 for d in weight.shape):
        raise ValueError("weight must be contiguous with dimensions divisible by 16")
    q, scale = _quantize(weight)
    return FP8Weight(q, scale, weight, weight._version)


def dequantize_linear_weight(prepared):
    """Dequantize to BF16; deliberately called inside the timed backward."""
    n, k = prepared.data.shape
    result = torch.empty((n, k), device=prepared.data.device, dtype=torch.bfloat16)
    _dequantize_weight[(triton.cdiv(n * k, 1024),)](
        prepared.data, prepared.scale, result, n, k, 1024)
    return result


class _FP8Linear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, prepared):
        rows = x.numel() // x.shape[-1]
        with torch.profiler.record_function("fp8_linear/activation_quantization"):
            q, scale = _quantize(x.reshape(rows, x.shape[-1]))
        with torch.profiler.record_function("fp8_linear/scaled_mm"):
            y = torch._scaled_mm(q, prepared.data.T, scale_a=scale,
                                 scale_b=prepared.scale.T,
                                 out_dtype=torch.bfloat16, use_fast_accum=False)
        ctx.save_for_backward(x, weight)
        ctx.prepared = prepared
        return y[:rows].reshape(*x.shape[:-1], weight.shape[0])

    @staticmethod
    def backward(ctx, dy):
        x, weight = ctx.saved_tensors
        dy = dy.reshape(-1, weight.shape[0]).contiguous()
        dx = dw = None
        if ctx.needs_input_grad[0]:
            with torch.profiler.record_function("fp8_linear/weight_dequantization"):
                w = dequantize_linear_weight(ctx.prepared)
            with torch.profiler.record_function("fp8_linear/bf16_input_gradient"):
                dx = (dy @ w).reshape_as(x)
        if ctx.needs_input_grad[1]:
            with torch.profiler.record_function("fp8_linear/bf16_weight_gradient"):
                dw = dy.T @ x.reshape(-1, weight.shape[1])
        return dx, dw, None


def fp8_linear(x, weight, prepared):
    """BF16 output, BF16 dX/dW; FP8 forward, BF16 backward GEMMs.

    Call quantize_linear_weight again after an optimizer changes weight.
    Activation quantization and backward weight dequantization are runtime work.
    """
    if prepared.parameter is not weight or prepared.version != weight._version:
        raise ValueError("stale FP8 weight: re-quantize after the optimizer update")
    if x.dtype != torch.bfloat16 or x.device != weight.device or not x.is_contiguous():
        raise ValueError("input must be contiguous BF16 on the weight's CUDA device")
    if x.shape[-1] != weight.shape[1]:
        raise ValueError("input and weight feature dimensions differ")
    return _FP8Linear.apply(x, weight, prepared)

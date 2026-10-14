# Copyright (c) 2026 The Qwen team, Alibaba Group.
# Licensed under The MIT License [see LICENSE for details]

import torch
import triton
import triton.language as tl

from .contiguous import input_guard


@torch.compile
def l2norm_fwd_compiled(
    x: torch.Tensor,
    eps: float = 1e-6,
    dim: int = -1,
):
    x_f32 = x.to(torch.float32)
    sum_sq = (x_f32 * x_f32).sum(dim=dim, keepdim=True) + eps
    rstd = torch.rsqrt(sum_sq)
    y = (x * rstd).to(x.dtype)
    return y, rstd


def l2norm_fwd(
    x: torch.Tensor,
    eps: float = 1e-6,
    dim: int = -1,
):
    assert dim == -1 or dim == len(x.shape) - 1
    assert x.stride(-1) == 1
    raw_shape = x.shape
    x = x.view(-1, raw_shape[-1])
    torch._dynamo.mark_dynamic(x, 0)
    y, rstd = l2norm_fwd_compiled(x, eps, dim)
    y = y.view(raw_shape)
    rstd = rstd.view(*raw_shape[:-1])
    return y, rstd


@torch.compile
def l2norm_bwd_compiled(
    dy: torch.Tensor,
    y: torch.Tensor,
    rstd: torch.Tensor,
    eps: float = 1e-6,
    dim: int = -1,
    output_dtype: torch.dtype | None = None,
):
    y_f32 = y.to(torch.float32)
    dy_f32 = dy.to(torch.float32)
    dot = (dy_f32 * y_f32).sum(dim=-1, keepdim=True)
    dx = (dy_f32 - dot * y_f32) * rstd
    return dx.to(output_dtype) if output_dtype is not None else dx


def l2norm_bwd(
    y: torch.Tensor,
    rstd: torch.Tensor,
    dy: torch.Tensor,
    eps: float = 1e-6,
    dim: int = -1,
    output_dtype: torch.dtype | None = None,
):
    assert dim == -1 or dim == len(dy.shape) - 1
    assert y.stride(-1) == 1
    assert dy.stride(-1) == 1
    raw_shape = dy.shape
    y = y.view(-1, raw_shape[-1])
    dy = dy.view(-1, raw_shape[-1])
    rstd = rstd.view(-1, 1)
    torch._dynamo.mark_dynamic(y, 0)
    torch._dynamo.mark_dynamic(dy, 0)
    torch._dynamo.mark_dynamic(rstd, 0)
    dx = l2norm_bwd_compiled(dy, y, rstd, eps, dim, output_dtype)
    dx = dx.view(raw_shape)
    return dx


@triton.jit
def _group_reduce_l2norm_bwd_kernel(
    Y,
    Rstd,
    DY,
    DX,
    rows,
    D: tl.constexpr,
    GROUP: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
):
    row = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    col = tl.arange(0, BLOCK_D)
    mask = (row[:, None] < rows) & (col[None, :] < D)
    dy = tl.full((BLOCK_ROWS, BLOCK_D), 0, tl.float32)
    for group in range(GROUP):
        dy += tl.load(
            DY + (row[:, None] * GROUP + group) * D + col[None, :], mask, other=0
        ).to(tl.float32)
    # Match group_reduce_vector's low-precision store before normalization.
    dy = dy.to(DY.dtype.element_ty).to(tl.float32)
    y = tl.load(Y + row[:, None] * D + col[None, :], mask, other=0).to(tl.float32)
    rstd = tl.load(Rstd + row, row < rows, other=0)
    dot = tl.sum(dy * y, axis=1)
    dx = (dy - dot[:, None] * y) * rstd[:, None]
    tl.store(DX + row[:, None] * D + col[None, :], dx, mask)


def group_reduce_l2norm_bwd(y: torch.Tensor, rstd: torch.Tensor, dy: torch.Tensor):
    """Reduce value-head gradients, normalize, and cast in a single kernel.

    Inputs use contiguous [B, T, H, D] layout. Accumulation is FP32;
    the grouped gradient is rounded to its input dtype to preserve the
    existing separate reduction/normalization semantics.
    """
    assert y.is_contiguous() and dy.is_contiguous() and rstd.is_contiguous()
    assert y.shape[:2] == dy.shape[:2] and y.shape[-1] == dy.shape[-1]
    assert dy.shape[-2] % y.shape[-2] == 0
    assert rstd.numel() == y.numel() // y.shape[-1]
    dx = torch.empty_like(y)
    d = y.shape[-1]
    rows = y.numel() // d
    _group_reduce_l2norm_bwd_kernel[(triton.cdiv(rows, 4),)](
        y,
        rstd,
        dy,
        dx,
        rows,
        d,
        dy.shape[-2] // y.shape[-2],
        triton.next_power_of_2(d),
        4,
        num_warps=4,
        enable_fp_fusion=False,
    )
    return dx


class L2NormFunction(torch.autograd.Function):
    @staticmethod
    @input_guard
    def forward(
        ctx,
        x,
        eps=1e-6,
    ):
        y, rstd = l2norm_fwd(x, eps)
        ctx.eps = eps
        ctx.save_for_backward(y, rstd)
        return y

    @staticmethod
    @input_guard
    def backward(ctx, dy):
        y, rstd = ctx.saved_tensors
        dx = l2norm_bwd(y, rstd, dy, ctx.eps)
        return dx, None, None


def l2norm(
    x: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    return L2NormFunction.apply(x, eps)

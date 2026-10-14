"""RMSNorm and SiLU output gating with BF16 intermediates and autograd.

Retains the same normalized output, SiLU output, and FP32 norm statistics as
the separate PyTorch operations. Backward preserves BF16 multiplication
rounding before differentiating RMSNorm and SiLU.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _forward(X, Z, W, Y, N, S, R, rows, D: tl.constexpr, BD: tl.constexpr,
             BR: tl.constexpr, EPS: tl.constexpr):
    row = tl.program_id(0) * BR + tl.arange(0, BR)
    col = tl.arange(0, BD)
    mask = (row[:, None] < rows) & (col[None, :] < D)
    x = tl.load(X + row[:, None] * D + col[None, :], mask, 0).to(tl.float32)
    z = tl.load(Z + row[:, None] * D + col[None, :], mask, 0).to(tl.float32)
    w = tl.load(W + col, col < D, 0).to(tl.float32)
    r = tl.rsqrt(tl.sum(x * x, axis=1) / D + EPS)
    n = (x * r[:, None] * w[None, :]).to(N.dtype.element_ty)
    s = (z / (1 + tl.exp(-z))).to(S.dtype.element_ty)
    y = n.to(tl.float32) * s.to(tl.float32)
    tl.store(Y + row[:, None] * D + col[None, :], y, mask)
    tl.store(N + row[:, None] * D + col[None, :], n, mask)
    tl.store(S + row[:, None] * D + col[None, :], s, mask)
    tl.store(R + row, r, row < rows)


@triton.jit
def _backward(X, Z, W, N, S, R, DY, DX, DZ, DW, rows,
              D: tl.constexpr, BD: tl.constexpr, BR: tl.constexpr,
              HEADS: tl.constexpr = 0, DZ_PITCH: tl.constexpr = 0):
    block = tl.program_id(0)
    row = block * BR + tl.arange(0, BR)
    col = tl.arange(0, BD)
    mask = (row[:, None] < rows) & (col[None, :] < D)
    x = tl.load(X + row[:, None] * D + col[None, :], mask, 0).to(tl.float32)
    z = tl.load(Z + row[:, None] * D + col[None, :], mask, 0).to(tl.float32)
    n = tl.load(N + row[:, None] * D + col[None, :], mask, 0).to(tl.float32)
    s = tl.load(S + row[:, None] * D + col[None, :], mask, 0).to(tl.float32)
    dy = tl.load(DY + row[:, None] * D + col[None, :], mask, 0).to(tl.float32)
    r = tl.load(R + row, row < rows, 0)
    w = tl.load(W + col, col < D, 0).to(tl.float32)
    # Match the two BF16 outputs from PyTorch multiplication backward.
    dn = (dy * s).to(DY.dtype.element_ty).to(tl.float32)
    ds = (dy * n).to(DY.dtype.element_ty).to(tl.float32)
    xhat = x * r[:, None]
    weighted = dn * w[None, :]
    dot = tl.sum(weighted * xhat, axis=1) / D
    dx = (weighted - xhat * dot[:, None]) * r[:, None]
    sigmoid = 1 / (1 + tl.exp(-z))
    dz = ds * (sigmoid * (1 + z * (1 - sigmoid)))
    tl.store(DX + row[:, None] * D + col[None, :], dx, mask)
    if DZ_PITCH:
        zo = (row // HEADS).to(tl.int64)[:, None] * DZ_PITCH + (row % HEADS)[:, None] * D + col[None, :]
    else:
        zo = row[:, None] * D + col[None, :]
    tl.store(DZ + zo, dz, mask)
    dw = tl.sum(dn * xhat, axis=0)
    tl.store(DW + block * D + col, dw, col < D)


class _NormGate(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, z, weight, eps):
        if x.dtype != torch.bfloat16 or z.dtype != x.dtype:
            raise ValueError("training_norm_gate requires BF16 activations")
        if not x.is_cuda or x.shape != z.shape or not x.is_contiguous() or not z.is_contiguous():
            raise ValueError("training_norm_gate requires matching contiguous CUDA activations")
        if weight.shape != (x.shape[-1],) or not weight.is_contiguous() or weight.device != x.device:
            raise ValueError("weight must be a contiguous last-dimension vector on the input device")
        d, rows = x.shape[-1], x.numel() // x.shape[-1]
        y, n, s = [torch.empty_like(x) for _ in range(3)]
        r = torch.empty(rows, device=x.device, dtype=torch.float32)
        _forward[(triton.cdiv(rows, 16),)](x, z, weight, y, n, s, r, rows,
                                          d, triton.next_power_of_2(d), 16, eps,
                                          num_warps=4, enable_fp_fusion=False)
        ctx.save_for_backward(x, z, weight, n, s, r)
        return y

    @staticmethod
    def backward(ctx, dy):
        x, z, weight, n, s, r = ctx.saved_tensors
        d, rows = x.shape[-1], x.numel() // x.shape[-1]
        dy = dy.contiguous()
        dx, dz = torch.empty_like(x), torch.empty_like(z)
        blocks = triton.cdiv(rows, 16)
        partial = torch.empty((blocks, d), device=x.device, dtype=torch.float32)
        _backward[(blocks,)](x, z, weight, n, s, r, dy, dx, dz, partial,
                             rows, d, triton.next_power_of_2(d), 16,
                             num_warps=4, enable_fp_fusion=False)
        dw = partial.sum(0).to(weight.dtype)
        return dx, dz, dw, None


def training_norm_gate(x, z, weight, eps=1e-6):
    """Return RMSNorm(x, weight) * SiLU(z), with input and weight gradients."""
    if weight.dtype != x.dtype:
        # Mixed-dtype PyTorch RMSNorm has different intermediate/output dtypes.
        # Keep those semantics rather than silently rounding its FP32 output.
        return torch.nn.functional.rms_norm(x, (x.shape[-1],), weight, eps) * torch.nn.functional.silu(z)
    return _NormGate.apply(x, z, weight, eps)

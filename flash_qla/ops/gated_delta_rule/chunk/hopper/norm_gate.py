"""Forward-only per-head RMS normalization and SiLU output gating."""
import torch
import triton
import triton.language as tl
import torch.nn.functional as F


@torch.compile(fullgraph=True, options={'emulate_precision_casts': True})
def norm_gate_fwd_compiled(output, gate, weight, eps=1e-6):
    """Compile RMSNorm, SiLU and multiplication into one PyTorch graph.

    Keep activation-dtype rounding at the original norm/SiLU boundaries.
    gate may be a strided view of the packed projection.
    """
    normalized = F.rms_norm(output, (output.shape[-1],), weight, eps).to(output.dtype)
    return normalized * F.silu(gate.view_as(output))


@triton.jit
def _norm_gate(O, Z, W, Y, N: tl.constexpr, H: tl.constexpr,
               Z_STRIDE: tl.constexpr, EPS: tl.constexpr,
               ROWS: tl.constexpr, D: tl.constexpr):
    rows = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    cols = tl.arange(0, D)
    offsets = rows[:, None] * D + cols[None, :]
    o = tl.load(O + offsets, rows[:, None] < N, other=0).to(tl.float32)
    # Z may be a view of a packed projection with padding between tokens.
    zo = (rows // H)[:, None] * Z_STRIDE + (rows % H)[:, None] * D + cols[None, :]
    z = tl.load(Z + zo, rows[:, None] < N, other=0).to(tl.float32)
    w = tl.load(W + cols).to(tl.float32)
    inv = tl.rsqrt(tl.sum(o * o, 1) / D + EPS)
    normalized = (o * inv[:, None] * w[None, :]).to(Y.dtype.element_ty).to(tl.float32)
    # Preserve the BF16 rounding at the original norm and SiLU boundaries.
    activated = (z * tl.sigmoid(z)).to(Y.dtype.element_ty).to(tl.float32)
    tl.store(Y + offsets, normalized * activated, rows[:, None] < N)


def norm_gate_fwd(output, gate, weight, eps=1e-6, *, rows_per_cta=32, num_warps=4):
    """Return RMSNorm(output) * SiLU(gate), with activation-dtype boundaries.

    output is contiguous [B,T,H,D]; gate is [B*T,H*D], optionally a token-strided
    packed projection view. This entry point intentionally does not supply autograd.
    """
    if torch.is_grad_enabled() and any(x.requires_grad for x in (output, gate, weight)):
        raise ValueError('norm_gate_fwd is forward-only')
    if output.ndim != 4 or not output.is_contiguous():
        raise ValueError('output must be contiguous [B,T,H,D]')
    b, t, h, d = output.shape
    if gate.shape != (b*t, h*d) or gate.stride(-1) != 1:
        raise ValueError('gate must have shape [B*T,H*D] and contiguous channels')
    if weight.shape != (d,) or not weight.is_contiguous():
        raise ValueError('weight must be contiguous [D]')
    if not output.is_cuda or any(x.device != output.device for x in (gate, weight)):
        raise ValueError('all inputs must be on the same CUDA device')
    if output.dtype not in (torch.bfloat16, torch.float16, torch.float32) or gate.dtype != output.dtype:
        raise ValueError('output and gate must have matching floating activation dtypes')
    if d & (d-1) or eps <= 0:
        raise ValueError('head dimension must be a power of two and eps must be positive')
    result = torch.empty_like(output)
    n = b*t*h
    _norm_gate[(triton.cdiv(n, rows_per_cta),)](
        output, gate, weight, result, n, h, gate.stride(0), eps, rows_per_cta, d,
        num_warps=num_warps, enable_fp_fusion=False)
    return result

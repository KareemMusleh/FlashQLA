"""Forward-only causal depthwise convolution, SiLU and Q/K L2 normalization."""
import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit
def _prepare_qkv(Q, K, V, WQ, WK, WV, OQ, OK, OV, A, B, AL, DT, G, BETA,
                 T: tl.constexpr, HK: tl.constexpr, HV: tl.constexpr,
                 SQ: tl.constexpr, SK: tl.constexpr, SV: tl.constexpr,
                 D: tl.constexpr, W: tl.constexpr, EPS: tl.constexpr, BT: tl.constexpr,
                 GATES: tl.constexpr, SA: tl.constexpr, SB: tl.constexpr):
    head = tl.program_id(1)
    if head < HK:
        X, CW, Y, stride, out_stride, h = Q, WQ, OQ, SQ, HK * D, head
    elif head < 2 * HK:
        X, CW, Y, stride, out_stride, h = K, WK, OK, SK, HK * D, head - HK
    else:
        X, CW, Y, stride, out_stride, h = V, WV, OV, SV, HV * D, head - 2 * HK
    tokens = tl.program_id(0) * BT + tl.arange(0, BT)
    channels = h * D + tl.arange(0, D)
    acc = tl.full((BT, D), 0, tl.float32)
    for i in tl.static_range(W):
        prev = tokens + i - W + 1
        x = tl.load(X + prev[:, None] * stride + channels[None, :],
                    (prev[:, None] >= 0) & (prev[:, None] < T), other=0).to(tl.float32)
        weight = tl.load(CW + channels * W + i).to(tl.float32)
        acc += x * weight[None, :]
    activated = (acc * tl.sigmoid(acc)).to(Y.dtype.element_ty)
    if head < 2 * HK:
        # Match the original BF16 convolution output before FP32 L2 reduction.
        f = activated.to(tl.float32)
        inv = tl.rsqrt(tl.sum(f * f, 1) + EPS)
        activated = (f * inv[:, None]).to(Y.dtype.element_ty)
    tl.store(Y + tokens[:, None] * out_stride + channels[None, :], activated,
             tokens[:, None] < T)
    if GATES:
        if head >= 2 * HK:
            a = tl.load(A + tokens * SA + h, tokens < T, other=0).to(tl.float32)
            b = tl.load(B + tokens * SB + h, tokens < T, other=0).to(tl.float32)
            gate_x = a + tl.load(DT + h).to(tl.float32)
            # Match PyTorch's FP32 softplus threshold and libdevice arithmetic.
            # Approximate exp changes beta by ~1 ulp, which can change the
            # subsequent BF16 inverse rounding enough to affect layer outputs.
            sp = tl.where(gate_x > 20., gate_x, libdevice.log1p(libdevice.exp(gate_x)))
            g = -libdevice.exp(tl.load(AL + h).to(tl.float32)) * sp
            tl.store(G + tokens * HV + h, g, tokens < T)
            beta = libdevice.div_rn(1., 1. + libdevice.exp(-b))
            tl.store(BETA + tokens * HV + h, beta, tokens < T)


def prepare_qkv_fwd(projections, conv_weights, key_heads, value_heads, *,
                    eps=1e-6, tokens_per_cta=32, num_warps=4, gate_biases=None):
    """Return contiguous normalized Q/K and SiLU V for one unpadded sequence.

    Inputs are [T,H*128] projections, including strided slices of one packed
    GEMM, and [H*128,W] depthwise weights. Causal history is zero at token zero.
    With gate_biases=(A_log,dt_bias), projections must also contain Z,A,B and the
    return value is (QKV, (g,beta)); gate arithmetic and outputs use FP32.
    This helper intentionally does not provide autograd or state-cache handling.
    """
    q, k, v = projections[:3]
    weights = tuple(conv_weights)
    if torch.is_grad_enabled() and any(x.requires_grad for x in (q, k, v, *weights)):
        raise ValueError('prepare_qkv_fwd is forward-only')
    d, t = 128, q.shape[0]
    sizes = (key_heads*d, key_heads*d, value_heads*d)
    if key_heads <= 0 or value_heads <= 0 or len(weights) != 3:
        raise ValueError('three convolution weights and positive head counts are required')
    w = weights[0].shape[-1]
    if w not in (2, 3, 4) or eps <= 0:
        raise ValueError('convolution width must be 2, 3 or 4 and eps must be positive')
    for x, weight, channels in zip((q, k, v), weights, sizes):
        if x.shape != (t, channels) or x.stride(-1) != 1:
            raise ValueError('projections must have shape [T,H*128] with contiguous channels')
        if weight.shape != (channels, w) or not weight.is_contiguous():
            raise ValueError('weights must be contiguous [H*128,W]')
        if not x.is_cuda or x.device != q.device or weight.device != q.device:
            raise ValueError('all inputs must be on the same CUDA device')
        if x.dtype != q.dtype or weight.dtype != q.dtype:
            raise ValueError('projections and weights must have the same dtype')
    if q.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise ValueError('unsupported activation dtype')
    outputs = tuple(torch.empty((1, t, h, d), device=q.device, dtype=q.dtype)
                    for h in (key_heads, key_heads, value_heads))
    with_gates = gate_biases is not None
    if with_gates:
        if len(projections) != 6 or len(gate_biases) != 2:
            raise ValueError('fused gates require six projections and two gate biases')
        a, b = projections[4:]
        al, dt = gate_biases
        for x in (a, b):
            if x.shape != (t, value_heads) or x.stride(-1) != 1 or x.device != q.device or x.dtype != q.dtype:
                raise ValueError('gate projections must be [T,Hv] with matching dtype/device')
        for x in (al, dt):
            if x.shape != (value_heads,) or not x.is_contiguous() or x.device != q.device:
                raise ValueError('gate biases must be contiguous [Hv] on the input device')
        if torch.is_grad_enabled() and any(x.requires_grad for x in (a,b,al,dt)):
            raise ValueError('prepare_qkv_fwd is forward-only')
        gates = tuple(torch.empty((1,t,value_heads),device=q.device,dtype=torch.float32) for _ in range(2))
        sa, sb = a.stride(0), b.stride(0)
    else:
        a = b = al = dt = q
        gates = (q, q)
        sa = sb = 0
    _prepare_qkv[(triton.cdiv(t, tokens_per_cta), 2*key_heads+value_heads)](
        q, k, v, *weights, *outputs, a, b, al, dt, *gates, t, key_heads, value_heads,
        q.stride(0), k.stride(0), v.stride(0), d, w, eps, tokens_per_cta, with_gates, sa, sb,
        num_warps=num_warps, enable_fp_fusion=True)
    return (outputs, gates) if with_gates else outputs

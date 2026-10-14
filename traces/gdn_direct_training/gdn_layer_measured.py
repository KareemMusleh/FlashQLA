"""Experimental single-sequence GDN training with direct projection gradients.

BF16, bias-free, Dk=Dv=128, zero convolution history, no distributed CP/state
cache. Uses the existing optimized forward and uncached default GDR backward.
"""
import copy

import torch
import torch.nn.functional as F
from torch.autograd.function import once_differentiable
import triton
import triton.language as tl
from triton.language.extra import libdevice

from flash_qla.ops.gated_delta_rule.chunk import ChunkGatedDeltaRuleFunction


@triton.jit
def _norm_gate_backward(O, Z, W, DY, DO, DZ, DW, N: tl.constexpr,
                        H: tl.constexpr, D: tl.constexpr, STRIDE: tl.constexpr,
                        EPS: tl.constexpr, R: tl.constexpr):
    rows = tl.program_id(0) * R + tl.arange(0, R)
    c = tl.arange(0, D)
    off = rows[:, None] * D + c[None, :]
    mask = rows[:, None] < N
    o = tl.load(O + off, mask, 0).to(tl.float32)
    z = tl.load(Z + off, mask, 0).to(tl.float32)
    w = tl.load(W + c).to(tl.float32)
    dy = tl.load(DY + off, mask, 0).to(tl.float32)
    inv = tl.rsqrt(tl.sum(o * o, 1) / D + EPS)
    s = libdevice.div_rn(1., 1. + libdevice.exp(-z))
    activated = (z * s).to(DY.dtype.element_ty).to(tl.float32)
    normalized = (o * inv[:, None] * w[None, :]).to(DY.dtype.element_ty).to(tl.float32)
    dn = (dy * activated).to(DY.dtype.element_ty).to(tl.float32)
    dz = (dy * normalized).to(DY.dtype.element_ty).to(tl.float32)
    dz = dz * s * (1. + z * (1. - s))
    weighted = dn * w[None, :]
    do = (weighted - o * (tl.sum(weighted * o, 1) / D * inv * inv)[:, None]) * inv[:, None]
    tl.store(DO + off, do, mask)
    zo = (rows // H).to(tl.int64)[:, None] * STRIDE + (rows % H)[:, None] * D + c[None, :]
    tl.store(DZ + zo, dz, mask)
    partial = tl.sum(dn * o * inv[:, None], 0)
    tl.store(DW + tl.program_id(0) * D + c, partial)


@triton.jit
def _gate_backward(A, B, AL, DT, DG, DB, DA, DBP, DAL, DDT,
                   T: tl.constexpr, H: tl.constexpr, STRIDE: tl.constexpr,
                   R: tl.constexpr, BH: tl.constexpr, PREPARED: tl.constexpr):
    t = tl.program_id(0) * R + tl.arange(0, R)
    h = tl.arange(0, BH)
    off = t[:, None] * H + h[None, :]
    mask = (t[:, None] < T) & (h[None, :] < H)
    a = tl.load(A + off, mask, 0).to(tl.float32)
    b = tl.load(B + off, mask, 0).to(tl.float32)
    al = tl.load(AL + h, h < H, 0).to(tl.float32)
    dt = tl.load(DT + h, h < H, 0).to(tl.float32)
    dg = tl.load(DG + off, mask, 0).to(tl.float32)
    db = tl.load(DB + off, mask, 0).to(tl.float32)
    u = a if PREPARED else a + dt[None, :]
    sp = tl.where(u > 20., u, libdevice.log1p(libdevice.exp(u)))
    decay = libdevice.exp(al)
    # Preserve the eager FP32 multiplication/division boundaries.
    ds = dg * (-decay[None, :])
    da = tl.where(u > 20., ds, libdevice.div_rn(ds, 1. + libdevice.exp(-u)))
    beta = b if PREPARED else libdevice.div_rn(1., 1. + libdevice.exp(-b))
    dbeta = db * beta * (1. - beta)
    out = t.to(tl.int64)[:, None] * STRIDE + h[None, :]
    tl.store(DA + out, da, mask)
    tl.store(DBP + out, dbeta, mask)
    dl = tl.where(mask, (-dg * sp) * decay[None, :], 0.)
    dd = tl.where(mask, da, 0.)
    tl.store(DAL + tl.program_id(0) * H + h, tl.sum(dl, 0), h < H)
    tl.store(DDT + tl.program_id(0) * H + h, tl.sum(dd, 0), h < H)


def norm_gate_backward_into(o, z, weight, dy, dz, eps):
    """Allocate recurrence/norm-weight gradients; write dZ into its final slice."""
    t, h, d = o.shape[1:]
    n, rows = t * h, 32
    do = torch.empty_like(o)
    partial = torch.empty((triton.cdiv(n, rows), d), device=o.device, dtype=torch.float32)
    _norm_gate_backward[(partial.shape[0],)](
        o, z, weight, dy, do, dz, partial, n, h, d, dz.stride(0), eps, rows,
        num_warps=4, enable_fp_fusion=False,
    )
    return do, partial.sum(0).to(weight.dtype)


def gate_backward_into(a, b, al, dt, dg, db, da, dbeta, *, prepared=False):
    t, h = a.shape
    rows = 128
    partials = [torch.empty((triton.cdiv(t, rows), h), device=a.device, dtype=torch.float32)
                for _ in range(2)]
    _gate_backward[(partials[0].shape[0],)](
        a, b, al, dt, dg, db, da, dbeta, *partials, t, h, da.stride(0), rows,
        triton.next_power_of_2(h), num_warps=4, enable_fp_fusion=False,
        PREPARED=prepared,
    )
    return partials[0].sum(0).to(al.dtype), partials[1].sum(0).to(dt.dtype)


class _GDRContext:
    def save_for_backward(self, *tensors):
        self.saved_tensors = tensors


class _DirectGDN(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, wq, wk, wv, wz, wa, wb, cq, ck, cv, al, dt, nw, wo,
                hk, hv, eps, partitions, direct_gradients):
        from causal_conv1d.cpp_functions import causal_conv1d_fwd_function
        weights = (wq, wk, wv, wz, wa, wb)
        p = tuple(F.linear(x, w) for w in weights)
        conv = tuple(causal_conv1d_fwd_function(
            z[None].transpose(1, 2), w, None, None, None, None, True,
        ).transpose(1, 2) for z, w in zip(p[:3], (cq, ck, cv)))
        q = conv[0].view(1, -1, hk, 128)
        k = conv[1].view_as(q)
        v = conv[2].view(1, -1, hv, 128)
        gate_input = p[4].float() + dt.float()
        g = (-al.float().exp() * F.softplus(gate_input))[None]
        beta = p[5].float().sigmoid()[None]
        inner = _GDRContext()
        o, _ = ChunkGatedDeltaRuleFunction.forward(
            inner, q, k, v, g, beta, output_final_state=False,
            use_qk_l2norm_in_kernel=True, enable_fwd_cp_cache=False,
            is_train=True, optimized_partitions=partitions, decay_threshold=None,
            cache_optimized_inverse=False,
        )
        gated = F.rms_norm(o, (128,), nw, eps) * F.silu(p[3].view_as(o))
        result = F.linear(gated.view(x.shape[0], -1), wo)
        # All are already retained by the unfused reference. No normalized
        # output, activated gate, inverse, or new statistics are retained.
        ctx.save_for_backward(x, *weights, cq, ck, cv, al, dt, nw, wo,
                              *p[:4], gate_input, o, gated, *inner.saved_tensors)
        inner.saved_tensors = None
        ctx.inner = inner
        ctx.eps = eps
        ctx.direct_gradients = direct_gradients
        return result

    @staticmethod
    @once_differentiable
    def backward(ctx, dy):
        from causal_conv1d.cpp_functions import causal_conv1d_bwd_function
        saved = ctx.saved_tensors
        x, wq, wk, wv, wz, wa, wb, cq, ck, cv, al, dt, nw, wo = saved[:14]
        weights = (wq, wk, wv, wz, wa, wb)
        p = saved[14:18]
        gate_input, o, gated = saved[18:21]
        inner = copy.copy(ctx.inner)
        inner.saved_tensors = saved[21:]
        # Preserve retain_graph semantics while allowing consumed activations
        # to die during ordinary backward, as they do in the unfused graph.
        ctx.maybe_clear_saved_tensors()
        del saved
        beta = inner.saved_tensors[6].view(x.shape[0], -1)
        sizes = [w.shape[0] for w in weights]
        if ctx.direct_gradients:
            # Dao's channel-last convolution requires an eight-element token
            # pitch. Small head counts may need padding after the beta slice.
            pitch = triton.cdiv(sum(sizes), 8) * 8
            storage = torch.empty((x.shape[0], pitch), device=x.device, dtype=x.dtype)
            packed = storage[:, :sum(sizes)]
            slices = packed.split(sizes, dim=-1)
        else:
            # Matched ablation: same kernels/math, with separate destinations
            # followed by concatenation. Not used by the default entry point.
            slices = [torch.empty((x.shape[0], s), device=x.device, dtype=x.dtype) for s in sizes]
        dwo = dy.T @ gated.view(x.shape[0], -1)
        dgate = (dy @ wo).view_as(o)
        do, dnw = norm_gate_backward_into(o, p[3], nw, dgate, slices[3], ctx.eps)
        del dgate, o, gated
        p = p[:3]
        dq, dk, dv, dg, db = ChunkGatedDeltaRuleFunction.backward(inner, do, None)[:5]
        del do, inner
        conv_grads = []
        for z, w, dz, destination in zip(p[:3], (cq, ck, cv), (dq, dk, dv), slices[:3]):
            dout = dz.view(1, x.shape[0], -1).transpose(1, 2)
            _, dw, _, _ = causal_conv1d_bwd_function(
                z[None].transpose(1, 2), w, None, dout, None, None, None,
                destination[None].transpose(1, 2), False, True,
            )
            conv_grads.append(dw)
        dal, ddt = gate_backward_into(gate_input, beta, al, dt, dg, db, slices[4], slices[5],
                                     prepared=True)
        del dq, dk, dv, dg, db
        if not ctx.direct_gradients:
            packed = torch.cat(slices, dim=-1)
        dx = packed @ torch.cat(weights)
        dw = (packed.T @ x).split(sizes)
        return dx, *dw, *conv_grads, dal, ddt, dnw, dwo, None, None, None, None, None


def direct_gdn_layer(x, weights, conv_weights, a_log, dt_bias, norm_weight,
                     output_weight, *, key_heads, value_heads, eps=1e-6, partitions=8,
                     direct_gradients=True):
    """Full bias-free GDN layer with first-order, direct-buffer backward.

    Explicit experimental entry point: one unpadded sequence, no initial/final
    state, CUDA BF16 parameters/activations, FP32 decay parameters. Higher-order
    gradients and autocast are not supported by this entry point.
    direct_gradients=False is a matched runtime-packing ablation.
    """
    kd, vd = key_heads * 128, value_heads * 128
    sizes = (kd, kd, vd, vd, value_heads, value_heads)
    if x.ndim != 2 or not x.is_cuda or x.dtype != torch.bfloat16 or not x.is_contiguous():
        raise ValueError("expected contiguous CUDA BF16 [tokens, hidden] input")
    if torch.is_autocast_enabled("cuda"):
        raise ValueError("direct_gdn_layer requires explicit BF16 tensors, without autocast")
    if key_heads <= 0 or value_heads % key_heads or len(weights) != 6 or len(conv_weights) != 3:
        raise ValueError("expected six projections, three convolutions, and compatible head counts")
    shapes = [(s, x.shape[1]) for s in sizes] + [(s, conv_weights[0].shape[-1]) for s in sizes[:3]]
    tensors = (*weights, *conv_weights)
    for t, shape in zip(tensors, shapes):
        if t.shape != shape or t.dtype != x.dtype or t.device != x.device or not t.is_contiguous():
            raise ValueError("invalid projection or convolution weight")
    if conv_weights[0].shape[-1] not in (2, 3, 4):
        raise ValueError("convolution width must be 2, 3, or 4")
    for t, shape in ((a_log, (value_heads,)), (dt_bias, (value_heads,)),
                     (norm_weight, (128,)), (output_weight, (x.shape[1], vd))):
        if t.shape != shape or t.device != x.device or not t.is_contiguous():
            raise ValueError("invalid gate, norm, or output weight")
    if a_log.dtype != torch.float32 or dt_bias.dtype != torch.float32 or norm_weight.dtype != x.dtype or output_weight.dtype != x.dtype:
        raise ValueError("expected FP32 decay parameters and BF16 norm/output weights")
    if eps <= 0 or partitions < 1:
        raise ValueError("eps and partitions must be positive")
    if x.shape[0] == 0 or x.shape[0] % (64 * partitions):
        raise ValueError("tokens must be a positive multiple of 64 * partitions")
    if torch.cuda.get_device_capability(x.device)[0] != 9:
        raise ValueError("direct_gdn_layer currently requires Hopper")
    return _DirectGDN.apply(x, *weights, *conv_weights, a_log, dt_bias,
                            norm_weight, output_weight, key_heads, value_heads, eps, partitions,
                            direct_gradients)

"""Chunk-parallel implicit token gradients, with a separate state adjoint scan."""
import torch
import tilelang
import tilelang.language as T
from .cp_bwd import fused_gdr_dh_ws


@tilelang.jit(out_idx=[9, 10, 11, 12, 13], pass_configs={
    tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
})
def _parallel_bwd(tokens, hk, hv, scale, v_first):
    C, D = 64, 128

    @T.macro
    def store_gradient(src, dst):
        T.fence_proxy_async()
        T.sync_threads()
        if T.get_thread_binding() < 32:
            T.tma_copy(src, dst)
            T.tma_store_wait(0, read=True)
        # The shared tile can be reused only after the store has read it.
        T.sync_threads()

    @T.prim_func
    def kernel(
        Q: T.Tensor((1, tokens, hk, D), "bfloat16"),
        K: T.Tensor((1, tokens, hk, D), "bfloat16"),
        V: T.Tensor((1, tokens, hv, D), "bfloat16"),
        A: T.Tensor((1, tokens, hv, C), "bfloat16"),
        G: T.Tensor((1, tokens, hv), "float32"),
        B: T.Tensor((1, tokens, hv), "float32"),
        DO: T.Tensor((1, tokens, hv, D), "bfloat16"),
        H: T.Tensor((1, tokens // C, hv, D, D), "bfloat16"),
        DH: T.Tensor((1, tokens // C, hv, D, D), "bfloat16"),
        DQ: T.Tensor((1, tokens, hv, D), "bfloat16"),
        DK: T.Tensor((1, tokens, hv, D), "bfloat16"),
        DV: T.Tensor((1, tokens, hv, D), "bfloat16"),
        DG: T.Tensor((1, tokens, hv), "float32"),
        DB: T.Tensor((1, tokens, hv), "float32"),
    ):
        with T.Kernel(tokens // C * hv, threads=256) as (block,):
            chunk, h = block // hv, block % hv
            start = chunk * C
            qs = T.alloc_shared((C, D), "bfloat16")
            ks = T.alloc_shared((C, D), "bfloat16")
            vs = T.alloc_shared((C, D), "bfloat16")
            ds = T.alloc_shared((C, D), "bfloat16")
            ws = T.alloc_shared((C, D), "bfloat16")
            us = T.alloc_shared((C, D), "bfloat16")
            zs = T.alloc_shared((C, D), "bfloat16")
            hs = T.alloc_shared((D, D), "bfloat16")
            dhs = T.alloc_shared((D, D), "bfloat16")
            a = T.alloc_shared((C, C), "bfloat16")
            ps = T.alloc_shared((C, C), "bfloat16")
            dps = T.alloc_shared((C, C), "bfloat16")
            ex = T.alloc_shared((C,), "float32")
            invex = T.alloc_shared((C,), "float32")
            beta = T.alloc_shared((C,), "float32")
            input_ready = T.alloc_barrier(arrive_count=32)
            acc = T.alloc_fragment((C, D), "float32")
            dk = T.alloc_fragment((C, D), "float32")
            prod = T.alloc_fragment((C, D), "float32")
            tile = T.alloc_fragment((C, C), "float32")
            dg = T.alloc_fragment((C,), "float32")
            db = T.alloc_fragment((C,), "float32")
            partial = T.alloc_fragment((256,), "float32")
            last = T.alloc_fragment((1,), "float32")
            T.annotate_layout({
                partial: tilelang.layout.Fragment((256,),
                    forward_thread_fn=lambda i: i,
                    forward_index_fn=lambda i: 0),
                acc: tilelang.layout.Fragment((C, D),
                    forward_thread_fn=lambda i, j: j // 64 * 128 + i % 64 // 16 * 32 + i % 8 * 4 + j % 8 // 2,
                    forward_index_fn=lambda i, j: j % 64 // 8 * 4 + i % 16 // 8 * 2 + j % 2),
            })

            # One load warp issues all independent TMA transfers. Scalar
            # gate preparation overlaps them; the arrival/wait publishes
            # the input tiles before any tensor-core consumer reads them.
            T.sync_threads()
            if T.get_thread_binding() < 32:
                T.tma_copy(Q[0, start:start + C, h // (hv // hk), :], qs, barrier=input_ready)
                T.tma_copy(K[0, start:start + C, h // (hv // hk), :], ks, barrier=input_ready)
                T.tma_copy(V[0, start:start + C, h, :], vs, barrier=input_ready)
                T.tma_copy(DO[0, start:start + C, h, :], ds, barrier=input_ready)
                T.tma_copy(A[0, start:start + C, h, :], a, barrier=input_ready)
                T.tma_copy(H[0, start // C, h, :, :], hs, barrier=input_ready)
                T.tma_copy(DH[0, start // C, h, :, :], dhs, barrier=input_ready)
                T.barrier_arrive(input_ready)
            T.copy(B[0, start:start + C, h], beta)
            for i in T.Parallel(C):
                ex[i] = T.exp2(G[0, start + i, h] * 1.4426950408889634)
                invex[i] = T.exp2(-G[0, start + i, h] * 1.4426950408889634)
            T.barrier_wait(input_ready, 0)
            T.clear(dg)
            T.clear(partial)
            for i, j in T.Parallel(D, D):
                if v_first:
                    partial[j // 64 * 128 + i % 64 // 16 * 32 + i % 8 * 4 + j % 8 // 2] += dhs[i, j] * hs[i, j]
                else:
                    partial[j // 64 * 128 + i % 64 // 16 * 32 + i % 8 * 4 + j % 8 // 2] += dhs[j, i] * hs[j, i]
            T.reduce_sum(partial, last, dim=0)
            # Scalar chunk-end decay derivative, before updating dh.
            for i in T.Parallel(C):
                if i == C - 1:
                    dg[i] = ex[C - 1] * last[0]

            # W = Gamma^-1 V - K S. U = A diag(beta) W.
            T.gemm(ks, hs, acc, transpose_B=v_first, clear_accum=True)
            for i, j in T.Parallel(C, D):
                ws[i, j] = vs[i, j] * invex[i] - acc[i, j]
                zs[i, j] = beta[i] * ws[i, j]
                ds[i, j] = ds[i, j] * (scale * ex[i])
            T.gemm(a, zs, acc, clear_accum=True)
            T.copy(acc, us)

            # P = causal(Q K.T); dU = gamma K dS_out + P.T D.
            T.gemm(qs, ks, tile, transpose_B=True, clear_accum=True)
            for i, j in T.Parallel(C, C):
                ps[i, j] = T.if_then_else(i >= j, tile[i, j], 0.0)
            T.gemm(ks, dhs, acc, transpose_B=v_first, clear_accum=True)
            T.clear(partial)
            for i, j in T.Parallel(C, D):
                acc[i, j] *= ex[C - 1]
                partial[j // 64 * 128 + i % 64 // 16 * 32 + i % 8 * 4 + j % 8 // 2] += acc[i, j] * us[i, j]
            T.reduce_sum(partial, last, dim=0)
            for i in T.Parallel(C):
                if i == C - 1:
                    dg[i] += last[0]
            T.gemm(ps, ds, acc, transpose_A=True)
            T.copy(acc, zs)
            T.gemm(a, zs, acc, transpose_A=True, clear_accum=True)
            T.copy(acc, zs)

            # Z W part of db. Keep W until this contraction finishes.
            for i, j in T.Parallel(C, D):
                prod[i, j] = acc[i, j] * ws[i, j]
            T.reduce_sum(prod, db, dim=1)
            for i, j in T.Parallel(C, D):
                ws[i, j] = beta[i] * acc[i, j]
                prod[i, j] = -vs[i, j] * invex[i] * ws[i, j]
                vs[i, j] = invex[i] * ws[i, j]
            T.reduce_sum(prod, dg, dim=1, clear=False)
            store_gradient(vs, DV[0, start:start + C, h, :])

            # dQ = D S.T + dP K; gates get row contractions of these
            # two output terms, with no 64x64 gate-ratio derivative.
            T.gemm(ds, hs, acc, transpose_B=not v_first, clear_accum=True)
            for i, j in T.Parallel(C, D):
                prod[i, j] = acc[i, j] * qs[i, j]
            T.reduce_sum(prod, dg, dim=1, clear=False)
            T.copy(acc, vs)
            T.gemm(ds, us, tile, transpose_B=True, clear_accum=True)
            for i, j in T.Parallel(C, C):
                tile[i, j] = T.if_then_else(i >= j, tile[i, j], 0.0)
                dps[i, j] = tile[i, j]
                tile[i, j] *= ps[i, j]
            T.reduce_sum(tile, dg, dim=1, clear=False)
            T.copy(vs, acc)
            T.gemm(dps, ks, acc)
            T.copy(acc, vs)
            store_gradient(vs, DQ[0, start:start + C, h, :])

            # Direct K contributions from output, state update, and W.
            T.gemm(us, dhs, dk, transpose_B=not v_first, clear_accum=True)
            for i, j in T.Parallel(C, D):
                dk[i, j] *= ex[C - 1]
            T.gemm(dps, qs, dk, transpose_A=True)
            T.gemm(ws, hs, acc, transpose_B=not v_first, clear_accum=True)
            for i, j in T.Parallel(C, D):
                dk[i, j] -= acc[i, j]

            # F = -strict_lower(Z U.T). Reuse A's dead storage.
            T.gemm(zs, us, tile, transpose_B=True, clear_accum=True)
            for i, j in T.Parallel(C, C):
                a[i, j] = T.if_then_else(i > j, -tile[i, j], 0.0)
            T.gemm(a, ks, acc, clear_accum=True)
            for i, j in T.Parallel(C, D):
                prod[i, j] = acc[i, j] * ks[i, j]
                dk[i, j] += beta[i] * acc[i, j]
                vs[i, j] = beta[i] * ks[i, j]
            T.reduce_sum(prod, db, dim=1, clear=False)
            T.gemm(a, vs, dk, transpose_A=True)
            T.copy(dk, vs)
            store_gradient(vs, DK[0, start:start + C, h, :])
            for i in T.Parallel(C):
                DG[0, start + i, h] = dg[i]
                DB[0, start + i, h] = db[i]


            if T.get_thread_binding() < 32:
                T.tma_store_wait(0, read=False)
    return kernel


def parallel_gdr_bwd(q, k, v, a, g, b, do, dht, h, scale=None,
                     cu_seqlens=None, chunk_size=64, state_v_first=False):
    """Experimental split backward; B=1,D=128 and aligned partition boundaries.

    State adjoints use the existing warp-specialized recurrence. The remaining
    token gradients run with one CTA per head/chunk, with no serial recurrence.
    """
    assert q.shape == k.shape and q.shape[0] == 1
    tokens, hk, hv = q.shape[1], q.shape[2], v.shape[2]
    assert tokens % 64 == 0 and chunk_size == 64
    assert q.shape[-1] == v.shape[-1] == 128 and hv % hk == 0
    assert h.shape == (1, tokens // 64, hv, 128, 128)
    factor = 128**-0.5 if scale is None else scale
    # The legacy helper treats scale=0 as its default; feed zero upstream
    # output gradients in that case so the state-only recurrence stays correct.
    state_do = torch.zeros_like(do) if factor == 0 else do
    dh, dh0 = fused_gdr_dh_ws(q, k, a, g, b, state_do, dht,
                             output_dh=True, output_dh0=True,
                             scale=1.0 if factor == 0 else factor,
                             cu_seqlens=cu_seqlens, state_v_first=state_v_first)
    dq, dk, dv, dg, db = _parallel_bwd(tokens, hk, hv, factor, state_v_first)(
        q, k, v, a, g, b, do, h, dh)
    return dq, dk, dv, dg, db, dh0

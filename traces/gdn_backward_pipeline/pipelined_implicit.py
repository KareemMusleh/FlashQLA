"""Experimental implicit-solve GDN adjoint for aligned Hopper BF16 inputs.

Uses the default unfurled inverse and saved chunk states. Gates are factored
into vectors; no pairwise gate-ratio matrix or inverse gradient is formed.
For L = I + diag(beta) strict_lower(K K.T), L U = diag(beta) W:
Z = L^-T dU, dW = diag(beta) Z, F = -strict_lower(Z U.T).
Then db = rowsum(Z W) + rowsum(K (F K)), and
dK_solve = diag(beta) (F K) + F.T (diag(beta) K).
The beta adjoint never divides by beta, including at beta=0.
"""

from functools import lru_cache

import torch
import tilelang
import tilelang.language as T


@tilelang.jit(out_idx=[10, 11, 12, 13, 14, 15], pass_configs={
    tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    tilelang.PassConfigKey.TL_DISABLE_THREAD_STORAGE_SYNC: True,
})
def _pipelined_bwd(tokens, hk, hv, parts, scale, v_first, use_dht, cu_dtype):
    C, D = 64, 128

    @T.macro
    def store_gradient(src, dst):
        T.fence_proxy_async()
        T.sync_threads(1, 256)
        if T.get_thread_binding() < 32:
            T.tma_copy(src, dst)
            T.tma_store_wait(0, read=True)
        # The shared tile can be reused only after the store has read it.
        T.sync_threads(1, 256)

    @T.macro
    def gemm(lhs, rhs, result, transpose_A=False, transpose_B=False, clear_accum=False):
        tx = T.get_thread_binding()
        T.fence_proxy_async()
        if tx < 128:
            T.sync_threads(8, 256)
        else:
            T.sync_threads(9, 256)
        T.wgmma_gemm(lhs, rhs, result, transpose_A=transpose_A,
                     transpose_B=transpose_B, clear_accum=clear_accum,
                     policy=T.GemmWarpPolicy.FullCol)
        if tx < 128:
            T.named_barrier_arrive(9, 256)
        else:
            T.named_barrier_arrive(8, 256)
        T.wait_wgmma(0)

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
        DHT: T.Tensor((parts, hv, D, D), "float32"),
        CU: T.Tensor((parts + 1,), cu_dtype),
        DQ: T.Tensor((1, tokens, hv, D), "bfloat16"),
        DK: T.Tensor((1, tokens, hv, D), "bfloat16"),
        DV: T.Tensor((1, tokens, hv, D), "bfloat16"),
        DG: T.Tensor((1, tokens, hv), "float32"),
        DB: T.Tensor((1, tokens, hv), "float32"),
        DH0: T.Tensor((parts, hv, D, D), "float32"),
    ):
        with T.Kernel(parts * hv, threads=384) as (block,):
            p, h = block // hv, block % hv
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
            scalars_ready = T.alloc_barrier(arrive_count=128)
            ps_ready = T.alloc_barrier(arrive_count=128)
            data_free = T.alloc_barrier(arrive_count=256)
            dh = T.alloc_fragment((D, D), "float32")
            acc = T.alloc_fragment((C, D), "float32")
            dk = T.alloc_fragment((C, D), "float32")
            tile = T.alloc_fragment((C, C), "float32")
            dg = T.alloc_fragment((C,), "float32")
            db = T.alloc_fragment((C,), "float32")
            partial = T.alloc_fragment((256,), "float32")
            last = T.alloc_fragment((1,), "float32")
            T.annotate_layout({
                partial: tilelang.layout.Fragment((256,),
                    forward_thread_fn=lambda i: i,
                    forward_index_fn=lambda i: 0),
                dh: tilelang.layout.Fragment((D, D),
                    forward_thread_fn=lambda i, j: j // 64 * 128 + i % 64 // 16 * 32 + i % 8 * 4 + j % 8 // 2,
                    forward_index_fn=lambda i, j: i // 64 * 32 + j % 64 // 8 * 4 + i % 16 // 8 * 2 + j % 2),
                acc: tilelang.layout.Fragment((C, D),
                    forward_thread_fn=lambda i, j: j // 64 * 128 + i % 64 // 16 * 32 + i % 8 * 4 + j % 8 // 2,
                    forward_index_fn=lambda i, j: j % 64 // 8 * 4 + i % 16 // 8 * 2 + j % 2),
            })

            if T.get_thread_binding() < 256:
                T.set_max_nreg(216, 1)
                if T.get_thread_binding() >= 128:
                    T.named_barrier_arrive(8, 256)
                if use_dht:
                    for i, j in T.Parallel(D, D):
                        if v_first:
                            dh[i, j] = DHT[p, h, i, j]
                        else:
                            dh[i, j] = DHT[p, h, j, i]
                else:
                    T.clear(dh)

                for c in T.serial((CU[p + 1] - CU[p]) // C):
                    start = CU[p + 1] - (c + 1) * C
                    T.barrier_wait(scalars_ready, c % 2)
                    T.copy(dh, dhs)
                    T.clear(dg)
                    T.clear(partial)
                    for i, j in T.Parallel(D, D):
                        if v_first:
                            partial[j // 64 * 128 + i % 64 // 16 * 32 + i % 8 * 4 + j % 8 // 2] += dh[i, j] * hs[i, j]
                        else:
                            partial[j // 64 * 128 + i % 64 // 16 * 32 + i % 8 * 4 + j % 8 // 2] += dh[i, j] * hs[j, i]
                    T.reduce_sum(partial, last, dim=0)
                    # Scalar chunk-end decay derivative, before updating dh.
                    for i in T.Parallel(C):
                        if i == C - 1:
                            dg[i] = ex[C - 1] * last[0]

                    # W = Gamma^-1 V - K S. U = A diag(beta) W.
                    gemm(ks, hs, acc, transpose_B=v_first, clear_accum=True)
                    for i, j in T.Parallel(C, D):
                        ws[i, j] = vs[i, j] * invex[i] - acc[i, j]
                        zs[i, j] = beta[i] * ws[i, j]
                        ds[i, j] = ds[i, j] * (scale * ex[i])
                    gemm(a, zs, acc, clear_accum=True)
                    T.copy(acc, us)

                    # P = causal(Q K.T); dU = gamma K dS_out + P.T D.
                    T.barrier_wait(ps_ready, c % 2)
                    gemm(ks, dhs, acc, transpose_B=True, clear_accum=True)
                    T.clear(partial)
                    for i, j in T.Parallel(C, D):
                        acc[i, j] *= ex[C - 1]
                        partial[j // 64 * 128 + i % 64 // 16 * 32 + i % 8 * 4 + j % 8 // 2] += acc[i, j] * us[i, j]
                    T.reduce_sum(partial, last, dim=0)
                    for i in T.Parallel(C):
                        if i == C - 1:
                            dg[i] += last[0]
                    gemm(ps, ds, acc, transpose_A=True)
                    T.copy(acc, zs)
                    gemm(a, zs, acc, transpose_A=True, clear_accum=True)
                    T.copy(acc, zs)

                    # Temporarily use dk to retain Z while acc holds row products.
                    T.copy(acc, dk)
                    # Z W part of db. Keep W until this contraction finishes.
                    for i, j in T.Parallel(C, D):
                        acc[i, j] = acc[i, j] * ws[i, j]
                    T.reduce_sum(acc, db, dim=1)
                    T.copy(dk, acc)
                    for i, j in T.Parallel(C, D):
                        ws[i, j] = beta[i] * acc[i, j]
                        acc[i, j] = -vs[i, j] * invex[i] * ws[i, j]
                        vs[i, j] = invex[i] * ws[i, j]
                    T.reduce_sum(acc, dg, dim=1, clear=False)
                    store_gradient(vs, DV[0, start:start + C, h, :])

                    # dQ = D S.T + dP K; gates get row contractions of these
                    # two output terms, with no 64x64 gate-ratio derivative.
                    gemm(ds, hs, acc, transpose_B=not v_first, clear_accum=True)
                    T.copy(acc, vs)
                    for i, j in T.Parallel(C, D):
                        acc[i, j] = acc[i, j] * qs[i, j]
                    T.reduce_sum(acc, dg, dim=1, clear=False)
                    gemm(ds, us, tile, transpose_B=True, clear_accum=True)
                    for i, j in T.Parallel(C, C):
                        tile[i, j] = T.if_then_else(i >= j, tile[i, j], 0.0)
                        dps[i, j] = tile[i, j]
                        tile[i, j] *= ps[i, j]
                    T.reduce_sum(tile, dg, dim=1, clear=False)
                    T.copy(vs, acc)
                    gemm(dps, ks, acc)
                    T.copy(acc, vs)
                    store_gradient(vs, DQ[0, start:start + C, h, :])

                    # Direct K contributions from output, state update, and W.
                    gemm(us, dhs, dk, clear_accum=True)
                    for i, j in T.Parallel(C, D):
                        dk[i, j] *= ex[C - 1]
                    gemm(dps, qs, dk, transpose_A=True)
                    gemm(ws, hs, acc, transpose_B=not v_first, clear_accum=True)
                    for i, j in T.Parallel(C, D):
                        dk[i, j] -= acc[i, j]

                    # F = -strict_lower(Z U.T). Reuse A's dead storage.
                    gemm(zs, us, tile, transpose_B=True, clear_accum=True)
                    for i, j in T.Parallel(C, C):
                        a[i, j] = T.if_then_else(i > j, -tile[i, j], 0.0)
                    gemm(a, ks, acc, clear_accum=True)
                    for i, j in T.Parallel(C, D):
                        dk[i, j] += beta[i] * acc[i, j]
                        acc[i, j] *= ks[i, j]
                        vs[i, j] = beta[i] * ks[i, j]
                    T.reduce_sum(acc, db, dim=1, clear=False)
                    gemm(a, vs, dk, transpose_A=True)
                    T.copy(dk, vs)
                    store_gradient(vs, DK[0, start:start + C, h, :])
                    for i in T.Parallel(C):
                        DG[0, start + i, h] = dg[i]
                        DB[0, start + i, h] = db[i]

                    # dS_in.T = gamma dS_out.T + D.T Q - dW.T K.
                    for i, j in T.Parallel(D, D):
                        dh[i, j] *= ex[C - 1]
                    gemm(ds, qs, dh, transpose_A=True)
                    for i, j in T.Parallel(C, D):
                        ws[i, j] = -ws[i, j]
                    gemm(ws, ks, dh, transpose_A=True)
                    T.barrier_arrive(data_free)

                if T.get_thread_binding() < 32:
                    T.tma_store_wait(0, read=False)
                for i, j in T.Parallel(D, D):
                    if v_first:
                        DH0[p, h, i, j] = dh[i, j]
                    else:
                        DH0[p, h, j, i] = dh[i, j]
            else:
                T.set_max_nreg(64, 0)
                qk = T.alloc_fragment((C, C), "float32")
                for c in T.serial((CU[p + 1] - CU[p]) // C):
                    start = CU[p + 1] - (c + 1) * C
                    if c > 0:
                        T.barrier_wait(data_free, (c - 1) % 2)
                    if T.get_thread_binding() < 288:
                        T.tma_copy(Q[0, start:start + C, h // (hv // hk), :], qs, barrier=input_ready)
                        T.tma_copy(K[0, start:start + C, h // (hv // hk), :], ks, barrier=input_ready)
                        T.tma_copy(V[0, start:start + C, h, :], vs, barrier=input_ready)
                        T.tma_copy(DO[0, start:start + C, h, :], ds, barrier=input_ready)
                        T.tma_copy(A[0, start:start + C, h, :], a, barrier=input_ready)
                        T.tma_copy(H[0, start // C, h, :, :], hs, barrier=input_ready)
                        T.barrier_arrive(input_ready)
                    T.copy(B[0, start:start + C, h], beta)
                    for i in T.Parallel(C):
                        ex[i] = T.exp2(G[0, start + i, h] * 1.4426950408889634)
                        invex[i] = T.exp2(-G[0, start + i, h] * 1.4426950408889634)
                    T.barrier_wait(input_ready, c % 2)
                    T.barrier_arrive(scalars_ready)
                    T.sync_threads(2, 128)
                    T.wgmma_gemm(qs, ks, qk, transpose_B=True, clear_accum=True)
                    T.wait_wgmma(0)
                    for i, j in T.Parallel(C, C):
                        ps[i, j] = T.if_then_else(i >= j, qk[i, j], 0.0)
                    T.fence_proxy_async()
                    T.barrier_arrive(ps_ready)
    return kernel


@lru_cache(maxsize=32)
def _dense_cu(device, tokens):
    return torch.tensor([0, tokens], dtype=torch.int32, device=device)


def implicit_gdr_bwd(q, k, v, a, g, b, do, dht, h, scale=None,
                     cu_seqlens=None, chunk_size=64, state_v_first=False):
    """Drop-in main backward kernel; boundaries must be 64-token aligned.

    Returns expanded-head dq/dk and gradients w.r.t. cumulative gates, matching
    fused_gdr_bwd. State reconstruction, CP correction, grouped-head reduction,
    normalization, and reverse gate-prefix sums remain in the existing driver.
    This prototype does not replace the public API's default backward.
    """
    assert q.shape == k.shape and q.shape[0] == 1
    tokens, hk, hv = q.shape[1], q.shape[2], v.shape[2]
    assert q.shape[-1] == v.shape[-1] == 128 and tokens % 64 == 0
    assert chunk_size == 64 and hv % hk == 0
    assert all(t.dtype == torch.bfloat16 for t in (q, k, v, a, do, h))
    assert g.dtype == b.dtype == torch.float32
    assert all(t.is_cuda and t.is_contiguous() and t.device == q.device
               for t in (q, k, v, a, g, b, do, h))
    assert h.shape == (1, tokens // 64, hv, 128, 128)
    cu = _dense_cu(q.device, tokens) if cu_seqlens is None else cu_seqlens
    parts = cu.numel() - 1
    assert cu.dtype in (torch.int32, torch.int64) and cu.device == q.device
    use_dht = dht is not None
    if dht is None:
        dht = torch.empty((parts, hv, 128, 128), dtype=torch.float32, device=q.device)
    assert dht.shape == (parts, hv, 128, 128) and dht.dtype == torch.float32
    return _pipelined_bwd(tokens, hk, hv, parts, 128**-0.5 if scale is None else scale,
                         state_v_first, use_dht, str(cu.dtype).split('.')[-1])(
        q, k, v, a, g, b, do, h, dht, cu)

# Benchmark-compatible main-kernel interface; not selected by the public API.
fused_gdr_bwd = implicit_gdr_bwd

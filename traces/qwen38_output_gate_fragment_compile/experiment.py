"""GDN output recurrence with register-held V-first state and TMA pipelines.

AB is inv(I + diag(beta) strict_lower(K K.T)) diag(beta).
With chunk-local Gamma = diag(exp(g)), use U.T =
(V.T Gamma^-1 - S.T K.T) AB.T. Then O.T =
scale (S.T Q.T + U.T causal(Q K.T).T) Gamma and
S_new.T = gamma_last (S.T + U.T K). This cancels the pairwise gate ratios.
"""

import torch
import tilelang
import tilelang.language as T


@tilelang.jit(out_idx=[8, 9], pass_configs={
    tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    tilelang.PassConfigKey.TL_DISABLE_THREAD_STORAGE_SYNC: True,
    tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
})
def _output_fwd(tokens, hk, hv, partition_size, scale, h0_dtype, state_v_first, final, gate_stride, norm_eps):
    C, D = 64, 128
    parts = tokens // partition_size

    @T.macro
    def gemm(lhs, rhs, acc, transpose_A=False, transpose_B=False,
             clear_accum=False, policy=T.GemmWarpPolicy.FullRow):
        tx = T.get_thread_binding()
        if tx < 128:
            T.sync_threads(8, 256)
        else:
            T.sync_threads(9, 256)
        T.wgmma_gemm(lhs, rhs, acc, transpose_A=transpose_A,
                     transpose_B=transpose_B, clear_accum=clear_accum, policy=policy)
        if tx < 128:
            T.named_barrier_arrive(9, 256)
        else:
            T.named_barrier_arrive(8, 256)
        T.wait_wgmma(0)

    @T.macro
    def project_state(state, keys, queries, sk, sq):
        # Queue both independent projections before one handoff and one wait.
        tx = T.get_thread_binding()
        if tx < 128:
            T.sync_threads(8, 256)
        else:
            T.sync_threads(9, 256)
        T.wgmma_gemm(state, keys, sk, transpose_B=True, clear_accum=True,
                     policy=T.GemmWarpPolicy.FullRow)
        T.wgmma_gemm(state, queries, sq, transpose_B=True, clear_accum=True,
                     policy=T.GemmWarpPolicy.FullRow)
        if tx < 128:
            T.named_barrier_arrive(9, 256)
        else:
            T.named_barrier_arrive(8, 256)
        T.wait_wgmma(0)

    @T.prim_func
    def kernel(Q: T.Tensor((1, tokens, hk, D), "bfloat16"),
               K: T.Tensor((1, tokens, hk, D), "bfloat16"),
               V: T.Tensor((1, tokens, hv, D), "bfloat16"),
               AB: T.Tensor((1, tokens, hv, C), "bfloat16"),
               G: T.Tensor((1, tokens, hv), "float32"),
               H0: T.Tensor((parts, hv, D, D), h0_dtype),
               Z: T.StridedTensor((tokens, hv, D), (gate_stride, D, 1), "bfloat16"),
               NW: T.Tensor((D,), "bfloat16"),
               O: T.Tensor((1, tokens, hv, D), "bfloat16"),
               HT: T.Tensor((1, hv, D, D), "float32")):
        with T.Kernel(parts * hv, threads=512) as (block,):
            p, h = block // hv, block % hv
            qs = T.alloc_shared((2, C, D), "bfloat16")
            ks = T.alloc_shared((2, C, D), "bfloat16")
            vs = T.alloc_shared((2, C, D), "bfloat16")
            ab = T.alloc_shared((2, C, C), "bfloat16")
            gs = T.alloc_shared((2, C), "float32")
            ex = T.alloc_shared((2, C), "float32")
            invex = T.alloc_shared((2, C), "float32")
            ps = T.alloc_shared((2, C, C), "bfloat16")
            out = T.alloc_shared((2, C, D), "bfloat16")
            T.annotate_layout({vs: tilelang.layout.make_swizzled_layout(vs),
                               out: tilelang.layout.make_swizzled_layout(out)})
            ready = T.alloc_barrier(arrive_count=[96, 96])
            free = T.alloc_barrier(arrive_count=[384, 384])
            qk_ready = T.alloc_barrier(arrive_count=[128, 128])
            qk_free = T.alloc_barrier(arrive_count=[256, 256])
            out_ready = T.alloc_barrier(arrive_count=[256, 256])
            out_free = T.alloc_barrier(arrive_count=[32, 32])
            st = T.alloc_fragment((D, D), "float32")
            sop = T.alloc_fragment((D, D), "bfloat16")
            wf = T.alloc_fragment((D, C), "float32")
            wop = T.alloc_fragment((D, C), "bfloat16")
            uf = T.alloc_fragment((D, C), "float32")
            uop = T.alloc_fragment((D, C), "bfloat16")
            of = T.alloc_fragment((D, C), "float32")
            ob = T.alloc_fragment((C, D), "bfloat16")
            norms = T.alloc_fragment((C,), "float32")
            normalized = T.alloc_local((1,), "bfloat16")
            silu = T.alloc_local((1,), "bfloat16")
            gate_value = T.alloc_local((1,), "float32")
            tx = T.get_thread_binding()

            if tx < 256:
                T.set_max_nreg(208, 1)
                if tx >= 128:
                    T.named_barrier_arrive(8, 256)
                for i, j in T.Parallel(D, D):
                    if state_v_first:
                        st[i, j] = H0[p, h, i, j]
                    else:
                        st[i, j] = H0[p, h, j, i]
                for c in T.serial(partition_size // C):
                    s = c % 2
                    start = p * partition_size + c * C
                    T.barrier_wait(ready[s], (c // 2) % 2)
                    T.copy(st, sop)
                    project_state(sop, ks[s, :, :], qs[s, :, :], wf, of)
                    for i, j in T.Parallel(D, C):
                        wop[i, j] = vs[s, j, i] * invex[s, j] - wf[i, j]
                    gemm(wop, ab[s, :, :], uf, transpose_B=True, clear_accum=True)
                    T.copy(uf, uop)
                    T.barrier_wait(qk_ready[s], (c // 2) % 2)
                    gemm(uop, ps[s, :, :], of)
                    T.barrier_arrive(qk_free[s])
                    T.barrier_wait(out_free[s], (c // 2 + 1) % 2)
                    # Transpose once for coalesced output writes.
                    for i, j in T.Parallel(C, D):
                        ob[i, j] = of[j, i] * (scale * ex[s, i])
                    for i, j in T.Parallel(D, C):
                        wf[i, j] = T.cast(ob[j, i], "float32") * T.cast(ob[j, i], "float32")
                    T.reduce_sum(wf, norms, dim=0)
                    for i, j in T.Parallel(C, D):
                        normalized[0] = T.cast(ob[i, j], "float32") * T.rsqrt(norms[i] / D + norm_eps) * T.cast(NW[j], "float32")
                        gate_value[0] = T.cast(Z[start + i, h, j], "float32")
                        silu[0] = gate_value[0] / (1.0 + T.exp(-gate_value[0]))
                        ob[i, j] = T.cast(normalized[0], "float32") * T.cast(silu[0], "float32")
                    T.copy(ob, out[s, :, :])
                    T.fence_proxy_async()
                    T.barrier_arrive(out_ready[s])
                    gemm(uop, ks[s, :, :], st)
                    for i, j in T.Parallel(D, D):
                        st[i, j] *= ex[s, C - 1]
                    T.barrier_arrive(free[s])
                if final and p == parts - 1:
                    for i, j in T.Parallel(D, D):
                        if state_v_first:
                            HT[0, h, i, j] = st[i, j]
                        else:
                            HT[0, h, i, j] = st[j, i]
            elif tx < 384:
                T.set_max_nreg(24, 0)
                if tx >= 288 and tx < 320:
                    # Output stores progress independently of gate preparation.
                    for c in T.serial(partition_size // C):
                        s = c % 2
                        start = p * partition_size + c * C
                        T.barrier_wait(out_ready[s], (c // 2) % 2)
                        T.tma_copy(out[s, :, :], O[0, start:start + C, h, :])
                        T.tma_store_wait(0, read=True)
                        T.barrier_arrive(out_free[s])
                    T.tma_store_wait(0, read=False)
                else:
                    for c in T.serial(partition_size // C):
                        s = c % 2
                        start = p * partition_size + c * C
                        T.barrier_wait(free[s], (c // 2 + 1) % 2)
                        if tx < 288:
                            T.tma_copy(Q[0, start:start + C, h // (hv // hk), :], qs[s, :, :], barrier=ready[s])
                            T.tma_copy(K[0, start:start + C, h // (hv // hk), :], ks[s, :, :], barrier=ready[s])
                            T.tma_copy(V[0, start:start + C, h, :], vs[s, :, :], barrier=ready[s])
                        elif tx < 352:
                            T.tma_copy(AB[0, start:start + C, h, :], ab[s, :, :], barrier=ready[s])
                        else:
                            T.copy(G[0, start:start + C, h], gs[s, :])
                            T.sync_threads(7, 32)
                            for i in T.Parallel(C):
                                ex[s, i] = T.exp2(gs[s, i] * 1.4426950408889634)
                                invex[s, i] = T.exp2(-gs[s, i] * 1.4426950408889634)
                            T.fence_proxy_async()
                        T.barrier_arrive(ready[s])
            else:
                # QK is independent of the incoming state. Compute it ahead
                # of the recurrence in a separate warp group.
                T.set_max_nreg(64, 0)
                pf = T.alloc_fragment((C, C), "float32")
                pb = T.alloc_fragment((C, C), "bfloat16")
                for c in T.serial(partition_size // C):
                    s = c % 2
                    T.barrier_wait(ready[s], (c // 2) % 2)
                    T.wgmma_gemm(ks[s, :, :], qs[s, :, :], pf,
                                 transpose_B=True, clear_accum=True,
                                 policy=T.GemmWarpPolicy.Square)
                    T.wait_wgmma(0)
                    T.barrier_arrive(free[s])
                    T.barrier_wait(qk_free[s], (c // 2 + 1) % 2)
                    for i, j in T.Parallel(C, C):
                        pb[i, j] = T.if_then_else(i <= j, pf[i, j], 0.0)
                    T.copy(pb, ps[s, :, :])
                    T.fence_proxy_async()
                    T.barrier_arrive(qk_ready[s])
    return kernel


def output_fwd(q, k, v, ab, g, initial_states, partition_size, *, scale=None,
               state_v_first=False, output_final_state=False, gate=None, norm_weight=None, norm_eps=1e-6):
    assert q.shape == k.shape and k.shape[0] == 1 and k.shape[-1] == v.shape[-1] == 128
    tokens, hk, hv = k.shape[1], k.shape[2], v.shape[2]
    assert tokens % partition_size == 0 and partition_size % 64 == 0
    assert initial_states.shape == (tokens // partition_size, hv, 128, 128)
    assert all(x.is_cuda and x.device == k.device and x.is_contiguous()
               for x in (q, k, v, ab, g, initial_states))
    assert q.dtype == k.dtype == v.dtype == ab.dtype == torch.bfloat16
    assert g.dtype == torch.float32 and initial_states.dtype in (torch.bfloat16, torch.float32)
    assert ab.shape == (1, tokens, hv, 64) and g.shape == (1, tokens, hv)
    assert gate.shape == (tokens, hv * 128) and gate.stride(-1) == 1
    assert gate.dtype == norm_weight.dtype == torch.bfloat16 and norm_weight.shape == (128,)
    o, ht = _output_fwd(tokens, hk, hv, partition_size, scale if scale is not None else 128**-0.5,
                       initial_states.dtype, state_v_first, output_final_state, gate.stride(0), norm_eps)(q, k, v, ab, g, initial_states, gate.view(tokens, hv, 128), norm_weight)
    return o, ht if output_final_state else None

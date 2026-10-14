"""Compile-time N-only suffix summaries, split into 64 value rows per CTA."""
import tilelang
import tilelang.language as T


@tilelang.jit(out_idx=[], pass_configs={
    tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    tilelang.PassConfigKey.TL_DISABLE_THREAD_STORAGE_SYNC: True,
    tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
})
def _prepare_n(tokens, hk, hv, partition_size, state_v_first, skip_last,
               out_dtype, beta_folded, has_transfer_mask):
    C, D = 64, 128
    parts = tokens // partition_size

    @T.macro
    def gemm(lhs, rhs, acc, transpose_A=False, transpose_B=False, clear=False):
        T.wgmma_gemm(lhs, rhs, acc, transpose_A=transpose_A,
                     transpose_B=transpose_B, clear_accum=clear,
                     policy=T.GemmWarpPolicy.FullRow)
        T.wait_wgmma(0)

    @T.macro
    def chunk(c, first, ks, vs, a, beta, decay, gamma, x,
              nt, operand, scratch, short_op, ready, free):
        T.barrier_wait(ready, c % 2)
        # Two small products reuse the same 64 x 64 scratch tile. Keeping
        # the complete 128 x 64 X accumulator would increase register pressure.
        for half in T.unroll(2):
            gemm(ks[:, half * C:(half + 1) * C], a, scratch,
                 transpose_A=True, clear=True)
            for i, j in T.Parallel(C, C):
                if beta_folded:
                    short_op[i, j] = -scratch[i, j]
                else:
                    short_op[i, j] = -scratch[i, j] * beta[j]
            T.copy(short_op, x[half * C:(half + 1) * C, :])
        T.sync_threads(6, 128)
        if first:
            for i, j in T.Parallel(C, C):
                short_op[i, j] = -vs[j, i] * decay[j]
        else:
            T.copy(nt, operand)
            gemm(operand, ks, scratch, transpose_B=True, clear=True)
            for i, j in T.Parallel(C, C):
                short_op[i, j] = gamma[0] * scratch[i, j] - vs[j, i] * decay[j]
            for i, j in T.Parallel(C, D):
                nt[i, j] *= gamma[0]
        gemm(short_op, x, nt, transpose_B=True)
        T.barrier_arrive(free)

    @T.prim_func
    def kernel(K: T.Tensor((1, tokens, hk, D), "bfloat16"),
               V: T.Tensor((1, tokens, hv, D), "bfloat16"),
               A: T.Tensor((1, tokens, hv, C), "bfloat16"),
               G: T.Tensor((1, tokens, hv), "float32"),
               Beta: T.Tensor((1, tokens, hv), "float32"),
               N: T.Tensor((parts, hv, D, D), out_dtype),
               M: T.Tensor((parts, hv, D, D), out_dtype),
               Counts: T.Tensor((parts, hv), "int32"),
               Mask: T.Tensor((parts, hv), "bool")):
        with T.Kernel((parts - int(skip_last)) * hv * 2, threads=256) as (block,):
            p, h, half = block // (hv * 2), (block // 2) % hv, block % 2
            ks = T.alloc_shared((C, D), "bfloat16")
            vs = T.alloc_shared((C, C), "bfloat16")
            a = T.alloc_shared((C, C), "bfloat16")
            gs = T.alloc_shared((C,), "float32")
            beta = T.alloc_shared((C,), "float32")
            decay = T.alloc_shared((C,), "float32")
            gamma = T.alloc_shared((1,), "float32")
            x = T.alloc_shared((D, C), "bfloat16")
            out = T.alloc_shared((C, D) if state_v_first else (D, C), out_dtype)
            T.annotate_layout({ks: tilelang.layout.make_swizzled_layout(ks),
                               vs: tilelang.layout.make_swizzled_layout(vs),
                               out: tilelang.layout.make_swizzled_layout(out)})
            ready = T.alloc_barrier(arrive_count=128)
            free = T.alloc_barrier(arrive_count=128)
            nt = T.alloc_fragment((C, D), "float32")
            operand = T.alloc_fragment((C, D), "bfloat16")
            scratch = T.alloc_fragment((C, C), "float32")
            short_op = T.alloc_fragment((C, C), "bfloat16")
            tx = T.get_thread_binding()
            active = T.alloc_var("bool")
            if has_transfer_mask:
                active = not Mask[p, h]
            else:
                active = Counts[p, h] < partition_size // C
            if active and tx < 128:
                T.set_max_nreg(144, 1)
                T.clear(nt)
                chunk(0, True, ks, vs, a, beta, decay, gamma, x,
                      nt, operand, scratch, short_op, ready, free)
                for c in T.serial(1, Counts[p, h]):
                    chunk(c, False, ks, vs, a, beta, decay, gamma, x,
                          nt, operand, scratch, short_op, ready, free)
                if state_v_first:
                    T.copy(nt, out)
                else:
                    for i, j in T.Parallel(D, C):
                        out[i, j] = nt[j, i]
                T.sync_threads(6, 128)
                T.fence_proxy_async()
                if state_v_first:
                    T.tma_copy(out, N[p, h, half * C:(half + 1) * C, :])
                else:
                    T.tma_copy(out, N[p, h, :, half * C:(half + 1) * C])
                T.tma_store_wait(0, read=False)
                # M is key-first regardless of N layout. Each value-half CTA
                # owns half of its zero rows, so the two writes do not overlap.
                for i, j in T.Parallel(C, D):
                    M[p, h, half * C + i, j] = 0.0
            elif active and tx >= 128:
                T.set_max_nreg(24, 0)
                for c in T.serial(Counts[p, h]):
                    start = p * partition_size + (partition_size // C - Counts[p, h] + c) * C
                    T.barrier_wait(free, (c + 1) % 2)
                    if tx < 160:
                        T.tma_copy(K[0, start:start + C, h // (hv // hk), :], ks, barrier=ready)
                    elif tx < 192:
                        T.tma_copy(V[0, start:start + C, h, half * C:(half + 1) * C], vs, barrier=ready)
                        T.tma_copy(A[0, start:start + C, h, :], a, barrier=ready)
                    else:
                        T.copy(G[0, start:start + C, h], gs)
                        if not beta_folded:
                            T.copy(Beta[0, start:start + C, h], beta)
                        T.sync_threads(7, 64)
                        for i in T.Parallel(C):
                            decay[i] = T.exp2((gs[C - 1] - gs[i]) * 1.4426950408889634)
                        if tx == 192:
                            gamma[0] = T.exp2(gs[C - 1] * 1.4426950408889634)
                        T.fence_proxy_async()
                    T.barrier_arrive(ready)
    return kernel

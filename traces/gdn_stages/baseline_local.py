"""Warp-shuffle diagonal inversion and tensor-core blockwise composition."""
import torch
import tilelang
import tilelang.language as T


@tilelang.jit(out_idx=[3, 4], pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True})
def _local_inverse(tokens, hk, hv, folded, compute_g=False):
    C, D = 64, 128

    @T.macro
    def offdiag8(block, inverse, raw, dpad, cpad, apad, tmp):
        base = block * 16
        acc = T.alloc_fragment((16, 16), "float32")
        for i, j in T.Parallel(16, 16):
            if i < 8 and j < 8:
                dpad[block, i, j] = inverse[base + 8 + i, base + 8 + j]
                cpad[block, i, j] = raw[base + 8 + i, base + j]
                apad[block, i, j] = inverse[base + i, base + j]
            else:
                dpad[block, i, j] = 0.0
                cpad[block, i, j] = 0.0
                apad[block, i, j] = 0.0
        T.gemm(dpad[block, :, :], cpad[block, :, :], acc, clear_accum=True)
        T.copy(acc, tmp[block, :, :])
        T.gemm(tmp[block, :, :], apad[block, :, :], acc, clear_accum=True)
        for i, j in T.Parallel(16, 16):
            if i < 8 and j < 8:
                inverse[base + 8 + i, base + j] = -acc[i, j]

    @T.macro
    def offdiag16(block, inverse, raw, tmp):
        lo, hi = block * 32, block * 32 + 16
        acc = T.alloc_fragment((16, 16), "float32")
        T.gemm(inverse[hi:hi + 16, hi:hi + 16], raw[hi:hi + 16, lo:lo + 16], acc, clear_accum=True)
        T.copy(acc, tmp[block, :, :])
        T.gemm(tmp[block, :, :], inverse[lo:lo + 16, lo:lo + 16], acc, clear_accum=True)
        for i, j in T.Parallel(16, 16):
            inverse[hi + i, lo + j] = -acc[i, j]

    @T.prim_func
    def kernel(K: T.Tensor((1, tokens, hk, D), "bfloat16"),
               Beta: T.Tensor((1, tokens, hv), "float32"),
               G: T.Tensor((1, tokens, hv), "float32"),
               A: T.Tensor((1, tokens, hv, C), "bfloat16"),
               GC: T.Tensor((1, tokens, hv), "float32")):
        with T.Kernel(tokens // C * hv, threads=128) as (block,):
            c, h = block // hv, block % hv
            ks = T.alloc_shared((C, D), "bfloat16")
            k_ready = T.alloc_barrier(arrive_count=32)
            inverse_ready = T.alloc_barrier(arrive_count=128)
            beta = T.alloc_shared((C,), "float32")
            raw = T.alloc_shared((C, C), "float16")
            inverse = T.alloc_shared((C, C), "float16")
            diagonal = T.alloc_shared((8, 8, 9), "float32")
            dpad = T.alloc_shared((4, 16, 16), "float16")
            cpad = T.alloc_shared((4, 16, 16), "float16")
            apad = T.alloc_shared((4, 16, 16), "float16")
            tmp8 = T.alloc_shared((4, 16, 16), "float16")
            tmp16 = T.alloc_shared((2, 16, 16), "float16")
            tmp32 = T.alloc_shared((32, 32), "float16")
            gram = T.alloc_fragment((C, C), "float32")
            rows = T.alloc_fragment((8, 8, 8), "float32")
            scale = T.alloc_fragment((8, 8), "float32")
            acc32 = T.alloc_fragment((32, 32), "float32")
            gf = T.alloc_fragment((1, C), "float32")
            T.annotate_layout({
                diagonal: tilelang.layout.make_linear_layout(diagonal),
                rows: tilelang.layout.Fragment((8, 8, 8),
                    forward_thread_fn=lambda b, i, j, r: b * 16 + i + r * 8,
                    forward_index_fn=lambda b, i, j: j, replicate=2),
                scale: tilelang.layout.Fragment((8, 8),
                    forward_thread_fn=lambda b, i, r: b * 16 + i + r * 8,
                    forward_index_fn=lambda b, i: 0, replicate=2),
            })
            tx = T.get_thread_binding()
            if tx < 32:
                T.tma_copy(K[0, c * C:(c + 1) * C, h // (hv // hk), :], ks,
                           barrier=k_ready)
                T.barrier_arrive(k_ready)
            T.copy(Beta[0, c * C:(c + 1) * C, h], beta)
            T.barrier_wait(k_ready, 0)
            T.gemm(ks, ks, gram, transpose_B=True, clear_accum=True)
            for i, j in T.Parallel(C, C):
                raw[i, j] = T.if_then_else(i > j, beta[i] * gram[i, j], 0.0)
                inverse[i, j] = 0.0
                if i // 8 == j // 8:
                    diagonal[i // 8, i % 8, j % 8] = T.if_then_else(
                        i == j, 1.0, T.if_then_else(i > j, beta[i] * gram[i, j], 0.0))
            for b, i, j in T.Parallel(8, 8, 8):
                rows[b, i, j] = diagonal[b, i, j]
            for src in T.unroll(7):
                for b, i in T.Parallel(8, 8):
                    scale[b, i] = -rows[b, i, src]
                for j in T.unroll(src):
                    for b, i in T.Parallel(8, 8):
                        value = T.shfl_sync(rows[b, i, j], src, width=8)
                        if i > src:
                            rows[b, i, j] += scale[b, i] * value
                for b, i in T.Parallel(8, 8):
                    if i > src:
                        rows[b, i, src] = scale[b, i]
            for b, i, j in T.Parallel(8, 8, 8):
                inverse[b * 8 + i, b * 8 + j] = rows[b, i, j]
            T.sync_threads()
            tx = T.get_thread_binding()
            if tx < 32:
                offdiag8(0, inverse, raw, dpad, cpad, apad, tmp8)
            elif tx < 64:
                offdiag8(1, inverse, raw, dpad, cpad, apad, tmp8)
            elif tx < 96:
                offdiag8(2, inverse, raw, dpad, cpad, apad, tmp8)
            else:
                offdiag8(3, inverse, raw, dpad, cpad, apad, tmp8)
            T.sync_threads()
            if tx < 32:
                offdiag16(0, inverse, raw, tmp16)
            elif tx < 64:
                offdiag16(1, inverse, raw, tmp16)
            # Release the two warp-local products to every warp consuming the
            # assembled diagonal blocks in the next tensor-core operation.
            T.barrier_arrive(inverse_ready)
            T.barrier_wait(inverse_ready, 0)
            T.gemm(inverse[32:, 32:], raw[32:, :32], acc32, clear_accum=True)
            # The shared-memory allocator reuses raw for tmp32. All warps must
            # finish loading raw before any warp writes the temporary product.
            T.sync_threads()
            T.copy(acc32, tmp32)
            T.gemm(tmp32, inverse[:32, :32], acc32, clear_accum=True)
            for i, j in T.Parallel(32, 32):
                inverse[32 + i, j] = -acc32[i, j]
            for i, j in T.Parallel(C, C):
                if folded:
                    A[0, c * C + i, h, j] = T.cast(inverse[i, j], "float32") * beta[j]
                else:
                    A[0, c * C + i, h, j] = inverse[i, j]
            if compute_g:
                for j in T.Parallel(C):
                    gf[0, j] = G[0, c * C + j, h]
                T.cumsum(gf, dim=1)
                for j in T.Parallel(C):
                    GC[0, c * C + j, h] = gf[0, j]
    return kernel


def local_inverse(k, beta, *, fold_beta=False):
    assert k.shape[0] == 1 and k.shape[-1] == 128 and k.shape[1] % 64 == 0
    assert beta.shape[:2] == k.shape[:2] and beta.shape[2] % k.shape[2] == 0
    assert k.dtype == torch.bfloat16 and beta.dtype == torch.float32
    assert k.is_cuda and beta.device == k.device and k.is_contiguous() and beta.is_contiguous()
    return _local_inverse(k.shape[1], k.shape[2], beta.shape[2], fold_beta)(k, beta, beta)[0]


def local_precompute(k, beta, g):
    """Return the beta-folded inverse and chunk-local cumulative log gates."""
    assert g.shape == beta.shape and g.dtype == beta.dtype == torch.float32
    assert g.is_cuda and g.device == k.device and g.is_contiguous()
    assert k.shape[0] == 1 and k.shape[-1] == 128 and k.shape[1] % 64 == 0
    assert beta.shape[:2] == k.shape[:2] and beta.shape[2] % k.shape[2] == 0
    assert k.dtype == torch.bfloat16 and k.is_contiguous() and beta.is_contiguous()
    return _local_inverse(k.shape[1], k.shape[2], beta.shape[2], True, True)(k, beta, g)

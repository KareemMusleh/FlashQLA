# Copyright (c) 2026 The Qwen team, Alibaba Group.
# Licensed under The MIT License [see LICENSE for details]
"""Partition summaries using register-operand WGMMA, optionally truncated.

For a K-first state, a partition represents S_out = M @ S_in + N.
Keep N.T and M.T in registers so they can be the *left* WGMMA operand.
This is an experimental, equal-partition forward kernel, not the default path.
"""

from functools import lru_cache

import torch
import tilelang
import tilelang.language as T


@tilelang.jit(out_idx=[5, 6], pass_configs={
    # Explicit producer, two math warpgroups, and named barriers below.
    tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    tilelang.PassConfigKey.TL_DISABLE_THREAD_STORAGE_SYNC: True,
    tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
})
def _prepare_mn(tokens, hk, hv, partition_size, state_v_first, skip_last, out_dtype,
                beta_folded=False, truncated=False, has_transfer_mask=False,
                stages=2, overlap=False):
    partitions = tokens // partition_size
    C, D = 64, 128

    # Alternate WGMMA issue between math groups. Two issue-order handoffs
    # make both groups' X stores visible before either consumes the full tile.
    @T.macro
    def ordered_gemm(lhs, rhs, accum, transpose_A=False, transpose_B=False,
                     clear_accum=False, policy=T.GemmWarpPolicy.FullRow, wait=True):
        tx = T.get_thread_binding()
        if tx < 128:
            T.sync_threads(8, 256)
        else:
            T.sync_threads(9, 256)
        T.wgmma_gemm(lhs, rhs, accum, transpose_A=transpose_A,
                     transpose_B=transpose_B, clear_accum=clear_accum, policy=policy)
        if tx < 128:
            T.named_barrier_arrive(9, 256)
        else:
            T.named_barrier_arrive(8, 256)
        if wait:
            T.wait_wgmma(0)

    @T.macro
    def compute_chunk(c, first, compute_m, ks, vs, a, beta, decay, gammas, x,
                      xf, xbf, nt, mt, operand, yz, yz_operand, ready):
        s = c % stages
        T.barrier_wait(ready[s], (c // stages) % 2)
        gamma = T.alloc_var("float32")
        gamma = gammas[s]
        # Store X.T = -K.T A diag(beta) using a matrix-store instruction.
        if overlap and not first:
            if compute_m:
                T.copy(mt, operand)
            else:
                T.copy(nt, operand)
            ordered_gemm(ks[s, :, :], a[s, :, :], xf, transpose_A=True,
                         clear_accum=True, wait=False)
            ordered_gemm(operand, ks[s, :, :], yz, transpose_B=True,
                         clear_accum=True)
        else:
            ordered_gemm(ks[s, :, :], a[s, :, :], xf, transpose_A=True, clear_accum=True)
        for i, j in T.Parallel(D, C):
            if beta_folded:
                xbf[i, j] = -xf[i, j]
            else:
                xbf[i, j] = -beta[s, j] * xf[i, j]
        T.copy(xbf, x)

        # FullRow assigns each warpgroup complete rows, preserving the
        # register layout when an accumulator becomes the next operand.
        # Accumulate the transfer without scalar decay: Mhat.T += Z.T X.
        # Apply exp(sum(g_last)) once at the final store, as in prepare_h.
        if compute_m:
            if first:
                # Identity transfer: Z.T is exactly K.T.
                for i, j in T.Parallel(D, C):
                    yz_operand[i, j] = ks[s, j, i]
                T.sync_threads(6, 256)
            else:
                if not overlap:
                    T.copy(mt, operand)
                    ordered_gemm(operand, ks[s, :, :], yz, transpose_B=True,
                                 clear_accum=True)
                T.copy(yz, yz_operand)
            ordered_gemm(yz_operand, x, mt, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
        # Y.T = gamma N.T K.T - V.T diag(exp(g_last - g)).
        if first:
            # Zero incoming state: Y.T is just the scaled value tile.
            for i, j in T.Parallel(D, C):
                yz_operand[i, j] = -vs[s, j, i] * decay[s, j]
        else:
            if not overlap or compute_m:
                T.copy(nt, operand)
                ordered_gemm(operand, ks[s, :, :], yz, transpose_B=True,
                             clear_accum=True)
            for i, j in T.Parallel(D, C):
                yz_operand[i, j] = gamma * yz[i, j] - vs[s, j, i] * decay[s, j]
            for i, j in T.Parallel(D, D):
                nt[i, j] *= gamma
        ordered_gemm(yz_operand, x, nt, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)

    @T.prim_func
    def kernel(
        K: T.Tensor((1, tokens, hk, D), "bfloat16"),
        V: T.Tensor((1, tokens, hv, D), "bfloat16"),
        A: T.Tensor((1, tokens, hv, C), "bfloat16"),
        G: T.Tensor((1, tokens, hv), "float32"),
        Beta: T.Tensor((1, tokens, hv), "float32"),
        Nout: T.Tensor((partitions, hv, D, D), out_dtype),
        Mout: T.Tensor((partitions, hv, D, D), out_dtype),
        NumChunks: T.Tensor((partitions, hv), "int32"),
        TransferMask: T.Tensor((partitions, hv), "bool"),
    ):
        with T.Kernel(partitions * hv, threads=384) as (block,):
            p, h = block // hv, block % hv
            ks = T.alloc_shared((stages, C, D), "bfloat16")
            vs = T.alloc_shared((stages, C, D), "bfloat16")
            a = T.alloc_shared((stages, C, C), "bfloat16")
            g = T.alloc_shared((stages, C,), "float32")
            beta = T.alloc_shared((stages, C,), "float32")
            x = T.alloc_shared((D, C), "bfloat16")
            out = T.alloc_shared((D, D), out_dtype)
            decay = T.alloc_shared((stages, C,), "float32")
            T.annotate_layout({vs: tilelang.layout.make_swizzled_layout(vs),
                               out: tilelang.layout.make_swizzled_layout(out)})
            gamma = T.alloc_var("float32")
            log_decay = T.alloc_var("float32")
            gammas = T.alloc_shared((stages,), "float32")
            # Producer publishes TMA inputs and coefficients; both math groups
            # release a stage only after all reads finish.
            ready = T.alloc_barrier(arrive_count=[128] * stages)
            free = T.alloc_barrier(arrive_count=[256] * stages)
            tx = T.get_thread_binding()
            iters = T.alloc_var("int32")
            if truncated:
                iters = T.if_then_else(skip_last and p == partitions - 1, 0, NumChunks[p, h])
            else:
                iters = T.if_then_else(skip_last and p == partitions - 1, 0, partition_size // C)
            compute_m = T.alloc_var("bool")
            if has_transfer_mask:
                compute_m = TransferMask[p, h]
            else:
                compute_m = not truncated or iters == partition_size // C
            xf = T.alloc_fragment((D, C), "float32")
            xbf = T.alloc_fragment((D, C), "bfloat16")
            nt = T.alloc_fragment((D, D), "float32")
            mt = T.alloc_fragment((D, D), "float32")
            operand = T.alloc_fragment((D, D), "bfloat16")
            yz = T.alloc_fragment((D, C), "float32")
            yz_operand = T.alloc_fragment((D, C), "bfloat16")

            if tx < 256:
                T.set_max_nreg(240, 1)
                if tx >= 128:
                    T.named_barrier_arrive(8, 256)
                T.clear(nt)
                log_decay = 0.0
                for i, j in T.Parallel(D, D):
                    mt[i, j] = T.if_then_else(i == j, 1.0, 0.0)

                if iters > 0:
                    compute_chunk(0, True, compute_m, ks, vs, a, beta, decay, gammas, x,
                                  xf, xbf, nt, mt, operand, yz, yz_operand, ready)
                    log_decay += g[0, C - 1]
                    T.barrier_arrive(free[0])
                for c in T.serial(1, iters):
                    compute_chunk(c, False, compute_m, ks, vs, a, beta, decay, gammas, x,
                                  xf, xbf, nt, mt, operand, yz, yz_operand, ready)
                    log_decay += g[c % stages, C - 1]
                    T.barrier_arrive(free[c % stages])

                for i, j in T.Parallel(D, D):
                    if state_v_first:
                        out[i, j] = nt[i, j]
                    else:
                        out[i, j] = nt[j, i]
                T.sync_threads(6, 256)
                T.fence_proxy_async()
                T.tma_copy(out, Nout[p, h, :, :])
                T.tma_store_wait(0, read=True)
                T.sync_threads(6, 256)
                gamma = T.exp2(log_decay * 1.4426950408889634)
                for i, j in T.Parallel(D, D):
                    if not compute_m and iters > 0:
                        # The suffix started from zero; drop the incoming state.
                        out[i, j] = 0.0
                    else:
                        out[i, j] = gamma * mt[j, i]
                T.sync_threads(6, 256)
                T.fence_proxy_async()
                T.tma_copy(out, Mout[p, h, :, :])
                T.tma_store_wait(0, read=False)

            else:
                T.set_max_nreg(24, 0)
                for c in T.serial(iters):
                    s = c % stages
                    if truncated:
                        start = p * partition_size + (partition_size // C - iters + c) * C
                    else:
                        start = p * partition_size + c * C
                    T.barrier_wait(free[s], (c // stages + 1) % 2)
                    if tx < 288:
                        T.tma_copy(K[0, start:start + C, h // (hv // hk), :],
                                   ks[s, :, :], barrier=ready[s])
                    elif tx < 320:
                        T.tma_copy(V[0, start:start + C, h, :], vs[s, :, :], barrier=ready[s])
                        T.tma_copy(A[0, start:start + C, h, :], a[s, :, :], barrier=ready[s])
                    else:
                        T.copy(G[0, start:start + C, h], g[s, :])
                        if not beta_folded:
                            T.copy(Beta[0, start:start + C, h], beta[s, :])
                        T.sync_threads(7, 64)
                        for i in T.Parallel(C):
                            decay[s, i] = T.exp2((g[s, C - 1] - g[s, i]) * 1.4426950408889634)
                        if tx == 320:
                            gammas[s] = T.exp2(g[s, C - 1] * 1.4426950408889634)
                        T.fence_proxy_async()
                    T.barrier_arrive(ready[s])

    return kernel


@lru_cache(maxsize=32)
def _full_chunk_counts(device, parts, heads, chunks):
    # Cached input for the untruncated specialization, which eliminates its reads.
    return (torch.full((parts, heads), chunks, device=device, dtype=torch.int32),
            torch.ones((parts, heads), device=device, dtype=torch.bool))


def prepare_mn(k, v, a, g, beta, partition_size, *, state_v_first=False,
               skip_last=False, output_dtype=torch.bfloat16, beta_folded=False,
               num_warmup_chunks=None, transfer_mask=None,
               pipeline_stages=2, overlap_gemm=False):
    """Return (N, M) for equal, 64-token-aligned partitions.

    num_warmup_chunks optionally selects the retained suffix length per head,
    in 64-token chunks. Shortened partitions return zero M: the incoming state
    and omitted prefix are deliberately discarded. Counts must be in [1,
    partition_size // 64]; skip_last still returns (zero, identity).
    transfer_mask optionally selects heads that need M. False skips its GEMMs
    and returns zero M even if all chunks are retained (cutoff at the oldest
    chunk). This is the boundary scan mask returned by get_warmup_chunks.

    Inputs use FlashQLA's existing kkt_solve output A and chunk-local cumulative
    log gates G. Keys must already be normalized if the model requires it.
    With beta_folded=True, A already includes right multiplication by diag(beta).
    skip_last returns (zero, identity) for the final partition and is valid only
    when that partition's outgoing state is not needed.
    M always uses K-first layout; N follows state_v_first. No initial state is
    consumed: apply M to it in the boundary scan. This function has no autograd.
    """
    assert k.ndim == v.ndim == 4 and k.shape[0] == v.shape[0] == 1
    _, tokens, hk, d = k.shape
    hv = v.shape[2]
    assert d == v.shape[-1] == 128 and hv % hk == 0
    assert partition_size > 0 and partition_size % 64 == 0
    assert tokens % partition_size == 0
    assert v.shape[1] == tokens and a.shape == (1, tokens, hv, 64)
    assert g.shape == beta.shape == (1, tokens, hv)
    assert k.dtype == v.dtype == a.dtype == torch.bfloat16
    assert g.dtype == beta.dtype == torch.float32
    assert output_dtype in (torch.bfloat16, torch.float32)
    assert all(t.is_cuda and t.is_contiguous() and t.device == k.device
               for t in (k, v, a, g, beta))
    truncated = num_warmup_chunks is not None
    has_transfer_mask = transfer_mask is not None
    if has_transfer_mask:
        assert truncated
        assert transfer_mask.shape == (tokens // partition_size, hv)
        assert transfer_mask.dtype == torch.bool and transfer_mask.device == k.device
        assert transfer_mask.is_contiguous()
    if truncated:
        assert num_warmup_chunks.shape == (tokens // partition_size, hv)
        assert num_warmup_chunks.dtype == torch.int32
        assert num_warmup_chunks.device == k.device and num_warmup_chunks.is_contiguous()
    else:
        num_warmup_chunks, _ = _full_chunk_counts(k.device, tokens // partition_size,
                                                 hv, partition_size // 64)
    if not has_transfer_mask:
        _, transfer_mask = _full_chunk_counts(k.device, tokens // partition_size,
                                              hv, partition_size // 64)
    return _prepare_mn(tokens, hk, hv, partition_size, state_v_first, skip_last,
                       output_dtype, beta_folded, truncated, has_transfer_mask,
                       pipeline_stages, overlap_gemm)(
                           k, v, a, g, beta, num_warmup_chunks, transfer_mask)

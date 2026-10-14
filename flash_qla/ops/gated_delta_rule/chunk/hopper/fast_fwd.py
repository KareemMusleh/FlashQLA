"""Opt-in Hopper forward with optional decay truncation in partition preparation."""

from functools import lru_cache
import math
import torch

from flash_qla.utils.math import l2norm_fwd
from .cp_fwd import correct_initial_states, get_warmup_chunks
from .local_precompute import local_precompute
from .prepare_mn import prepare_mn
from .output_fwd import output_fwd


@lru_cache(maxsize=32)
def _scan_metadata(device, parts, heads):
    return (torch.ones((parts, heads), device=device, dtype=torch.bool),
            torch.tensor([0, parts], device=device, dtype=torch.int32))


@lru_cache(maxsize=32)
def _truncation_metadata(device, parts, length):
    return (torch.arange(parts + 1, device=device, dtype=torch.int32) * length,
            torch.zeros(parts, device=device, dtype=torch.bool))


@lru_cache(maxsize=32)
def _training_cp_context(device, tokens, parts):
    """Match the optimized equal partitions without CPU reads during graph capture."""
    from ..cp.context import FlashQLACPContext
    boundaries = torch.arange(parts + 1, device=device, dtype=torch.int32) * (tokens // parts)
    return FlashQLACPContext(
        is_intra=True,
        cu_seqlens=torch.tensor([0, tokens], device=device, dtype=torch.int32),
        intra_cp_cu_seqlens=boundaries,
        seq_map_r2c=torch.tensor([0, parts], device=device, dtype=torch.int32),
        seq_map_c2r=torch.zeros(parts, device=device, dtype=torch.int32),
        ht_mask=torch.arange(parts, device=device) == parts - 1,
        ht_mask_bwd=torch.arange(parts, device=device) == 0,
    )


def chunk_gated_delta_rule_hopper(q, k, v, g, beta, *, partitions=8, scale=None,
                                 initial_state=None, output_final_state=False,
                                 use_qk_l2norm_in_kernel=False, state_v_first=False,
                                 decay_threshold=None, return_intermediates=False,
                                 return_cp_cache=False):
    """Forward-only BF16 GDN; B=1, D=128 and equal partitions.

    decay_threshold=None retains full history. A threshold in (0, 1) retains
    only the newest whole chunks whose cumulative decay reaches that threshold
    in partition preparation; the output recurrence still processes all tokens.

    Returns (output, final_state). Inputs follow the ordinary FlashQLA API.
    The local inverse uses FP16 intermediate operands and folds beta once.
    The public API supplies autograd using the standard FlashQLA backward.
    This low-level wrapper itself is forward-only.
    """
    if decay_threshold is not None and not 0 < decay_threshold < 1:
        raise ValueError("decay_threshold must be None or a number in (0, 1)")
    if return_cp_cache and (not return_intermediates or partitions == 1 or decay_threshold is not None):
        raise ValueError("Partition caching requires saved intermediates, multiple partitions, and full history")
    tensors = (q, k, v, g, beta) + ((initial_state,) if initial_state is not None else ())
    if torch.is_grad_enabled() and any(x.requires_grad for x in tensors):
        raise ValueError("This low-level Hopper wrapper is forward-only; use chunk_gated_delta_rule with fwd_backend='optimized' for autograd")
    assert q.shape == k.shape and k.shape[0] == 1 and k.shape[-1] == v.shape[-1] == 128
    assert partitions > 0 and k.shape[1] % (partitions * 64) == 0
    assert torch.cuda.get_device_capability(k.device)[0] == 9
    if use_qk_l2norm_in_kernel:
        q, k = l2norm_fwd(q)[0], l2norm_fwd(k)[0]
    length, heads = k.shape[1] // partitions, v.shape[2]
    if initial_state is not None:
        assert initial_state.shape == (1, heads, 128, 128)
    if return_intermediates:
        ab, gc, ar = local_precompute(k, beta, g, return_inverse=True)
    else:
        ab, gc = local_precompute(k, beta, g)
    if partitions == 1:
        h0 = initial_state if initial_state is not None else torch.zeros(
            (1, heads, 128, 128), device=k.device, dtype=torch.float32)
    else:
        mask, seqmap = _scan_metadata(k.device, partitions, heads)
        num_chunks = None
        if decay_threshold is not None:
            cu, skip = _truncation_metadata(k.device, partitions, length)
            num_chunks, mask = get_warmup_chunks(
                gc, cu, skip, threshold=math.log(decay_threshold))
        truncation_args = ({"num_warmup_chunks": num_chunks, "transfer_mask": mask}
                           if num_chunks is not None else {})
        n, m = prepare_mn(k, v, ab, gc, beta, length, state_v_first=state_v_first,
                          skip_last=not return_cp_cache, output_dtype=torch.bfloat16, beta_folded=True,
                          **truncation_args)
        h0 = correct_initial_states(initial_state, n, m, mask, seqmap,
                                     state_v_first=state_v_first)
    o, ht = output_fwd(q, k, v, ab, gc, h0, length, scale=scale,
                      state_v_first=state_v_first, output_final_state=output_final_state)
    if return_cp_cache:
        from .cp_fwd import get_warmup_chunks_bidi
        from ..cp.preprocess import CPCache
        context = _training_cp_context(k.device, k.shape[1], partitions)
        _, num_bwd, _, fallback_bwd = get_warmup_chunks_bidi(
            gc, context.intra_cp_cu_seqlens, context.ht_mask, context.ht_mask_bwd,
            seq_map_r2c=context.seq_map_r2c,
        )
        cache = CPCache(h0=h0, mt=m, fallback_bwd=fallback_bwd, num_warmup_bwd=num_bwd)
        return o, ht, gc, ar, cache
    return (o, ht, gc, ar) if return_intermediates else (o, ht)

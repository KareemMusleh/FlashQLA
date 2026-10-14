"""Separate input projections with a combined, recomputing backward.

Weights remain independent parameters. Packing is temporary backward workspace,
so optimizer updates need no synchronization with a cached packed weight.
"""

import torch
import torch.nn.functional as F


class _CombinedInputProjection(torch.autograd.Function):
    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda")
    def forward(ctx, x, combine_weight_gradients, *weights):
        ctx.save_for_backward(x, *weights)
        ctx.combine_weight_gradients = combine_weight_gradients
        ctx.set_materialize_grads(False)
        return tuple(F.linear(x, weight) for weight in weights)

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx, *grad_outputs):
        x, *weights = ctx.saved_tensors
        if all(g is None for g in grad_outputs):
            return (None,) * (len(weights) + 2)
        # No packed activation or weight is retained between forward/backward.
        rows = x.numel() // x.shape[-1]
        gradients = torch.cat([
            g.reshape(rows, w.shape[0]) if g is not None
            else x.new_zeros((rows, w.shape[0]))
            for g, w in zip(grad_outputs, weights)
        ], dim=-1)
        dx = None
        if ctx.needs_input_grad[0]:
            dx = (gradients @ torch.cat(weights, dim=0)).view_as(x)
        dw = [None] * len(weights)
        if any(ctx.needs_input_grad[2:]) and ctx.combine_weight_gradients:
            packed_dw = gradients.T @ x.reshape(rows, x.shape[-1])
            for i, part in enumerate(packed_dw.split([w.shape[0] for w in weights])):
                if ctx.needs_input_grad[i + 2] and grad_outputs[i] is not None:
                    dw[i] = part
        elif any(ctx.needs_input_grad[2:]):
            del gradients
            for i, g in enumerate(grad_outputs):
                if ctx.needs_input_grad[i + 2] and g is not None:
                    dw[i] = g.reshape(rows, weights[i].shape[0]).T @ x.reshape(rows, x.shape[-1])
        return dx, None, *dw


def combined_input_projection(x, *weights, combine_weight_gradients=True):
    """Bias-free projections; one input-gradient and one weight-gradient GEMM.

Forward returns separate contiguous outputs, matching independent F.linear
calls. Backward includes runtime gradient/weight packing. Saved tensors are
only the existing input and original weights. This opt-in implementation may
change low-precision input-gradient rounding by combining the reductions.
    Set combine_weight_gradients=False to retain separate weight-gradient GEMMs.
"""
    if x.ndim < 2 or not weights:
        raise ValueError("expected an input with at least two dimensions and weights")
    for weight in weights:
        if weight.ndim != 2 or weight.shape[1] != x.shape[-1]:
            raise ValueError("weights must have shape [output_features, input_features]")
        if weight.device != x.device or weight.dtype != x.dtype:
            raise ValueError("input and weights must have matching device and dtype")
    return _CombinedInputProjection.apply(x, combine_weight_gradients, *weights)

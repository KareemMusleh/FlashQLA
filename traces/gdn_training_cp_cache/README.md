# Training evaluation and partition-state reuse

The optimized public training path now reuses forward partition initial states
and transfer matrices when `bwd_backend="optimized"`, `enable_fwd_cp_cache=True`,
the automatic training context uses intra-CP, and forward decay truncation is
disabled. Unsupported inputs, the default backward, one-partition execution,
and truncated optimized forwards retain their previous behavior.

```python
output, final_state = chunk_gated_delta_rule(
    q, k, v, g, beta,
    fwd_backend="optimized", bwd_backend="optimized",
    fwd_partitions=8, fwd_decay_threshold=None,
    enable_fwd_cp_cache=True,
)
loss = output.float().square().mean()
loss.backward()
```

## What changes

Previously the optimized forward chose eight equal partitions, while the
backward's automatic CP policy chose six on the benchmark shape. Saving an
eight-partition cache and indexing it with the six-partition boundaries would
be incorrect. This path constructs matching equal forward/backward boundaries
and retains the corrected incoming states and transfer matrices. Static
metadata is cached without CPU tensor reads during CUDA graph capture.

Forward now computes the last partition's transfer matrix too. Its incoming
state was sufficient for forward, but reverse boundary correction may require
its outgoing transfer. Backward reuses the saved matrices and incoming states;
it still reconstructs per-chunk states and uses the existing FlashQLA backward
kernels. The existing backward warmup cutoff is unchanged. Full-history here
describes optimized forward, not removal of backward's CP cutoff.

The saved cache adds about 36 MiB for eight partitions and 48 value heads,
independent of sequence length: FP32 incoming states plus BF16 transfers and
small masks/counts. This is in addition to the inverse cache's 96 MiB at 16k
or 768 MiB at 128k. Set `enable_fwd_cp_cache=False` for inverse-only reuse,
or `bwd_backend="default"` to rebuild the original-precision inverse.

## Complete training comparison

H100, B=1, Hk=16, Hv=48, D=128, BF16, Q/K normalization, no initial/final state.
Public API timings include GDN forward and backward, including preparation,
state reconstruction and gradient postprocessing; they exclude the loss,
optimizer and other layers. Nine alternating-order CUDA-graph samples follow
50 alternating warmup rounds. Standard FlashQLA uses its normal training
policy and caches; it is not forced to disable its own decay truncation.
The ordinary training policy chooses six partitions, while partition reuse
matches the optimized forward's eight. These are comparisons of selectable
training modes, not isolated kernels with identical partition counts.

The weak-gate fixture uses `g=logsigmoid(randn)*0.001`; the stronger-gate fixture
uses `g=logsigmoid(randn)/16`. Both disable optional optimized-forward truncation.

| Gates / length | Standard FlashQLA | Optimized + inverse only | Optimized + partition reuse |
|---|---:|---:|---:|
| Weak / 16k | 3.034 ms | 3.124 ms | 2.667 ms |
| Weak / 128k | 20.384 ms | 21.133 ms | 19.776 ms |
| Strong / 16k | 2.379 ms | 2.465 ms | 2.424 ms |
| Strong / 128k | 17.425 ms | 18.236 ms | 18.265 ms |

For weak gates, partition reuse improves our previous inverse-only mode by
14.6% at 16k and 6.4% at 128k. Compared with standard FlashQLA, complete GDN
forward/backward is 12.1% faster at 16k and 3.0% faster at 128k. An independent
weak-gate run reproduces the direction (11.7% and 4.5% versus FlashQLA).
Strong gates already make boundary reconstruction inexpensive: the gain is
1.7% at 16k and absent at 128k, and FlashQLA remains faster on that fixture.
There is no universal training speedup.

Samples/checks: [weak repeat](weak_repeat.json), [weak first run](weak_decay.json),
[strong isolated run](isolated.json). Chrome traces use names such as
`16384_optimized_cached_trace.json` and `131072_flashqla_trace.json` in this
directory; the latest traces are from the weak repeat. Earlier files are
retained for investigation. `matched.json` has anomalous 16k timings near
5.6 ms and is excluded from the table.

## Gradients and optimization trajectory

The benchmark checks both fixed upstream gradients and gradients from each
implementation's own scalar MSE loss. Compared with standard FlashQLA,
partition-cached fixed-upstream gradients differ by about 0.11–0.33% relative
RMS on the strong fixture and 0.41–0.60% on the weak fixture. With each output's
own MSE upstream gradient, the largest difference is approximately 0.87% on
the weak fixture. These are fixture measurements, not arbitrary-input bounds.
The default backward with optimized forward retains bitwise identical
fixed-upstream gradients on these fixtures.

A separate 50-step synthetic regression trains Q/K/V/g/beta and output
projections with AdamW. It uses T=256, Hk=2, Hv=4, D=128, hidden width 64,
FP32 parameters, BF16 GDN activations, and matching initialization/targets.
The weak-gate run reduces the loss about 71.4 times in all four modes; final
losses are 2.73671e-5 for FlashQLA and 2.73667e-5 for partition reuse.
The final parameter relative RMS difference is 0.081%. The strong-gate
50-step run reduces loss about 91.1 times in all modes.

This is a finite-gradient and optimization-trajectory smoke test. It excludes
convolution, output normalization/gating, language-model data, distributed
training and held-out evaluation. Full-model convergence remains untested.
The backward targets the recurrence, rather than differentiating the literal
FP16 inverse algorithm and its rounding operations.

## Validation and remaining work

62 forward, preparation and cached-backward tests pass. Independent FP64
autograd checks include initial/final-state gradients, both state layouts,
zero/unit beta, normalization and zero output scale. New tests verify matching
boundaries, one remaining chunk-state reconstruction, cache-disabled behavior
and truncated-forward fallback. Kernel-filtered racecheck passes eight cases
with zero hazards; it checks the optimized forward kernels and excludes legacy
backward kernels. Logs and hashes are in [validation.json](validation.json).

An independent-GEMM queueing experiment gives bitwise identical gradients
but no repeatable performance gain. It is archived under `../gdn_bwd_queued/`
and is not enabled in production. The main backward remains the largest
individual kernel and the next scheduling target. The separate implicit-solve
prototype still needs a better pipeline before it is competitive.

```sh
LD_PRELOAD=/usr/local/cuda/compat/lib/libcuda.so.1 CUDA_VISIBLE_DEVICES=6 \
  /opt/ydt/bin/python profile/bench_gdn_training.py --decay 0.001 --trace \
  --output traces/gdn_training_cp_cache/weak_repeat.json
```

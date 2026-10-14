# Backward investigation and full-layer training optimization

The main-backward scheduling experiments did not establish a safe, repeatable
speedup. They remain archived experiments and are not selected by the public
GDN API. Full-layer profiling identified a larger opportunity: output RMSNorm
and SiLU gating now have a fused training implementation with input and weight
gradients.

## Full-layer measurement

`profile/bench_gdn_full_layer_training.py` includes six input projections,
causal convolution and SiLU on Q/K/V, decay/beta preparation, Q/K normalization,
GDN, output RMSNorm/SiLU gating, and the output projection. Autograd computes
input gradients and gradients for every parameter, including convolution,
gate biases, decay parameters, normalization weights and projection weights.
The loss and optimizer are excluded. This is a synthetic GDN layer, not a
complete language-model convergence experiment.

Shapes: B=1, hidden=2560, Hk=16, Hv=48, D=128, convolution width=4, RMSNorm
epsilon=1e-6; BF16 projections/activations/normalization weights, FP32 decay
parameters. Synthetic initial dt=0.001. Both variants use identical standard
GDN forward, current GDN backward, eight partitions, saved intermediates and
truncation disabled. Only the output norm/gate implementation changes.

| Sequence | PyTorch norm/gate | Fused norm/gate training | Reduction in full forward/backward latency |
|---|---:|---:|---:|
| 16k | 14.603 ms | 13.425 ms | 8.1% |
| 128k | 119.789 ms | 107.793 ms | 10.0% |

CUDA graphs use 50 alternating warmups and nine alternating-order samples.
All input/parameter gradients are finite; the largest relative RMS difference
from the PyTorch full-layer path is 0.337% at 16k and 0.428% at 128k, on the key
projection weight. These are measured fixture differences, not accuracy bounds.
See [full-layer samples and gradient checks](full_layer_fused_gate/results.json)
and its forward/backward Chrome traces.

An independent repeat after CUDA recovered confirmed the improvement:

| Sequence | PyTorch norm/gate | Fused norm/gate training | Latency reduction |
|---|---:|---:|---:|
| 16k | 14.632 ms | 13.324 ms | 8.9% |
| 128k | 119.184 ms | 108.322 ms | 9.1% |

See [repeat samples and gradient checks](full_layer_fused_gate_repeat/results.json).
Full-layer output relative RMS differences were 0.00499% at 16k and 0.00422%
at 128k. The largest gradient differences remained 0.337% and 0.428%, on the
key projection weight. Samples show timing variation; the consistent result
across the two runs is roughly 8–10% lower full-layer latency.
Other GPU workloads finished during compilation, before the timing phase.

Chrome traces from the repeat:

- [16k separate operations](full_layer_fused_gate_repeat/16384_torch_norm_gate.json)
- [16k fused norm/gate](full_layer_fused_gate_repeat/16384_fused_norm_gate.json)
- [128k separate operations](full_layer_fused_gate_repeat/131072_torch_norm_gate.json)
- [128k fused norm/gate](full_layer_fused_gate_repeat/131072_fused_norm_gate.json)

The 128k fused trace measures approximately 2.71 ms for fused norm/gate forward
and 3.67 ms for its backward input/Z gradients and partial weight reduction.
The final weight reduction adds separate reduction/cast kernels. The separate
operations' forward/backward block was around 17 ms in the earlier full-layer
trace. Projections remain the largest block, roughly 59 ms in that trace.
Trace times and graph medians come from separate measurements and are not
interchangeable.

The initial full-layer comparison of committed GDN postprocessing versus the
current fused grouped normalization, with identical PyTorch output norm/gate,
measured 14.906 -> 14.496 ms at 16k and 120.822 -> 118.686 ms at 128k. See
[full_layer/results.json](full_layer/results.json). This demonstrates why an
operator-only GDN backward percentage is smaller at full-layer scope.

## Isolated norm/gate measurement

The standalone training-stage benchmark includes input, gate-input and
normalization-weight gradients, using unit-variance BF16 random activations
and random BF16 weights. It measures forward plus backward as:

| Sequence | Separate PyTorch operations | Fused implementation |
|---|---:|---:|
| 16k | 4.612 ms | 1.863 ms |
| 128k | 36.914 ms | 14.128 ms |

See [norm_gate_stage/results.json](norm_gate_stage/results.json) for separate
forward/backward samples and gradient checks. This fixture differs from the
full layer, whose GDN outputs have small amplitudes and whose norm weights
start at one. These medians must not be added to the full-layer timings.

## Implementation and use

`flash_qla/ops/gated_delta_rule/chunk/hopper/training_norm_gate.py` provides:

```python
from flash_qla.ops.gated_delta_rule.chunk.hopper.training_norm_gate import training_norm_gate

# output and z are matching contiguous BF16 tensors with last dimension 128;
# weight is the BF16 normalization weight vector, shape [128].
gated = training_norm_gate(output, z, weight, eps=1e-6)
```

Forward computes RMSNorm and SiLU in FP32, rounds their individual outputs to
BF16, then multiplies them. It saves the same normalized output, SiLU output
and FP32 norm statistics as the separate path, alongside the inputs/weight.
There is no additional forward cache used to obtain the speedup.

Backward rounds the two multiplication gradients to BF16 before computing
RMSNorm and SiLU derivatives. It writes input and Z gradients together and
forms FP32 weight-gradient partials, followed by a reduction/cast. Partials
are backward temporaries, not retained forward intermediates. FP32 weights
use the ordinary PyTorch operations to preserve mixed-dtype output semantics.
The fused path is an explicit callable; it does not replace model code or
change public GDN backward dispatch automatically.

## Correctness status

Nine GPU tests passed, covering PyTorch parity, non-multiple tile sizes,
64/128 dimensions, grouped tensor shapes, BF16/FP32 weights, zero weights,
a noncontiguous upstream gradient, small activations where epsilon matters,
and an independent FP64 formula with a large weight-gradient reduction.
Compute Sanitizer memcheck ran all nine tests and reported zero errors; see
[validation/memcheck.log](validation/memcheck.log). The full-layer benchmark
also checks output and every input/parameter gradient at 16k and 128k before
timing. These checks establish fixture correctness, not training convergence.

Fresh CUDA processes temporarily failed initialization after the first
measurements. CUDA subsequently recovered, allowing the remaining tests and
sanitizer checks to complete. Initialization failure logs are retained as
diagnostics, not kernel failures. The callable remains opt-in and supports
first-order gradients; higher-order gradients have not been implemented.

## dQ repeatability investigation

The original main kernel was tested at 128k, eight partitions, Hk=16/Hv=48,
with changing gradient-buffer allocations. Twelve repetitions were bitwise
stable both with an identity-inverse/arbitrary-state fixture and with a real
standard-forward inverse and reconstructed states. Results:
`repeat_original.json`, `repeat_real.json`.

A filtered Compute Sanitizer racecheck of the original main backward passed
a real-state, one-head, 16k/one-partition case (256 chunk iterations), with zero
hazards. See `racecheck.log`. This does not prove the initial 128k variation
cannot occur; its cause remains unresolved and no race fix is claimed.

## Main-kernel experiments

- `qk_gate_overlap.py`: queue QK WGMMA while forming the independent gate tile.
  Passed checks; did not outperform the original main kernel.
- `async_state_updates.py`: additionally publish state-update handoffs before
  waiting for their WGMMA completion. No repeatable gain was established.
- `final_store_wait.py`: explicitly finish TMA gradient stores at loop exit.
  No repeatability fix or performance improvement was established.
- `reuse_acc_implicit.py`: remove a full FP32 product temporary from the serial
  implicit-solve prototype. Passed the experimental gradient limit, but remains
  slower than the original main kernel (1.393 vs 1.067 ms in the 16k run).
- `pipelined_implicit.py`: a separate QK preparation warp group plus two math
  groups and reused accumulators. The initial register requests exceeded the
  effective CTA register budget and stalled; they were reduced. The resulting
  kernel ran but failed the dK accuracy limit (5.43% RMS versus a 4% limit).
  It remains an invalid experiment, not a performance candidate.

Some early runs were interrupted by other workloads exhausting GPU memory;
affected logs are retained and are not performance evidence. No external
workload was stopped. The stalled processes stopped during debugging were
our own experimental kernel processes.

## Reproduce

```sh
LD_PRELOAD=/usr/local/cuda/compat/lib/libcuda.so.1 CUDA_VISIBLE_DEVICES=6 \
  TILELANG_CACHE_DIR=/tmp/flashqla-tilelang-cache \
  TRITON_CACHE_DIR=/tmp/flashqla-triton-cache \
  /opt/ydt/bin/python profile/bench_gdn_full_layer_training.py \
  --compare-norm-gate --trace \
  --output traces/gdn_backward_pipeline/full_layer_fused_gate_repeat/results.json

LD_PRELOAD=/usr/local/cuda/compat/lib/libcuda.so.1 CUDA_VISIBLE_DEVICES=7 \
  /opt/ydt/bin/python -m pytest tests/test_training_norm_gate.py -q
```

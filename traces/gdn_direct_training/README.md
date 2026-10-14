# Direct projection-gradient writes for full GDN training

The layer now defaults to fused output norm/gate forward. Measurements in this
report describe the earlier eager output forward; use
`output_gate_mode="eager"` to reproduce that path. The follow-up comparison is
documented in [fused-forward results](../gdn_direct_fused_forward/README.md).

`flash_qla/ops/gdn_layer.py` provides an explicit experimental full-layer
autograd Function. It follows the YDT bias-free layer: six projections,
depthwise convolution/SiLU, decay/beta preparation, normalized gated delta
rule, output RMSNorm/SiLU gating, and output projection.

Backward allocates one BF16 `[tokens, projection_width]` buffer. Dao convolution
backward writes Q/K/V gradients directly into slices; fused norm/gate backward
writes Z; fused decay/beta backward writes A/B. Projection backward consumes
that buffer in two GEMMs. It does not concatenate projection gradients.
Separate projection parameters remain authoritative; only the small packed
weight is rebuilt during backward. No inverse or partition cache is retained.

The implementation is first-order only, Hopper, BF16 activations/weights,
FP32 decay parameters, one unpadded sequence, zero convolution history,
Dk=Dv=128, and no distributed context parallelism or state cache. Sequence
length must divide equally into `partitions` chunks of 64 tokens. It rejects
autocast; use explicitly typed inputs. YDT modules and existing defaults remain
unchanged.

```python
from flash_qla.ops.gdn_layer import direct_gdn_layer

y = direct_gdn_layer(
    x,
    (self.q_proj.weight, self.k_proj.weight, self.v_proj.weight,
     self.g_proj.weight, self.a_proj.weight, self.b_proj.weight),
    (self.q_conv1d.weight.squeeze(1), self.k_conv1d.weight.squeeze(1),
     self.v_conv1d.weight.squeeze(1)),
    self.a_log_bias, self.dt_bias, self.o_norm.weight, self.o_proj.weight,
    key_heads=self.config.num_k_heads, value_heads=self.config.num_v_heads,
    eps=1e-6, partitions=8,
)
```

Use the actual configured norm epsilon; the synthetic Qwen fixture uses 1e-6.
Output-projection bias is not supported by this entry point.

## Experiment and ablation

The benchmark compares the original eager autograd layer with two whole-layer
Functions that use identical forward operations and backward kernels:

- `packed`: separate gradient destinations followed by concatenation.
- `direct`: shared destinations with no gradient concatenation.

This isolates the buffer-layout contribution from output norm/gate and scalar
gate backward fusion. Both candidates use full-history optimized recurrence
forward and the existing default uncached recurrence backward.

The synthetic H100 fixture has hidden=2560, Hk=16, Hv=48, D=128, convolution
width=4. Fixed upstream gradients exercise the input and every parameter.
Timing uses 21 CUDA graph samples with alternating execution order and external
events around each variant. Optimizer updates, activation-stat logging, and
distributed communication are excluded. This compares eager layer training;
it does not establish a speedup over compiled full-layer training.

The fixture retains the existing dt_max=.01 setting rather than YDT's original
.1, whose optimized-forward limitation is documented in GDN_OPTIMIZATIONS.md.

Before the environment restart, measured complete forward/backward times were:

| Tokens | Original eager | Matched packing | Direct writes |
| --- | ---: | ---: | ---: |
| 16K | 14.246 ms | 13.323 ms | 12.781 ms |
| 128K | 113.952 ms | 109.579 ms | 104.880 ms |

At 16K the direct trace has 61 GPU kernels, versus 62 for matched packing and
87 for the original. The gradient-concatenation kernel disappears; weight
packing remains. Removing concatenation accounts for about 0.54 ms at 16K and
4.70 ms at 128K in these runs. Remaining improvement comes from the other
backward fusions and scheduling changes in the whole-layer Function.

## Memory and correctness

Unique saved activation storage is lower: 1697 to 1307 MiB at 16K, and 13576
to 10456 MiB at 128K. The existing FP32 softplus input and sigmoid output are
retained, while norm statistics and intermediate norm/gate activations are
recomputed. No raw beta projection, unfolded inverse, or boundary state is
added to the saved set.

There is one shared gradient buffer, not six separate buffers plus a packed
copy. It occupies 515 MiB at 16K and 4120 MiB at 128K for this fixture. Current
weights require 80.47 MiB of temporary packing. Eight-element token-pitch
padding supports small head counts; packed destination offsets use 64-bit
arithmetic because the 128K fixture exceeds signed 32-bit element indexing.

Eliminating concatenation does not guarantee lower total peak memory: the
shared buffer is allocated early for Z and remains live during recurrence
backward. Consumed saved tensors are released during ordinary backward;
retain_graph=True preserves them for later calls. Final native allocator event
history, with garbage collection before the allocation baseline, measures:

| Tokens | Original step peak | Matched packing | Direct step peak | Direct increase |
| --- | ---: | ---: | ---: | ---: |
| 16K | 2313 MiB | 2633 MiB | 2633 MiB | 320 MiB |
| 128K | 18042 MiB | 20602 MiB | 20602 MiB | 2560 MiB (2.5 GiB) |

These are additional live requested allocations during the layer step,
excluding resident input/parameters and validation reference tensors. Native
alloc/free-request events observe backward threads without changing operator
dispatch. Final event deltas differ from live allocation counters by only
896 bytes of small-allocation rounding. This measurement excludes optimizer
states and CUDA allocator reserved-but-unused memory.

The shared buffer avoids a second copy of all projection gradients, but its
Q/K/V slices are allocated before they are needed. That accounts for the extra
320 MiB at 16K / 2.5 GiB at 128K during recurrence backward. The matched packing
ablation also allocates all six destinations early, so its step peak is almost
identical. The next memory improvement would shorten this buffer's lifetime or
reuse its unused slices as recurrence workspace.

Full-layer outputs match the eager fixture exactly. Maximum gradient relative
RMS error is about 0.39%, dominated by the combined BF16 input-gradient
reduction. Tests cover grouped and ungrouped heads, repeated retained-graph
backward, FP64 norm/gate references, strided destinations and untouched padding,
gate extremes, and offsets beyond 2**31 elements.

Isolated timing measurements are `results.json` and `results128k.json`; their
exact benchmark and production sources are archived in
`bench_direct_gdn_training_measured.py` and `gdn_layer_measured.py`. The current
production source adds a positive-value-head validation guard, which does not
affect these positive-head-count fixtures.

`final_memory.json` contains the final output/gradient, saved-storage, native
allocation-history checks, and current source hashes for both lengths. Forward
outputs match exactly; maximum gradient RMS differences are 0.388% / 0.390%.
Twenty regular tests passed. Focused memcheck passed all six convolution,
norm/gate, gate, and long-offset tests with zero errors. The unfiltered
sanitizer run, including legacy TileLang recurrence kernels, failed and later
segfaulted; its log is retained. No unfiltered sanitizer success is claimed.

After the environment restart, an external job occupied every GPU. Later
timing runs were shared, and the 128K CUDA-graph capture ran out of available
memory. They are excluded from isolated speedup claims. `final16k.json` and
`final128k.json` are marked superseded; 16K Chrome traces are later shared-GPU
captures useful for kernel structure, not isolated latency. Intermediate peak
counter and operator-dispatch instrumentation was superseded by native
allocator event history; only `final_memory.json` is final memory evidence.

```sh
CUDA_VISIBLE_DEVICES=2 /opt/ydt/bin/python -m pytest -q tests/test_direct_gdn_layer.py tests/test_input_projection.py
CUDA_VISIBLE_DEVICES=0 /opt/ydt/bin/python profile/bench_direct_gdn_training.py --trace --output traces/gdn_direct_training/final16k.json
CUDA_VISIBLE_DEVICES=3 /opt/ydt/bin/python profile/bench_direct_gdn_training.py --seqlens 131072 --output traces/gdn_direct_training/final128k.json
CUDA_VISIBLE_DEVICES=0 /opt/ydt/bin/python profile/bench_direct_gdn_training.py --seqlens 16384 131072 --memory-only --output traces/gdn_direct_training/final_memory.json
CUDA_VISIBLE_DEVICES=6 /usr/local/cuda/bin/compute-sanitizer --tool memcheck --error-exitcode 1 /opt/ydt/bin/python -m pytest -q tests/test_direct_gdn_layer.py -k 'not full_layer'
```

# GDN optimizations

## Recommended implementation

For the best measured BF16 full-layer path in this repository, use the
experimental `direct_gdn_layer` entry point. It combines the optimized Hopper
GDN forward with the existing FlashQLA recurrence backward, fused output
RMSNorm/SiLU gating, and direct writes of component gradients into their final
packed buffer. The recurrence inverse is rebuilt during backward, never saved.
The default backward workspace reuses dead Q/K/V gradient columns for the
temporary inverse and, when the layout permits, the recurrence output gradient.

```python
output = direct_gdn_layer(
    x, weights, conv_weights, a_log, dt_bias, norm_weight, output_weight,
    key_heads=Hk, value_heads=Hv, partitions=8,
)
```

This is the most optimized measured path for a single unpadded sequence on
Hopper: contiguous BF16 activations and weights, FP32 decay parameters,
`Dk=Dv=128`, zero convolution history, and a sequence length divisible by 512
with the default eight partitions. It does not support autocast, biases,
initial/final state, distributed context parallelism, variable-length batches,
or higher-order gradients. The regular `chunk_gated_delta_rule` API remains the
integration path for other layouts and state modes.

The layer's selected stages are:

| Stage | Selected implementation |
|---|---|
| Q/K/V/Z/A/B projections | Six independent BF16 projections in forward; backward packs their gradients and uses two GEMMs for input and weight gradients. Parameters remain independent. |
| Q/K/V causal convolution | Existing Dao causal convolution with SiLU; backward writes directly into the packed projection-gradient slices. |
| Scalar gates | FP32 softplus/sigmoid forward values; fused backward writes A/B gradients directly and reduces decay-parameter gradients. |
| Gated delta rule | Hopper local inverse preparation, eight-partition affine state summaries, and pipelined output recurrence; full history, with no decay truncation. |
| Output RMSNorm and Z gate | One Triton forward kernel; backward recomputes statistics and writes dZ into its final gradient slice. |
| Recurrence backward | Existing FlashQLA default backward and state recomputation. It rebuilds the local inverse into ephemeral workspace; no inverse or other new activation is saved. |
| Projection-gradient workspace | A single packed allocation is reused for inverse and recurrence scratch before its Q/K/V columns become final gradients. |

On the H100 fixture, the fused training layer measures 12.30 ms at 16K and
100.20 ms at 128K for forward plus backward. Fusing the output norm/gate
forward improves the otherwise identical direct-buffer layer by 5.4% and 4.2%,
respectively, and removes two GPU kernels. These runs exclude the optimizer.
The separately measured workspace reuse lowers the 16K allocator peak by
42.5 MiB while saving the same activations; there is no measured 128K peak or
latency result for that change. The timing pair used separate recurrence
scratch, so it does not measure the latency effect of enabling workspace reuse.
The forward fusion and workspace reports link
the paired samples, traces, tests, and measurement scope below.

The optimized forward can also be selected through the public recurrence API
with `fwd_backend="optimized", fwd_partitions=8, fwd_decay_threshold=None`.
For training, pair it with `bwd_backend="default"`; this rebuilds backward
intermediates on demand. Do not enable inverse caching when the no-extra-saved-
tensor policy is required.

The FP8 and DeepGEMM sections below are experiments rather than the recommended
path: they have larger numerical differences and no convergence validation.
The combined forward projection-backward alternative also failed to improve
the 128K full-layer result and uses more workspace. Older experiments are
retained as evidence for these choices; the sections below distinguish measured
improvements from ideas that did not become the selected implementation.

---

## Optimized Hopper forward details

The opt-in path targets BF16 inference with batch size 1, key/value dimension
128, and equal partitions aligned to 64 tokens. It retains the full history
by default. Optional decay truncation is described below. Training uses the
optimized forward with the existing FlashQLA backward.

## Kernels

- **[Local preparation](flash_qla/ops/gated_delta_rule/chunk/hopper/local_precompute.py):**
  Computes the 64-token inverse
  `AB = (I + diag(beta) strict_lower(K Kᵀ))⁻¹ diag(beta)` and the chunk-local
  cumulative log gates. It inverts 8×8 diagonal blocks with warp shuffles, then
  assembles larger blocks with tensor-core matrix multiplications. Folding beta
  and computing the gate prefix sum here removes repeated work and a separate
  prefix-sum launch. Inverse intermediates use FP16. The 8×8 composition reads
  existing block-diagonal 16×16 slices instead of materializing padded operands;
  the 8×8 and 16×16 stages keep their intermediate products in registers.
- **[Partition preparation](flash_qla/ops/gated_delta_rule/chunk/hopper/prepare_mn.py):**
  Computes each partition's affine state summaries, `S_out = M S_in + N` in
  key-first notation. Two math warp groups keep the summaries in registers
  while producer warps stream inputs through shared memory with TMA. The first
  chunk skips redundant identity/zero products, and scalar decay is applied
  once at the final M store. The last outgoing summary can be skipped. With
  truncation enabled, heads whose incoming-state transfer is discarded skip
  M's matrix products entirely, including when the cutoff is reached at the
  oldest retained chunk. Their N computation is unchanged. For later chunks
  that retain M, its update and N's projection share one warp-group handoff
  and completion wait; the existing scratch buffers are reused.
- **[Output recurrence](flash_qla/ops/gated_delta_rule/chunk/hopper/output_fwd.py):**
  Starts from the corrected partition state and computes outputs and state
  updates chunk by chunk. It keeps the state in registers in value-first
  orientation, uses WGMMA for the matrix products, and pipelines input loads
  and output stores. A 512-thread CTA has two state math warp groups, a separate
  QK math warp group, and one producer group. A dedicated producer warp writes
  outputs independently of gate preparation. QK tiles have their own two-stage
  ready/free pipeline, so their computation overlaps the state recurrence.
  Transposing the output fragment before its shared-memory copy enables
  efficient STMatrix stores followed by TMA writes. The two state projections,
  `SᵀKᵀ` and `SᵀQᵀ`, are queued together before one warp-group handoff and one
  completion wait, preserving the existing operand layouts.

[The forward wrapper](flash_qla/ops/gated_delta_rule/chunk/hopper/fast_fwd.py)
connects these kernels through the existing initial-state correction kernel.
It uses BF16 partition summaries and caches scan metadata. Enable it with
`fwd_backend="optimized", fwd_partitions=8` on the public API.

## Overlap the M update and N projection

Optimized preparation now queues `Mᵀ += ZᵀX` and `Yᵀ = NᵀKᵀ` under one
warp-group handoff and one completion wait. The operations use separate
accumulators, and the wait finishes both before the shared scratch operand
is reused to finish N. The first chunk still waits immediately because N
has no incoming-state projection there. Heads that discard M keep their
existing N-only schedule. `prepare_mn(..., overlap_mn=False)` restores the
previous schedule; the optimized path enables the paired schedule by default.

Matched H100 runs with 8 partitions, full history, folded beta, BF16 summaries,
and the final outgoing summary skipped:

| Sequence | Preparation before → after | Full forward before → after |
|---|---:|---:|
| 16k | 215.2 → 210.6 µs | 0.6113 → 0.6034 ms |
| 128k | 1.659 → 1.621 ms | 4.4956 → 4.4436 ms |

Summaries and outputs were bit-for-bit identical. This is an approximately 2%
preparation improvement and 1% full-forward improvement in these runs; timings
remain subject to variation. Strongly truncated preparation has no M update
to overlap and showed no material improvement. A version that only deferred
the M wait, while keeping separate handoffs, did not improve performance.

See the [benchmark](profile/bench_mn_overlap.py),
[16k results](traces/gdn_mn_overlap/pair16k.json), and
[128k results](traces/gdn_mn_overlap/pair128k.json).

Validation: 44 forward/preparation tests passed, including a long weak-decay
case with significant M (6 training tests were excluded). The six new
overlap cases passed kernel-filtered racecheck with zero hazards. Both
scratch-lifetime variants compiled without register spills or local/stack
memory usage. Details are in the
[validation record](traces/gdn_mn_overlap/validation.json).

## Follow-up: skip M and batch the state projections

On the H100 fixture with 8 partitions, skipping M reduced truncated preparation
from approximately 36.8 to 32.7 µs at 16k, and 38.1 to 33.7 µs at 128k. Queuing
both state projections reduced the isolated output kernel from 253.9 to
249.3 µs at 16k and 1.978 to 1.883 ms at 128k. All matched benchmark results
were bit-for-bit identical to the previous optimized path. End-to-end gains
are smaller, and the timing samples show variation; full-history forward
medians did not show a meaningful improvement.

A single wider `Sᵀ[Kᵀ | Qᵀ]` GEMM was also implemented and checked. It needed
additional shared-memory tiles and duplicate input loads for the auxiliary QK
and state-update operations. Its small timing improvement did not justify
that cost; the active kernel uses the queued pair instead. Measurements and
the wider experiment are saved under
[gdn_two_optimizations](traces/gdn_two_optimizations/queued16k.json).

Validation for these changes: 31 forward/preparation tests and 6 training tests
passed, and all 9 pipeline/final-state/suffix cases passed a racecheck
filtered to the optimized kernels, with zero hazards. Both checked kernel
variants have zero stack/local-memory usage and no register spills.

## Training

The [full-layer training investigation](traces/gdn_backward_pipeline/README.md)
now includes projections, convolution/SiLU, gate preparation, GDN, output
norm/gating and all input/parameter gradients. An explicit autograd-enabled
`training_norm_gate` callable fuses output RMSNorm/SiLU gating and its backward,
with matching BF16 intermediate rounding and forward-cache contents. Initial
full-layer measurements improve 14.60 -> 13.42 ms at 16k and 119.79 -> 107.79 ms
at 128k. Nine GPU tests passed, including an independent FP64 reference and
small activations where epsilon matters. Compute Sanitizer memcheck reported
zero errors across all nine tests. The callable remains opt-in and supports
first-order gradients. An independent full-layer repeat measured
14.63 -> 13.32 ms at 16k and 119.18 -> 108.32 ms at 128k, approximately 9%
lower latency. Both variants use identical GDN/cache/partition policies;
only the output norm/gate implementation changes. Repeat Chrome traces and
all output/parameter gradient checks are linked in the investigation report.

The [FP8 projection training experiment](traces/gdn_fp8_training/README.md)
uses prequantized E4M3 weights, timed activation quantization and timed backward
weight dequantization, with BF16 input/parameter gradients from BF16 backward
GEMMs. All seven projections in FP8 forward reduce full-layer latency from
13.29 to 12.69 ms at 16k and 108.64 to 102.82 ms at 128k (4.5–5.4%). Three
GPU tests and memcheck pass. This changes numerical precision: full-layer
output relative RMS differs about 8.3%, with maximum gradient differences
about 9–11% on these fixtures. It remains opt-in; convergence is untested.

A [DeepGEMM blockwise experiment](traces/gdn_fp8_training/deepgemm/README.md)
uses YDT's actual quantization kernels (1x128 activation and 128x128 weight
blocks with power-of-two scales) and DeepGEMM 2.1.1 FP8 GEMMs. A matched
three-way full-layer benchmark measures BF16/rowwise FP8/DeepGEMM as
13.34/12.59/12.61 ms at 16k and 109.32/102.63/102.15 ms at 128k. DeepGEMM
and rowwise FP8 are effectively tied at training scope, with BF16 backward
GEMMs still dominant. Quantization/dequantization and scale conversion costs
are included; weight quantization alone is excluded. Three independent GPU
tests and memcheck pass. Output relative RMS differences remain about 8.7%;
convergence is untested.

The DeepGEMM adapter now [writes activation scales directly](traces/gdn_fp8_training/deepgemm/direct_scales/README.md)
in its SM90 TMA-aligned MN-major layout. FP8 payload/scales are bit-identical
to YDT. All seven scale-transpose launches disappear from full-layer traces.
At 128k, isolated activation quantization plus layout takes 0.360 vs 0.446 ms
for width 2560 and 0.861 vs 1.167 ms for width 6144 (19–26% lower latency).
Six GPU tests and memcheck pass. Full-layer timing is noisier and does not
establish a stable percentage improvement. Weight dequantization and BF16
gradient GEMMs remain timed. Direct scales are the experimental DeepGEMM
default; `--legacy-scales` selects the original layout.

The [next training optimizations](traces/gdn_training_next/README.md) share FP8
activation quantization across six projections and write projection gradients
directly into one backward buffer. The explicit `fp8_gdn_layer` entry point
uses two combined BF16 projection-backward GEMMs, with all weight dequantization
timed and the same standard GDN/partition/cache policy. The preceding FP8 path
measures 12.42/99.77 ms at 16k/128k; direct buffers measure 11.58/95.03 ms
(6.8%/4.7% lower latency), with bitwise-identical forward output and 0.367%
maximum gradient RMS difference. Peak workspace rises about 0.5/3.8 GiB.
Optional convolution/SiLU/QK-normalization fusion has no consistent extra
training gain. State preparation removes unused M/Z workspace but establishes
no latency improvement. All 21 new/affected tests pass under memcheck, all
12 cached-backward tests pass, and four state cases pass filtered racecheck.
A new pitched-store int32 overflow at 128k was fixed with int64 indexing and
an explicit long-offset regression. This is not a fix for the earlier main
backward kernel's unresolved variation.

A [matched backward comparison](traces/gdn_fair_backward/README.md) now uses
the committed FlashQLA backward as reference, identical standard-precision
inverse/gates, eight partitions, and the same cache/truncation policy for both.
Fused grouped-head reduction, normalization and output casts reduce complete
backward latency about 12–13% with partition caches enabled, and 10–11% when
both paths reconstruct them. The main backward kernel has no measured gain.
The earlier training tables below compare complete configurations and should
not be interpreted as isolated backward-kernel improvements.

The public API supports autograd for eligible optimized inputs, including Q/K
normalization, grouped value heads, initial-state gradients, and losses on the
final state. By default, backward rebuilds the default local inverse and cumulative gates,
then calls the existing FlashQLA backward with its usual state recomputation
and partitioning. With `bwd_backend="default"`, optimized forward summaries
are not used as a backward cache.

These are the default recurrence's gradients evaluated with the upstream
gradients from the optimized outputs. They do not differentiate through the
optimized inverse's precision changes or optional decay truncation. Full history
remains the default. Unsupported layouts continue to use the default forward
and backward. The low-level `chunk_gated_delta_rule_hopper` remains a forward-only
kernel wrapper; use the public `chunk_gated_delta_rule` API for training.

Set `bwd_backend="optimized"` alongside `fwd_backend="optimized"` to retain an
unfolded BF16 inverse and the cumulative gates from the optimized training
forward. The forward emits this inverse from the computation it already
performs; backward reuses it with the existing FlashQLA backward kernels.
This removes duplicate inverse/gate preparation rather than moving that work
to another stage. Unsupported forward layouts use the default implementation.
Inference does not retain the training inverse.

This mode uses the optimized local solve's FP16 intermediate precision and
retains an additional `2 * T * Hv * 64` bytes until backward: 96 MiB at 16k
and 768 MiB at 128k with 48 value heads. Forward outputs are bit-for-bit
unchanged. On the H100 fixture, gradients differ from the default backward by
about 0.11–0.33% relative RMS. Default backward precision remains available with
`bwd_backend="default"`.

| Sequence | Default backward | Reused preparation | Full training step, default / reused |
|---|---:|---:|---:|
| 16k | 1.959 ms | 1.792 ms | 2.628 / 2.463 ms |
| 128k | 14.174 ms | 12.851 ms | 19.269 / 17.983 ms |

These inverse-only measurements precede partition caching. Set
`enable_fwd_cp_cache=False` to retain that mode. The backward improves by
8.5–9.3%; forward plus backward improves by 6.3–6.7%.
See the [measurements and validation](traces/gdn_cached_bwd/README.md).

For full-history optimized training with intra-CP, `enable_fwd_cp_cache=True`
(the default) now also retains the partition initial states and transfer
matrices. Backward uses the same equal boundaries as optimized forward rather
than independently choosing another partition count. The last partition's
transfer matrix is computed too: forward alone does not need it, but reverse
boundary correction can. This removes one partition-state preparation pass;
per-chunk states still get reconstructed. The default backward and truncated
forward modes keep their previous behavior. The existing backward warmup
cutoff remains in use.

For eight partitions and 48 heads, the retained FP32 initial states and BF16
transfer matrices occupy about 36 MiB, in addition to the saved inverse. The
benefit depends on forgetting: weak gates make the removed state preparation
more expensive; strong gates already shorten it. See the
[training comparison, gradient checks, and regression experiment](traces/gdn_training_cp_cache/README.md).

```python
# Inputs require gradients; the forward has the same optimized eligibility rules.
output, final_state = chunk_gated_delta_rule(
    q, k, v, g, beta,
    fwd_backend="optimized", bwd_backend="optimized", fwd_partitions=8,
)
output.float().square().mean().backward()
```

An [experimental implicit-solve backward](flash_qla/ops/gated_delta_rule/chunk/hopper/implicit_bwd.py)
uses factored gates and 17 GEMMs per chunk, avoiding the explicit inverse
gradient. It passes independent FP64 gradient checks, but its current schedule
is slower than the default backward and is not selected by the public API.
See the [experiment and measurements](traces/gdn_implicit_bwd/README.md).

## Optional decay truncation

Set `fwd_decay_threshold=5e-5` to enable truncation; `None` disables it.
For each partition and value head, preparation scans whole chunks backward
from the partition end until their cumulative decay is below the threshold.
It retains that suffix, including the chunk that crosses the threshold, and
starts its state contribution from zero. Older chunks are skipped, and the
boundary scan drops the incoming-state contribution for heads that reach
the cutoff. Heads that never reach it keep their full affine summary.

This assumes nonpositive log gates and introduces an approximation; the decay
threshold is not a universal bound on output error. Local preparation and the
output recurrence still compute every token. One-partition execution has no
partition-preparation work to truncate.

```python
with torch.no_grad():
    output, final_state = chunk_gated_delta_rule(
        q, k, v, g, beta,
        fwd_backend="optimized", fwd_partitions=8,
        fwd_decay_threshold=5e-5,
    )
```

The FlashInfer comparison always uses `fwd_decay_threshold=None`. To measure
truncation as a separate additional result, run:

```sh
CUDA_VISIBLE_DEVICES=3 /opt/ydt/bin/python profile/bench_fast_gdn.py \
    --decay-threshold 5e-5 --output traces/gdn_optional_truncation_16k
```

The `optimized` result remains full-history; `optimized_truncated` is separate.

Measured with `5e-5`, using the same fixture and 8 partitions:

| Sequence | Full-history optimized | Truncated | RMS difference from full |
|---|---:|---:|---:|
| 16k | 0.614 ms | 0.458 ms | 0.0152% |
| 128k | 4.623 ms | 3.254 ms | 0.0052% |

Preparation retained about 3.7 chunks per head and partition, compared with
32 chunks at 16k and 256 at 128k. These are approximate results, separately
reported from the full-history FlashInfer comparison.

## Measurements

H100, B=1, Hq/Hk=16, Hv=48, Dk/Dv=128; normalized Q/K, no initial/final state,
full history, matched partition counts. Times are forward CUDA-graph medians.

| Sequence / partitions | Existing FlashQLA | Optimized | FlashInfer |
|---|---:|---:|---:|
| 16k / 2 | 1.206 ms | 0.612 ms | 0.662 ms |
| 16k / 8 | 1.026 ms | 0.596 ms | 0.682 ms |
| 128k / 8 | 7.404 ms | 4.418 ms | 4.564 ms |

Latest isolated stage comparisons, with 8 partitions (microseconds):

| Kernel / sequence | Previous version | Optimized | FlashInfer |
|---|---:|---:|---:|
| Local / 16k | 112 | 96 | 100 |
| Output / 16k | 288 | 264 | 277 |
| Local / 128k | 901 | 726 | 773 |
| Output / 128k | 2,191 | 1,970 | 2,143 |

FlashQLA local preparation includes cumulative log gates;
FlashInfer computes those gates in the output stage. The output stage benchmark
uses zero incoming states for each partition; full-forward measurements above
include partition preparation and state correction. See the
[stage benchmark](profile/bench_gdn_stages.py) and
[16k measurements](traces/gdn_stages/clean16k.json) and
[128k measurements](traces/gdn_stages/repeat128k.json). A separate 128k run on
GPU 2 suffered timing interference; its samples are retained with a validity
note. The reported 128k stage repeat and full-forward runs used GPU 0.

The optimized output differs from the existing reference
by about 0.48% relative RMS, versus about 0.27% for FlashInfer in this fixture.
The path remains opt-in because it changes intermediate precision.

## Shared-memory race and fix

The local inverse has a handoff between two stages: warps 0 and 1 write the
16×16 off-diagonal inverse blocks, then all four warps load the assembled
32×32 blocks for the next multiplication. Numerical tests passed, but Compute
Sanitizer reported read-after-write hazards between a shared-memory store in
the first stage and an `ldmatrix` load in the second. A correct result on a
particular run does not establish that this handoff is synchronized.

There was already a CTA synchronization call at this boundary. Replacing it
with a named CTA barrier still produced reports. The successful fix uses an
explicit shared-memory barrier initialized with an arrival count of 128:

```python
# Allocate once, before the computation.
inverse_ready = T.alloc_barrier(arrive_count=128)

# After the warp-local offdiag16 computations; all 128 threads participate.
T.barrier_arrive(inverse_ready)
T.barrier_wait(inverse_ready, 0)
# Now all four warps consume the assembled inverse blocks.
```

This establishes an explicit arrival/wait handoff before the consumers load
the shared results. We have verified that it clears the report; we have not
isolated why the original generated CTA barrier was insufficient, or whether
that behavior originates in code generation, compiler scheduling, or sanitizer
modeling.

A separate protective CTA barrier was also added after the next multiplication
and before writing its temporary product: the allocator aliases `raw` and
`tmp32` to the same shared-memory storage. All warps must finish reading `raw`
before any warp overwrites it as `tmp32`. This alone did not clear the reported
race; the explicit arrival/wait handoff above did.

The vectorized local-output load also uses a different lane distribution from
the final MMA stores. The kernel reuses `inverse_ready` with phase 1 after those
stores, before any lane reads the completed inverse for output. This cleared
a second read-after-write report exposed during the latest optimization.

Initial full-history validation: 24 tests passed; racecheck reported zero hazards for local
preparation and all six forward/final-state cases, including multi-chunk
pipelines. The checked local and output kernel variants had zero local-memory
loads/stores and zero stack usage. See the saved
[validation results and logs](traces/gdn_fast_final_16k/validation.json) and
[benchmark script](profile/bench_fast_gdn.py).

With optional truncation, all 32 tests pass, including independent FP64 suffix
summaries, mixed decays, initial/final states, unchanged weak-decay heads, and
invalid thresholds. Kernel-filtered racecheck reports zero hazards in the
optimized kernels. The unfiltered run reported 28 hazards in the legacy
`kkt_solve` kernel used to construct the test reference; that kernel is not
used by optimized forward. Logs are saved with the
[truncation validation](traces/gdn_optional_truncation_16k/validation.json).

Latest local/output changes: all 32 tests pass, and unfiltered racecheck reports
zero hazards across local preparation and all six forward/final-state cases.
The checked 16k kernels have no register spills or stack usage. See the latest
[validation](traces/gdn_next_clean_16k/validation.json),
[16k trace](traces/gdn_next_clean_16k/p8.json), and
[128k trace](traces/gdn_next_clean_128k/p8.json).

## Partition-count and M/N pipeline experiments

A full-history sweep with frozen copies of the previous optimized output and
current M/N kernels found that eight partitions remain best for this fixture:

| Sequence | 4 partitions | 8 partitions | 16 partitions | 32 partitions |
|---|---:|---:|---:|---:|
| 16k | 0.650 ms | **0.598 ms** | 0.652 ms | 0.740 ms |
| 128k | 4.993 ms | **4.460 ms** | 4.523 ms | 4.597 ms |

These runs exclude FlashInfer: the installed version does not expose the
matched-partition argument used by the earlier checkout. The frozen modules
avoid mixing concurrent workspace edits into the comparisons. See the
[16k sweep](traces/gdn_partition_sweep_before_16k/results.json) and
[128k sweep](traces/gdn_partition_sweep_before_128k/results.json).

No new M/N pipeline variant was promoted. Three-stage input buffering was
approximately neutral. Overlapping X preparation with a state projection
introduced 48 bytes of stack use and increased the 16k preparation time from
195 to 342 µs. Moving X preparation to an auxiliary warp group introduced
96 bytes of stack use and measured about 284–302 µs. A narrower M-update/N-
projection overlap had no spills, but was slightly slower at 16k and showed
no reliable improvement at 128k. Smaller auxiliary tiles also regressed.

The existing production M/N schedule is retained. Candidate numerical checks
passed against the frozen baseline; rejected candidates were not promoted or
claimed racecheck-clean. The
[experiment summary](traces/gdn_mn_pipeline/summary.json),
[compiled resources](traces/gdn_mn_pipeline/resources.json), and candidate
sources are saved alongside the results. Reproduce a controlled comparison
with [bench_mn_pipeline.py](profile/bench_mn_pipeline.py), passing the archived
candidate and baseline paths; use `--variants 2:1` for the fixed-schedule
`update_overlap.py` candidate.

## Spill-free lifetime and N-only specializations

The combined kernel now explicitly reuses one FP32 scratch tile and one BF16
scratch tile for X and the state projections. `lifetime_control=True` also
converts the next state operand while X is in flight, then waits before reusing
the accumulator. It does not keep two large FP32 projections live together.

`split_n_only=True` on the low-level `prepare_mn` function enables a separate,
compile-time N-only specialization for truncated heads whose transfer is
discarded. It removes the M accumulator entirely and uses part of the freed
register space to overlap X with N's projection. The combined launch handles
heads that still need M and the skipped last partition; an in-place N-only
launch fills the remaining outputs. Routing stays on the GPU and supports
mixed masks, including a discarded transfer with every chunk retained.

Both checked 16k BF16 variants have zero stack usage and no LDL/STL instructions.
All 43 preparation/forward tests pass. Racecheck filtered to the optimized
preparation kernels reports zero hazards in eight suffix/mixed-mask cases;
the unfiltered run also reports pre-existing races in the legacy `kkt_solve`
kernel used to construct the test reference.

The new scheduling flags remain opt-in: for the usual 16k fixture, combined
suffix preparation takes about 33 µs, whereas split dispatch takes about
37 µs. With an explicitly retained 128-chunk suffix at 128k, split dispatch
also regressed (633 versus 567 µs). Avoiding spills succeeded, but does not
by itself establish a speedup. See the
[lifetime benchmark](profile/bench_mn_lifetimes.py),
[16k measurements](traces/gdn_mn_lifetimes/16k_shared.json), and
[validation](traces/gdn_mn_lifetimes/validation.json).

## Whole attention-layer benchmark

`profile/bench_gdn_layer.py` follows the forward operation order in YDT's
`nn/modules/attention/gdn.py` and `delta_rule.py`. It measures six separate
bias-free input projections (Q, K, V, output gate Z, decay A, beta B), three
FLA causal depthwise convolutions with fused SiLU, FP32 sigmoid(beta) and
`-exp(A_log) * softplus(alpha + dt_bias)`, Q/K L2 normalization, GDN, per-head
RMSNorm multiplied by SiLU(Z), and the output projection. Q/K normalization
is measured as its own stage and is included exactly once in end-to-end timing.
The Chrome trace labels all seven regions. Stage timings use precomputed
inputs; the complete layer regenerates all intermediates every invocation.

This is an attention-module forward baseline with synthetic weights. It excludes
residual/MLP, activation-statistics instrumentation, state caches and distributed
communication. Head counts are 16 key / 48 value with dimension 128. The hidden
size defaults provisionally to 4096 because the saved model config only contains
head dimensions; override `--hidden-size` for the target model. Convolution size
4 and RMSNorm epsilon 1e-5 are also configurable. Linear/conv weights use
variance-preserving synthetic initialization, rather than pretrained weights.

```sh
CUDA_VISIBLE_DEVICES=0 /opt/ydt/bin/python profile/bench_gdn_layer.py \
  --hidden-size 4096 --seqlen 16384 --dt-max .01 \
  --output traces/gdn_layer_16k_dt001
```

At 16k tokens on H100, this run measured 6.220 ms end to end. Individual stages:
input projections 3.234 ms; convolution/SiLU 0.343 ms; decay/beta 0.026 ms;
Q/K normalization 0.099 ms; optimized GDN 0.765 ms; RMSNorm/output gate
0.850 ms; output projection 1.060 ms. Independent stage medians need not sum
exactly to end-to-end timing. The trace contains 26 GPU kernel invocations.
Complete-layer relative RMS error versus standard FlashQLA is 0.00542;
GDN error is 0.00450. The convolution is separately checked against FP32 PyTorch
grouped convolution with causal padding and SiLU, including the sequence start.

**Numerical limitation discovered by this benchmark:** YDT initializes dt in
[0.001, 0.1]. With the synthetic projection weights and seed 42, this gives
minimum chunk log-decay -196.70. Standard FlashQLA remains finite, but the
optimized GDN produces non-finite outputs. Validation fails before timing; see
`traces/gdn_layer_16k_ydt_decay/validation.json`. The performance run above
explicitly uses `--dt-max .01` (minimum chunk log-decay -23.77), retaining the
same layer equations and full history. It is not a validated optimized result
for the original dt initialization range. Use `--gdn-backend default` to
benchmark that range with the standard kernels. The optimized output shader's
separate `exp(gc)` and `exp(-gc)` factors are a likely overflow source; fixing
that requires a numerically stable decay formulation, rather than clipping gates.

The next fusion targets suggested by this baseline are packing input projections
into fewer GEMMs; fusing causal convolution/SiLU with Q/K normalization; and
fusing output RMSNorm, SiLU and multiplication. Output norm/gating alone costs
about as much as GDN here. Decay/beta pointwise fusion is a smaller opportunity.
No new layer fusions are enabled by this benchmark.

The 128k run (`traces/gdn_layer_128k_dt001`) measured 49.876 ms end to end:
input projections 26.044 ms, convolution 2.436 ms, gates 0.118 ms,
normalization 0.732 ms, GDN 4.786 ms, output norm/gating 6.440 ms,
and output projection 9.210 ms. Layer relative RMS error is 0.00541.
Both sequence lengths have exported Chrome traces with 26 GPU kernels.
Random initialization is independently seeded per run; changing input shape
changes subsequent RNG draws, so these are not identical-weight sequences.

A separate 16k baseline using the original YDT dt range and standard GDN
(`traces/gdn_layer_16k_ydt_standard`) is finite and takes 6.358 ms end to end.
Its trace is also exported. This is a separate decay setting, so its timing
should not be treated as a matched optimized-versus-standard comparison.

### Qwen3.8-Flash-Next shapes, 16k (supersedes provisional 4096 defaults)

The benchmark now defaults to hidden size **2560**, convolution width 4,
RMSNorm epsilon **1e-6**, 16 key heads and 48 value heads of width 128.
Projection dimensions were verified against the existing YDT layer trace
`/home/kareemm/arcadia/ml/torch/ydt/qwen38_ydt_gdn_16k_flash_qla_trace.json`;
the local Qwen layer oracle confirms the convolution and norm parameters.
The new results retain hashes of these sources.

At 16,384 tokens, input activation shape is `[16384,2560]`; Q/K projection
weights are `[2048,2560]`, V/Z weights `[6144,2560]`, A/B weights `[48,2560]`,
and output projection weight `[2560,6144]`. GDN inputs are
Q/K `[1,16384,16,128]`, V `[1,16384,48,128]`, and g/beta `[1,16384,48]`.

`traces/qwen38_flash_next_layer_16k/results.json` and `layer_trace.json`
contain the optimized run using the previously documented mild synthetic decay
setting (`--dt-max .01`, full history). Complete forward is **4.537 ms**:
input projections 1.990 ms; convolution/SiLU 0.335 ms; gates 0.025 ms;
Q/K normalization 0.098 ms; GDN 0.780 ms; output RMSNorm/gate 0.860 ms;
output projection 0.648 ms. The trace has 26 GPU kernels. Layer RMS error
against standard GDN on the same inputs is 0.00539 and convolution errors are
below 8e-6 versus FP32 PyTorch.

The separate standard-kernel run with YDT's original dt upper bound .1 takes
4.648 ms (`traces/qwen38_flash_next_layer_16k_standard`). Its inputs and output
are finite. It uses a different decay setting from the optimized run and is
not a matched speed comparison. Both use synthetic weights, not checkpoints.

```sh
CUDA_VISIBLE_DEVICES=0 /opt/ydt/bin/python profile/bench_gdn_layer.py --dt-max .01
# Original YDT decay initialization with standard GDN:
CUDA_VISIBLE_DEVICES=0 /opt/ydt/bin/python profile/bench_gdn_layer.py \
  --gdn-backend default --output traces/qwen38_flash_next_layer_16k_standard
```

### Convolution backend comparison

The layer benchmark now defaults to `--conv-backend cuda` and supports
`--compare-conv-backends` to measure FLA Triton, FLA's CUDA wrapper, and the
direct Dao CUDA entry point using identical projections/weights. Both the
installed FLA and local `fla-chunk128` fork implement their CUDA backend as a
wrapper around `causal_conv1d.causal_conv1d_fn`; there is no separate FLA-authored
CUDA forward kernel in these versions. FLA's independent forward is Triton.

Direct Dao uses transposed views of the contiguous token-major projections, so
no channel layout copy is needed. For this one-sequence benchmark it omits
sequence IDs; FLA's wrapper receives the usual cu_seqlens metadata and its
metadata handling is included in the measured call. All backends fuse SiLU.
Each backend is checked against the selected convolution and the complete
layer against standard FlashQLA; a separate convolution check uses FP32 PyTorch.

```sh
CUDA_VISIBLE_DEVICES=0 /opt/ydt/bin/python profile/bench_gdn_layer.py \
  --dt-max .01 --conv-backend cuda --compare-conv-backends \
  --output traces/qwen38_flash_next_layer_16k_conv_compare
```

The comparison completed with `causal-conv1d` 1.7.0 built against the installed
PyTorch 2.13.0/CUDA 13.0. At the Qwen3.8-Flash-Next 16k shapes, matched
CUDA-graph stage medians for all three Q/K/V convolutions (including SiLU) are:

| Entry point | Three convolutions | Complete layer |
| --- | ---: | ---: |
| FLA Triton | 0.301 ms | 4.554 ms |
| FLA CUDA wrapper | 0.388 ms | 4.583 ms |
| Direct Dao CUDA | 0.259 ms | 4.505 ms |

These calls use identical synthetic projections/weights, optimized GDN with
full history and `dt-max=.01`. Convolution outputs are bit-identical across
all three backends on this fixture. Layer relative RMS error against standard
GDN is 0.00539 for each. Results and the trace containing all backends are in
`traces/qwen38_flash_next_layer_16k_conv_compare`. The direct call is specifically
for one unpadded sequence; its sequence-ID-free path is not a replacement for
packed variable-length handling. The FLA CUDA wrapper and direct call reach the
same extension but use different sequence metadata paths, so their timing
is not a comparison of two independently implemented CUDA kernels.

The next optimization priority remains output RMSNorm/SiLU/multiply fusion
(currently 0.853 ms), followed by a measured packed-projection experiment
(six input GEMMs total 2.013 ms). Convolution/QK-normalization fusion is a
smaller remaining opportunity now that direct CUDA convolution is 0.259 ms.
For the best measured single-sequence CUDA path use `--conv-backend dao`;
`--conv-backend cuda` exercises FLA's CUDA entry point.

### Fused output gate and packed input projections

`flash_qla/ops/gated_delta_rule/chunk/hopper/norm_gate.py` adds a readable
Triton forward kernel for per-head RMSNorm followed by SiLU output gating.
It reads the GDN output and projected gate once and writes only the final gated
activation. Reduction and nonlinear arithmetic use FP32; it preserves BF16/FP16
rounding at the original normalization and SiLU output boundaries before their
product. A packed gate view is supported through its token stride, so packing
projections does not force a gate copy. The helper is explicitly forward-only
and rejects tensors requiring autograd. The layer benchmark enables it by
default (`--norm-gate-mode fused`); `eager` keeps the original comparison path.

Input projection weights can be packed once outside timed forwards. The
benchmark supports `--projection-mode packed` (one GEMM, strided slice views),
`qkv_zab` (two GEMMs), or `packed_contiguous` (one GEMM plus six runtime slice
copies), alongside the original default `separate`. All downstream layout and
convolution costs are included in complete-forward timings. These prepacked
weights are static benchmark fixtures; this is not integration of weight updates
or layer backward into a training module.

`profile/bench_gdn_layer_fusions.py` compares all variants with identical Qwen
shapes, synthetic weights and inputs, direct Dao CUDA convolution, eight GDN
partitions and full history. The existing mild decay setting (`dt_max=.01`) is
retained because of the previously documented GDN numerical issue.

| Variant | Complete layer | GPU kernels in trace |
| --- | ---: | ---: |
| Original separate projections and eager gate | 4.418 ms | 26 |
| Separate projections, fused gate | 3.870 ms | 24 |
| One packed projection, fused gate | 3.842 ms | 19 |
| Packed projection with contiguous copies, fused gate | 4.450 ms | 25 |
| QKV/ZAB projection pair, fused gate | 3.880 ms | 20 |

The isolated norm/gate stage drops from 0.922 ms to 0.198 ms (4.65x).
Separate input projections take 2.040 ms; one packed GEMM takes 1.995 ms.
Packing produces a small latency change and substantially fewer launches;
contiguous slice copies erase its benefit. The fused gate remains enabled by
default, while projection layout remains selectable and defaults to separate
because the additional packed-GEMM latency gain is small. Launch tuning across
4/8/16/32/64 rows and 4/8 warps found approximately 0.196–0.201 ms for most
configurations; the readable default of 32 rows and 4 warps is within this range.

All projection outputs are bit-identical across layouts on the benchmark fixture.
All optimized layer variants have relative RMS error 4.60e-5 versus the original
layer with the same GDN kernels, and 0.00539 versus standard GDN.
Twenty-five kernel tests pass against independent FP64 and PyTorch references, covering
nonuniform norm weights in activation dtype and FP32, BF16/FP16/FP32 activations,
contiguous and packed gate strides,
tail tiles, zero/tiny activations and epsilon values 1e-6/1e-5. Autograd misuse
is checked separately. Results, validation and the multi-variant Chrome trace
are in `traces/qwen38_flash_next_layer_fusions_16k`.

```sh
CUDA_VISIBLE_DEVICES=1 /opt/ydt/bin/python profile/bench_gdn_layer.py \
  --dt-max .01 --conv-backend dao --projection-mode packed \
  --output traces/qwen38_flash_next_layer_16k_fused
```

The selected packed/fused path also passes the ordinary layer benchmark CLI;
its standalone trace is `traces/qwen38_flash_next_layer_16k_fused/layer_trace.json`
with 19 GPU kernels and a separate-run complete-forward median of 3.895 ms.
Use the matched multi-variant run above for before/after comparisons.

### Convolution, SiLU, Q/K normalization, and gate preparation fusion

`flash_qla/ops/gated_delta_rule/chunk/hopper/prepare_qkv.py` implements a single
Triton preparation kernel. Each CTA owns one Q/K/V head and a tile of tokens.
It applies the causal depthwise convolution with zero history, accumulates in
FP32, evaluates SiLU, rounds to activation dtype, and normalizes Q/K with
`rsqrt(sum(x*x) + 1e-6)` in FP32. V is not normalized. Inputs may be strided
views of the packed input GEMM; outputs are contiguous GDN tensors. This avoids
materializing unnormalized Q/K and their separate normalization passes.

Optionally, value-head CTAs also prepare FP32 decay and beta from the A/B
projections. `gate_biases=(A_log,dt_bias)` selects this mode and returns
`(normalized_QKV, (g,beta))`. All heads/tokens have a unique writer, and this
kernel uses no shared-memory synchronization. The helper is forward-only,
handles one unpadded sequence and head width 128, and supports convolution
widths 2/3/4 and BF16/FP16/FP32 activations.

The fully fused layer is selectable with:

```sh
CUDA_VISIBLE_DEVICES=1 /opt/ydt/bin/python profile/bench_gdn_layer.py \
  --dt-max .01 --conv-backend dao --projection-mode packed \
  --preparation-mode fused_gates --output traces/qwen38_flash_next_layer_16k_prepared
```

Use `--preparation-mode fused` to fuse just convolution/SiLU/normalization,
leaving the original decay/beta expressions. `separate` retains direct CUDA
convolution plus existing normalization. The comparison driver is
`profile/bench_gdn_preparation.py`.

At 16k, convolution/SiLU/QK normalization alone drops from 0.367 to 0.287 ms
with the readable 32-token/4-warp default; 64 tokens yields 0.284 ms. Including
decay/beta, preparation drops from 0.390 to 0.310 ms. The packed layer's launch
count drops from 19 to 15 for QKV fusion and to **7** for QKV plus gate fusion.

Gate precision matters: the first trial used approximate exponential and a
stable rearrangement of softplus. Gate differences of about 1e-7 produced
0.00216 relative RMS layer differences through subsequent BF16 inverse rounding.
The final kernel uses libdevice exponential/log1p, PyTorch's softplus threshold,
and correctly rounded division for sigmoid. QKV, gates, and layer outputs are
bit-identical to the prior CUDA-convolution path on the 16k fixture. Error
versus standard GDN remains 0.00539; full history and the documented synthetic
`dt_max=.01` setting are retained.

Fifty preparation/output-gate tests pass, including FP64 mathematical references,
PyTorch comparisons, tail tiles, the causal sequence start, packed strides,
small/zero Q/K vectors, nonuniform norm weights, and extreme gate projections.
The precise-gate multi-variant run is saved in
`traces/qwen38_flash_next_preparation_precise_gates_16k`. The earlier approximate
gate trial is retained separately and is not the final implementation.

The final paired comparison is saved in
`traces/qwen38_flash_next_preparation_paired_16k`. Both complete forwards run
back-to-back in the same CUDA graph with external timing events; their order
alternates across 120 samples to limit bias from GPU clock drift. The fused path
wins 119/120 pairs, with median paired savings **0.0657 ms (~1.7%)** and mean
savings 0.0646 ms. Individual forward medians are 3.969 ms (CUDA baseline) and
3.898 ms (fused preparation). Preparation medians are 0.390 and 0.313 ms.
The full-layer improvement is modest because the unchanged GEMMs dominate.

Long replay runs showed overlapping/noisy whole-layer medians. One run on GPU0
became shared and nearly doubled in duration; its results are explicitly marked
unsuitable as final performance evidence. Stage timings alone are not used to
claim a large end-to-end improvement.

The ordinary layer benchmark now defaults to packed projections and
`preparation_mode=fused_gates`, alongside the existing fused output gate. These
supersede the earlier default descriptions above. Use `--projection-mode separate` and `--preparation-mode separate` to restore
separate projections/preparation.
Convolution-backend comparisons explicitly select separate preparation so they
still exercise the requested convolution backend. This is still a synthetic
forward benchmark, not layer backward/training integration; the existing
optimized GDN limitation at the original dt range remains unchanged.

### Combined input-projection backward without additional saved activations

`flash_qla/ops/input_projection.py` adds an opt-in training helper that retains
the original six weights and separate forward outputs. Backward packs the
incoming gradients and current weights as temporary workspace, then computes
one input-gradient GEMM and one weight-gradient GEMM. Only the existing input
and weights are saved; there is no persistent packed weight or new activation
cache. The external YDT modules and existing defaults have not been changed.

`profile/bench_gdn_projection_training.py` measures both projections and the
complete synthetic YDT layer, including convolution/SiLU, gate preparation,
Q/K normalization, recurrence, output RMSNorm/SiLU gate, and output projection.
The recurrence uses full-history optimized forward and default uncached
backward. All runtime packing is included. Three 16K complete-layer comparisons
showed modest forward/backward gains of approximately 1.4–3.3%; the final
projection backward measured 4.508 to 4.260 ms. At 128K, projection backward
regressed from 34.696 to 35.752 ms. Combining only dX did not improve this.

Saved activation storage is unchanged, but projection-backward peak temporary
allocation increases by 435 MiB at 16K and 3480 MiB at 128K. Other backward work
dominates the measured 16K full-layer peak, which is unchanged. Input-gradient
differences are about 0.38% relative RMS due to BF16 reduction boundaries;
parameter-gradient differences stay below 0.09% on these fixtures. Twelve tests
cover independent FP64 references, gradcheck, SGD updates, saved-tensor
pointers, unused outputs, frozen weights, BF16, and CUDA autocast.

The [training experiment report](traces/gdn_projection_training/README.md)
contains final timing samples, memory checks, source hashes, and Chrome traces.
This remains opt-in: avoiding runtime gradient concatenation through a shared
upstream gradient layout is the next opportunity for long sequences.

### Projection tuning and GDN output epilogue experiments (2026-10-13)

`profile/bench_gdn_projections.py` queries and measures explicit cuBLASLt
algorithms through the forward-only `ProjectionPlan` helper. It compares both
row/column-major descriptors, normal/pretransposed weights, PyTorch cuBLAS,
and PyTorch cuBLASLt on the exact input `[M,N,K]=[16384,16480,2560]` and output
`[16384,2560,6144]` GEMMs. Inputs/outputs are BF16 with FP32 accumulation,
64 MiB workspace per plan, and algorithm/layout choices are made before capture.
Weight transposes are static fixture preparation, not runtime layer copies.
The selected algorithm is queried from the current library; opaque algorithm
metadata is saved only for auditing, not reused across library/GPU versions.
A plan's workspace is intended for serialized calls on the caller's CUDA stream.

In the expanded sweep, 34 variants per projection were measured (32 explicit
cuBLASLt algorithm/layout candidates plus the two PyTorch selections). Input
projection stage medians improved 2.038 to 1.988 ms; output projection 0.751 to
0.731 ms. However, the paired complete layer showed only a 0.00152 ms median
saving, and the tuned path won 63/120 samples. Its output was bit-identical.
This is not a reliable whole-layer improvement, so the default projections
remain PyTorch. Results, trace and frozen source snapshots are saved in
`traces/qwen38_projection_tuning_layouts_16k`. Four ABI/layout tests pass against
independent FP64 matmul. The first row-major-only sweep is retained separately.

Two output epilogue prototypes add RMSNorm, SiLU and output-gate multiplication
inside GDN, preserving activation-dtype rounding. They are benchmark-only files
and are not wired into the production path. `profile/bench_gdn_output_gate.py`
compares the original output recurrence plus standalone fused norm/gate against
the prototype and then measures both complete layers back-to-back. All layer
work, including the actual projected gate and output GEMM, remains inside timing.

* A single output-store warp computes four head rows per iteration before TMA
  stores. It serializes too much work: the fused output stage takes 12.18 ms at
  16k. Its relative RMS stage/layer differences pass the numerical checks.
* A parallel version shares the epilogue across the 256 math threads, with
  explicit named barriers around shared output accesses and 8-lane reductions.
  It improves substantially over the one-warp prototype but still slows the
  output-plus-gate stage from 0.690 to 1.513 ms and the full layer from 3.618 to
  4.505 ms. The added work delays the recurrence; saving a global output pass
  does not compensate in this implementation. Results/source snapshots are in
  `traces/qwen38_output_gate_parallel_experiment_16k`.
* A register-fragment version reuses the dead state-projection scratch for
  squared output values and attempts a column reduction directly from the
  accumulator layout. TileLang cannot lower that reduction layout: its source
  index depends on a thread-owned reduction segment. It was rejected at compile
  time and never launched. Source and compiler diagnostics are archived in
  `traces/qwen38_output_gate_fragment_compile`.

None of these experiments is promoted. The current fused preparation and
standalone output norm/gate remain enabled. The measurements retain the same
synthetic BF16 fixture, full history and documented `dt_max=.01`; they do not
address the existing strong-decay numerical limitation. Concurrent workspace
changes to training/backward code were preserved.

### torch.compile output norm/gate alternative

`norm_gate_fwd_compiled` compiles the same RMSNorm, SiLU and gate multiplication
with `fullgraph=True` and `emulate_precision_casts=True`. The latter preserves
the original activation-dtype rounding boundaries. Select it in the layer
benchmark with `--norm-gate-mode compiled`; the existing fused Triton default
remains unchanged. `profile/bench_gdn_compiled_gate.py` compares eager, compiled,
and existing Triton stages, then measures the two optimized complete layers
back-to-back and exports a stage trace. Four additional numerical cases check
packed/contiguous gate inputs and BF16/FP32 norm weights against FP64 math.

This compiles the post-GDN operation; GDN remains its separate TileLang call.
It does not fuse normalization into the recurrent output kernel.

Initial CUDA initialization failures recovered on a later retry, with the GPU
context explicitly initialized before module imports. GPU validation and timing
completed successfully. At 16k, eager output norm/gating takes 0.964 ms,
`torch.compile` takes 0.204 ms, and the existing Triton helper takes 0.199 ms.
The compiled and handwritten implementations each launch one GPU kernel;
eager launches three. Complete-layer paired medians are 3.842 ms (Triton) and
3.845 ms (compiled), so the existing Triton default remains selected.
Twenty-nine numerical tests pass, including four new compiled-path cases.
Results, trace, successful test log and retained initialization diagnostics are
in `traces/qwen38_compiled_output_gate_16k`.

### Direct projection-gradient buffers for whole-layer training

`flash_qla/ops/gdn_layer.py` adds an explicit experimental whole-layer autograd
Function. Convolution backward writes Q/K/V gradients into shared-buffer
slices, fused norm/gate backward writes Z, and fused scalar gate backward writes
A/B. Two projection GEMMs consume the completed buffer without gradient
concatenation. The original weights remain independent parameters and are
packed temporarily during backward. No inverse or partition cache is retained.

The helper supports first-order training on Hopper with BF16 activations and
weights, FP32 decay parameters, D=128, one unpadded sequence, equal partitions,
and zero convolution history. Biases, distributed CP, state cache, and autocast
are outside this entry point. Existing defaults and external YDT modules are
unchanged. The synthetic fixture uses the documented dt_max=.01 setting.

Initial complete-layer measurements reduced forward/backward from 14.246 to
12.781 ms at 16K and from 113.952 to 104.880 ms at 128K. Matched backward kernels
with separate destinations plus packing took 13.323 and 109.579 ms, respectively,
isolating about 0.54/4.70 ms from direct gradient writes. Other gains come from
backward norm/gate fusion. These comparisons use eager training, rather than
compiled whole-layer training. Forward outputs match; maximum gradient RMS
differences are about 0.39%, dominated by combined BF16 input-gradient rounding.

Saved activation storage decreases from 1697 to 1307 MiB at 16K and from 13576
to 10456 MiB at 128K. The shared gradient buffer replaces six buffers plus a
concatenated copy, but early allocation can increase peak workspace. Final peak
accounting uses native allocator event history: additional live step allocations
increase from 2313 to 2633 MiB at 16K and from 18042 to 20602 MiB at 128K,
excluding resident inputs, parameters, and validation references. The increases
are 320 MiB / 2.5 GiB, caused by allocating Q/K/V gradient slices early while
recurrence backward runs. Saved activation storage still decreases.

Twenty tests passed; focused CUDA memcheck passed six direct-destination tests
with zero errors, including convolution and gating offsets beyond 2**31
elements. Later timing attempts on shared GPUs are excluded from isolated
performance claims. Final validation, memory accounting, limits, and source
hashes are in the [whole-layer training report](traces/gdn_direct_training/README.md).

### Fused output norm/gate forward in the direct-buffer training layer

`direct_gdn_layer` now calls the existing forward-only `norm_gate_fwd` inside
its autograd Function. This fuses RMSNorm, SiLU, and multiplication into one
kernel; the layer's existing recomputing backward still supplies gradients.
`output_gate_mode="fused"` is the default, and `"eager"` restores the previous
forward. No norm statistics, normalized outputs, or activated gates are added
to the saved set. The recurrence still uses default uncached backward.

Matched isolated H100 full-layer forward/backward medians improve from 13.001
to 12.297 ms at 16K (5.4%) and from 104.604 to 100.201 ms at 128K (4.2%). These
are additional gains relative to direct-buffer training with eager output
gating, rather than comparisons across the earlier benchmark runs. Forward
medians improve from 4.487 to 3.801 ms and from 36.471 to 30.148 ms. Full-step
kernel counts decrease from 61 to 59 at both lengths.

Saved activation storage and native allocator-history step peaks are unchanged:
1307/10456 MiB saved and 2633/20602 MiB additional live allocations at 16K/128K.
Output differences are about 0.004% relative RMS; maximum gradient differences
stay below 0.01% on the fixed-upstream fixtures. Thirty-nine direct-layer and
norm/gate tests pass, including both forward modes, grouped heads, gradients,
saved-storage checks, independent FP64 norm/gate references, and long offsets.
The synthetic fixture keeps the documented dt_max=.01 setting. Source hashes,
paired samples, allocation checks, and traces are in the
[fused training-forward report](traces/gdn_direct_fused_forward/README.md).

### Reusing backward projection storage for recurrence scratch

`direct_gdn_layer` now uses the packed projection-gradient allocation for the
uncached inverse solve's output by default. It also places the recurrence output
gradient in unused Q/K/V columns when those columns fit before the Z-gradient
slice. The recurrence backward finishes reading each scratch region before the
convolution backward writes its final Q/K/V gradients. This adds no saved tensor
and does not retain the inverse; set `reuse_gradient_workspace=False` to compare
the separate-allocation path.

A 16K-token native allocator trace measured a 44,514,944-byte (42.5 MiB, 1.6%)
whole-step peak reduction, from 2633.0 to 2590.5 MiB above resident inputs and
parameters. Both paths save the same 1307 MiB of activation storage. The trace is
for memory only; it does not establish a latency gain, and no 128K measurement
has been collected for this workspace change. Six workspace tests pass across
three head ratios, including repeated backward, inverse output strides, and
gradient agreement. Details are in the
[workspace-reuse report](traces/gdn_workspace_reuse/README.md).

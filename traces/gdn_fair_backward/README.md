# Fair GDN backward comparison

The backward speedup here comes from gradient postprocessing, with no cache or
inverse-precision advantage. The main recurrence backward is still FlashQLA's
kernel. Earlier complete-training results compared different forward paths,
partition counts and intermediate precision; they should not be used as
isolated backward speedups.

## Matched setup

H100 SXM, B=1, T=16k/128k, Hk=16, Hv=48, Dk=Dv=128, BF16, key-first state,
Q/K normalization, fixed upstream gradients, no initial/final-state loss.
Both paths use eight equal partitions, hence 384 main-kernel CTAs.

One shared standard-precision forward preparation supplies the exact same
tensor objects for the inverse, cumulative gates, normalized Q/K and their
statistics, partition incoming states, transfer matrices, warmup metadata and
upstream gradients. There is no optimized FP16 inverse in this comparison.
All preparation is outside backward timing. The saved inverse and gates are
available to both paths, as in regular FlashQLA training.

Two partition-cache policies are evaluated: both paths reuse the same cache,
or both paths reconstruct it. In the second case, the inverse remains saved
for both; only the partition-state cache is disabled. Truncation is disabled
for both by setting the forward/backward warmup threshold to negative infinity.
The benchmark supports `--truncation default` for a separate matched-cutoff run.

Reference backward driver, normalization math and main-kernel sources are
snapshots of commit `da06429d54b0f577de0a638f451ac8f0b395e0ac` (v0.1.3).
The reference driver's postprocessing reproduces the committed autograd path:
grouped-head reduction, separate FP32 normalization backward and BF16 casts.
CP/state reconstruction helpers are shared between variants. Source snapshots
and hashes are stored with each run. The current path fuses grouped-head
reduction, normalization and the cast; it preserves the intermediate BF16
rounding of the separate grouped reduction and uses FP32 accumulation.

## Complete backward results

Independent repeat on GPU 7, CUDA graphs, 50 alternating warmups and nine
alternating-order timing samples. Times include state reconstruction,
boundary-gradient correction, the main kernel, grouped Q/K gradient processing
and reverse gate-prefix sums. They exclude forward, loss and optimizer work.

Weak-gate fixture: `g=logsigmoid(randn)*0.001`.

| Sequence / partition cache | Committed FlashQLA | Current fused postprocessing | Reduction in latency |
|---|---:|---:|---:|
| 16k / cached | 2.262 ms | 1.982 ms | 12.4% |
| 128k / cached | 17.248 ms | 15.014 ms | 13.0% |
| 16k / reconstructed by both | 2.767 ms | 2.484 ms | 10.2% |
| 128k / reconstructed by both | 20.733 ms | 18.619 ms | 10.2% |

With stronger gates (`logsigmoid(randn)/16`) and truncation still off, cached
results are 2.265 -> 1.982 ms at 16k and 17.251 -> 15.021 ms at 128k.
Reconstructed results are 2.775 -> 2.494 ms and 20.859 -> 18.591 ms.
See [repeat samples and checks](repeat_gpu7/results.json). The first run on
GPU 6 reproduces the direction and similar gains: [results.json](results.json).

The first 128k cached Chrome traces show approximately 3.89 ms across separate
grouped reduction, normalization and casts, versus 1.71 ms across the two
fused Q/K kernels. The main kernel is about 8.18 ms in both traces. Replacing
the six postprocessing launches with two accounts for the measured gain.

## Main backward and experimental implicit solve

The main-only comparison uses identical reconstructed chunk states and
corrected terminal-state gradients too. In the first weak-gate cached run:

| Sequence | Committed main | Current main | Experimental implicit main |
|---|---:|---:|---:|
| 16k | 1.083 ms | 1.083 ms | 1.421 ms |
| 128k | 8.156 ms | 8.152 ms | 10.994 ms |

There is no main-kernel performance improvement to claim. Its kernel body is
unchanged; the current wrapper's only change from the committed wrapper is
the previously fixed handling of `scale=0`, irrelevant to these measurements.
The implicit-solve experiment remains slower even after identical preparation
and cache policy. Its algebra changes arithmetic rounding, so it is reported
separately rather than included in the matched-precision speedup claim.

## Correctness and memory

All eight matched configurations pass the gradient checks. The maximum
current-vs-reference complete-backward relative RMS difference is 5.34e-6
(0.000534%). All gradients have matching output dtypes and are finite.
The 22 normalization/postprocessing regression tests also pass; see
[tests.log](tests.log). This is parity against committed FlashQLA, not a
new independent FP64 validation of its main kernel.

Both paths retain the same preparation tensors. The shared partition cache
occupies about 36 MiB. Backward allocation peak increments in the first run
are identical: 1,054 MiB cached / 1,078 MiB reconstructed at 16k, and
8,264 MiB cached / 8,288 MiB reconstructed at 128k. The main kernel's temporary
states and gradients dominate the peak, despite less postprocessing traffic.

Memory counters are measured after compilation/warmup, before benchmark graph
capture, with a common fixture and warmed runtime in the same process.
`allocated_before_bytes`, `peak_allocated_bytes`, and `peak_increment_bytes`
are all retained in JSON. These are controlled incremental measurements,
not minimal standalone training-process memory estimates; compiled runtime
buffers and the cache retained for the other policy remain allocated.

Initial 128k main-only checks showed small dQ differences even when comparing
the committed kernel against another call to itself (up to roughly 0.14%
relative RMS). dK, dV, dg, db and dh0 were bitwise stable in those checks.
Later three-repeat checks were bitwise stable for both committed and current
main kernels in every repeat-run configuration. The cause is unresolved;
there is no universal determinism claim. A proxy-fence experiment is archived
in `fenced_main.py` and `fence_probe/`, but did not establish a fix and is not
enabled in production. Its drifting main-only timing samples are not used
as performance evidence.

An intermediate repeat on GPU 4 was interrupted when another workload took
most of the GPU memory. Its partial data and failure are retained in `repeat/`
and `repeat.log`, and excluded from the reported table. No external jobs were
stopped.

## Reproduce

```sh
LD_PRELOAD=/usr/local/cuda/compat/lib/libcuda.so.1 CUDA_VISIBLE_DEVICES=7 \
  TILELANG_CACHE_DIR=/tmp/flashqla-tilelang-cache \
  TRITON_CACHE_DIR=/tmp/flashqla-triton-cache \
  /opt/ydt/bin/python profile/bench_fair_gdn_backward.py --trace \
  --output traces/gdn_fair_backward/repeat_gpu7/results.json
```

Add `--implicit` to measure the experimental implicit-solve kernel with the
same preparation. The default settings use both cache policies, both sequence
lengths, both gate fixtures, eight partitions and truncation off. Chrome traces
in `repeat_gpu7/` are backward-only, with names such as
`131072_decay0.001_cached_optimized_postprocessing.json`.

# DeepGEMM blockwise FP8 projection experiment

The subsequent [direct-scale update](direct_scales/README.md) writes activation
scales directly in DeepGEMM's SM90 layout and removes its scale transpose.
The original measurements below use the earlier YDT contiguous scale layout.

This path uses DeepGEMM 2.1.1 `fp8_gemm_nt` and the actual YDT quantization
implementation in `core/ydt_core/fp8/quantization/blockwise.py`, loaded directly
from the requested checkout. No YDT files were changed. Loading the standalone
file avoids a circular import in YDT's full FP8 package initialization.
The benchmark records the quantizer path and SHA256.

Quantization matches YDT: E4M3 with 1x128 activation blocks, 128x128 weight
blocks, and power-of-two FP32 dequantization scales. DeepGEMM produces BF16
outputs. Its internal scale-layout conversion is included in timing.

Weight quantization occurs before the measured step. Forward activation
quantization, runtime weight dequantization, and BF16 input/parameter gradient
GEMMs are included. Backward retains the preceding experiment's semantics:
`dX = dY @ dequantize(W_fp8)` and `dW = dY.T @ X_original_bf16`.
This is a straight-through gradient; neither quantization nor its scales are
differentiated. No dequantized weight or converted scales are cached.

## Matched full-layer comparison

One H100 SXM 80GB, B=1, T=16k/128k, hidden=2560, Hk=16, Hv=48, D=128.
All seven projections participate. All variants use identical standard GDN
forward/backward/cache policy, P8, truncation off, and fused training norm/gate.
Includes projections, convolution/SiLU, scalar gates, GDN, output gating,
output projection, input gradient and every parameter gradient. Loss/optimizer
are excluded. BF16, rowwise FP8 and DeepGEMM are interleaved in the same
process/GPU using 50 alternating warmups and nine alternating-order samples.

| Sequence | BF16 | Previous rowwise FP8 | DeepGEMM blockwise FP8 | Reduction vs BF16 |
|---|---:|---:|---:|---:|
| 16k | 13.342 ms | 12.587 ms | 12.610 ms | 5.5% |
| 128k | 109.318 ms | 102.628 ms | 102.149 ms | 6.6% |

[Samples, source hashes and numerical checks](three_way/results.json).
DeepGEMM and rowwise FP8 are effectively tied at full-layer scope: the 128k
difference is 0.47%, smaller than sample timing variation. Two backward GEMMs
per projection remain BF16, limiting the training improvement from FP8 forward.

Chrome traces:

- [16k DeepGEMM](three_way/16384_fp8.json)
- [128k DeepGEMM](three_way/131072_fp8.json)
- [128k rowwise FP8](three_way/131072_rowwise.json)
- [128k BF16](three_way/131072_bf16.json)

The 128k profiling pass logged an allocator allocation failure followed by
successful recovery while tracing; the process completed successfully. Timing
samples were already collected before that profiling pass. The benchmark now
releases unused allocator memory before tracing to reduce profiling pressure.

## Isolated projections

A separate run on an otherwise idle GPU measures these 128k forward times,
including dynamic activation quantization:

| Projection | BF16 forward | DeepGEMM forward |
|---|---:|---:|
| Q | 2.028 ms | 1.535 ms |
| K | 2.010 ms | 1.545 ms |
| V | 5.708 ms | 3.623 ms |
| Z | 6.077 ms | 3.804 ms |
| A | 0.241 ms | 0.566 ms |
| B | 0.237 ms | 0.567 ms |
| Output | 5.635 ms | 4.176 ms |

[Forward and forward/backward samples](isolated_clean/results.json).
Tiny scalar projections do not benefit. These isolated medians must not be
added to full-layer medians. The first all-scope run (`results.json`) encountered
another GPU process during isolated measurements; use `isolated_clean/` for
operator timing and `three_way/` for the matched full-layer comparison.

## Numerical checks

In the matched full-layer fixture, DeepGEMM output relative RMS differences
versus BF16 are 8.63% at 16k and 8.67% at 128k. Maximum gradient relative RMS
differences are 9.71% and 9.68%, respectively (K convolution weight in both).
All outputs/gradients are finite and gradient dtypes match the BF16 baseline.
This blockwise variant does not establish improved full-layer numerical
accuracy over the previous rowwise FP8 experiment. Convergence is untested.

Three GPU tests pass, comparing against an independent power-of-two block
quantization formula and exactly checking the BF16 backward GEMMs. They cover
partial activation/weight tiles, grouped inputs, zero values and stale-weight
rejection. Compute Sanitizer memcheck reports zero errors across all three.
See [tests.log](tests.log) and [memcheck.log](memcheck.log).

The adapter is `flash_qla/ops/deepgemm_linear.py`, experimental and opt-in.
It requires DeepGEMM and an explicitly supplied YDT quantization source. It
supports first-order gradients. Refresh prepared weights after optimizer updates.

## Reproduce

```sh
LD_PRELOAD=/usr/local/cuda/compat/lib/libcuda.so.1 CUDA_VISIBLE_DEVICES=1 \
  TILELANG_CACHE_DIR=/tmp/flashqla-tilelang-cache \
  TRITON_CACHE_DIR=/tmp/flashqla-triton-cache \
  /opt/ydt/bin/python profile/bench_gdn_fp8_training.py \
  --fp8-backend deepgemm --legacy-scales --compare-rowwise --scopes full_layer --trace \
  --output traces/gdn_fp8_training/deepgemm/three_way/results.json
```

Use `--ydt-quantization-source PATH` for another checkout. Use
`--scopes projections` for isolated operator measurements.

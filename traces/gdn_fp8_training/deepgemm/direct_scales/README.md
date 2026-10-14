# Direct DeepGEMM scale layout on H100

The activation quantization kernel in `flash_qla/ops/deepgemm_linear.py` now
writes scales in DeepGEMM's required SM90 layout directly. For an MxK activation,
the logical FP32 scale tensor is `[M, K/128]` with strides
`(1, ceil(M/4)*4)`. The backing allocation includes the full padded last column
so TMA accesses stay within the allocation even when M is not divisible by four.

The quantization formula remains YDT's: 1x128 E4M3 activation blocks and the
same power-of-two multiplier, including zero and tiny-value handling. A Triton
CTA processes 128 rows and one K block, writing FP8 activations and MN-major
scales together. The 128x128 weight-block quantization and timed BF16 backward
weight dequantization remain YDT's original kernels.

On SM90, activation scales need MN-major, 16-byte-aligned storage; weight block
scales already use the required contiguous layout. The adapter targets the
H100/SM90 experiment. Blackwell's packed UE8M0 scale format is not implemented
by this kernel. See the [DeepGEMM layout rules](https://github.com/deepseek-ai/DeepGEMM/blob/v2.1.1/csrc/apis/layout.hpp).

Six GPU tests pass. The direct FP8 payload and scale values match YDT bit-for-bit.
DeepGEMM's scale-layout helper returns the same data pointer and strides for
the direct scales, proving that it accepts this allocation without a copy.
Independent forward/block-quantization and BF16 gradient checks also pass.
Compute Sanitizer memcheck reports zero errors across all six tests. See
[tests.log](tests.log) and [memcheck.log](memcheck.log).

The Chrome traces confirm zero `transpose_fp32` kernels in the direct path,
versus seven in the original-layout path for each full-layer training step.
At 128k, the original trace totals 0.098 ms for those seven transpose kernels.
The new quantization kernel is also faster than the original YDT quantizer on
the large activation fixtures below; savings exceed the transpose cost alone.

## Isolated activation quantization plus required layout

Both paths receive identical BF16 input. The original path uses YDT quantization
followed by DeepGEMM's scale-layout helper; the direct path writes both results
in one launch. CUDA graphs, 50 alternating warmups, nine alternating-order
samples, one H100. Values and quantization precision are unchanged.

| Tokens | Width | Original quantization + conversion | Direct scales |
|---|---:|---:|---:|
| 16k | 2560 | 0.0487 ms | 0.0468 ms |
| 16k | 6144 | 0.1143 ms | 0.1083 ms |
| 128k | 2560 | 0.4455 ms | 0.3603 ms |
| 128k | 6144 | 1.1668 ms | 0.8608 ms |

[Samples and source hashes](quantization_results.json). At 128k these are
19% and 26% reductions respectively. Do not add these isolated medians to
full-layer medians; the fixtures and buffer lifetimes differ.

## Matched full-layer training

Identical GDN kernels, P8, truncation off, forward caches, BF16 backward GEMMs,
timed weight dequantization, and all input/parameter gradients. Weight
quantization alone is excluded. Direct and original layouts are interleaved
with a BF16 baseline in one process on one GPU.

| Tokens | Original DeepGEMM scales | Direct scales |
|---|---:|---:|
| 16k | 12.462 ms | 12.617 ms |
| 128k | 103.209 ms | 101.668 ms |

[Samples, output/gradient checks and source hashes](results.json),
[128k direct Chrome trace](131072_fp8.json),
[128k original-layout Chrome trace](131072_legacy_scales.json).
The 128k samples contain substantial timing variation, and the 16k result
does not show a gain. These full-layer measurements do not establish a stable
percentage improvement. The isolated timing improvement and removal of all
scale-transpose kernels are directly verified.

This changes storage layout, not FP8 precision. Full-layer output and gradient
differences versus BF16 remain those of the preceding blockwise FP8 experiment;
it does not establish training convergence.

## Reproduce

Direct scales are the default for the experimental DeepGEMM adapter. Use
`--legacy-scales` to retain the old quantizer/layout, or compare both:

```sh
LD_PRELOAD=/usr/local/cuda/compat/lib/libcuda.so.1 CUDA_VISIBLE_DEVICES=1 \
  TILELANG_CACHE_DIR=/tmp/flashqla-tilelang-cache \
  TRITON_CACHE_DIR=/tmp/flashqla-triton-cache \
  /opt/ydt/bin/python profile/bench_gdn_fp8_training.py \
  --fp8-backend deepgemm --compare-scale-layout --scopes full_layer --trace \
  --output traces/gdn_fp8_training/deepgemm/direct_scales/results.json
```

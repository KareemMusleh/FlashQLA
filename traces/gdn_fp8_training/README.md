# FP8 forward projections with BF16 backward

`flash_qla/ops/fp8_linear.py` implements bias-free E4M3 forward projections
with dynamic per-token activation scales and per-output-channel weight scales.
The scaled GEMM produces BF16 outputs. Activation quantization is timed.
Weight quantization is explicitly prepared outside the measured step, as
requested; it must be refreshed after each optimizer update.

Backward dequantizes the FP8 weight to BF16 on every invocation, then computes
both gradients with BF16 GEMMs:

```
dX = dY @ dequantize(W_fp8)       # BF16 result
dW = dY.T @ X_original_bf16      # BF16 result
```

This defines a straight-through surrogate gradient. Quantization/scales are
not differentiated. The original BF16 parameter receives dW; a version check
rejects stale prepared weights after ordinary in-place optimizer updates.
No dequantized weight is cached across forward/backward or benchmark replays.
The implementation supports first-order training gradients, not higher-order
autograd. This experiment uses FP8 forward GEMMs and BF16 backward GEMMs;
it does not use FP8 backward GEMMs that merely return BF16 gradients.

## Full-layer training results

One H100 SXM 80GB, B=1, hidden=2560, Hk=16, Hv=48, D=128, P=8.
Both variants use identical standard GDN kernels/cache policy, truncation off,
and fused training output norm/gate. Timing includes all seven projections,
Q/K/V causal convolution/SiLU, scalar gates, GDN, output norm/gate, and input
and every parameter gradient. Loss and optimizer are excluded.

All seven projections using FP8 forward:

| Sequence | BF16 projections | FP8 forward / BF16 backward | Latency reduction |
|---|---:|---:|---:|
| 16k | 13.288 ms | 12.689 ms | 4.5% |
| 128k | 108.643 ms | 102.815 ms | 5.4% |

[Raw samples and numerical checks](results.json),
[16k FP8 Chrome trace](16384_fp8.json),
[128k FP8 Chrome trace](131072_fp8.json).
The corresponding BF16 traces are `16384_bf16.json` and `131072_bf16.json`.
Nine samples use alternating variant order after 50 alternating warmups.
The benchmark GPU was free of other active GPU jobs; other devices were busy.

The 128k trace includes seven activation quantization kernels totaling
2.712 ms and seven weight dequantization kernels totaling 0.065 ms.
Projection GEMMs total 53.424 ms versus 61.509 ms for BF16. These profiled
kernel sums and graph timing medians come from separate measurements.
Two backward GEMMs remain BF16, limiting the improvement in training time.

Keeping the tiny A/B projections BF16 while using FP8 for Q/K/V/Z/output:

| Sequence | BF16 projections | Five FP8 forward projections | Latency reduction |
|---|---:|---:|---:|
| 16k | 13.366 ms | 12.823 ms | 4.1% |
| 128k | 108.365 ms | 102.941 ms | 5.0% |

[Raw samples, numerical checks and traces](large_projections/results.json).
This does not establish a full-layer advantage over quantizing all seven;
the small difference is comparable to run-to-run timing variation. The
isolated A/B projections are slower with FP8 (0.95 versus 0.72 ms each at
128k, forward plus backward). Their forward saves little GEMM work while
activation quantization still reads the entire hidden activation.

## Numerical results and validation

All full-layer outputs and gradients are finite and gradient dtypes match
the baseline. FP8 is a different precision algorithm, not a parity-preserving
fusion. With all seven projections quantized, output relative RMS differences
versus BF16 are 8.30% at 16k and 8.33% at 128k. The largest parameter/input
gradient differences are 11.04% at 16k (A_log) and 9.39% at 128k (K convolution
weight). Keeping A/B BF16 gives output differences of 8.24%/8.28%, and maximum
gradient differences of 12.27%/9.30%. These are synthetic-fixture measurements,
not convergence or loss-quality results. The path remains experimental/opt-in.

Three GPU tests pass, including an independently formed dequantized-weight
reference for dX, exact BF16 GEMM comparison for dW, grouped inputs, padded
activation rows, zero inputs/weights and stale-weight rejection. Compute
Sanitizer memcheck ran all three tests with zero errors. See [tests.log](tests.log)
and [memcheck.log](memcheck.log). Full-layer checks record each parameter's
error rather than applying the <1% threshold used for precision-preserving
fusion tests.

## Use and reproduce

```python
from flash_qla.ops.fp8_linear import fp8_linear, quantize_linear_weight

# Run after an optimizer update, outside the measured forward/backward.
prepared = quantize_linear_weight(weight)
y = fp8_linear(x, weight, prepared)
# backward dequantizes weight and produces BF16 dX/dW.
```

```sh
LD_PRELOAD=/usr/local/cuda/compat/lib/libcuda.so.1 CUDA_VISIBLE_DEVICES=0 \
  TILELANG_CACHE_DIR=/tmp/flashqla-tilelang-cache \
  TRITON_CACHE_DIR=/tmp/flashqla-triton-cache \
  /opt/ydt/bin/python profile/bench_gdn_fp8_training.py --trace

LD_PRELOAD=/usr/local/cuda/compat/lib/libcuda.so.1 CUDA_VISIBLE_DEVICES=0 \
  TILELANG_CACHE_DIR=/tmp/flashqla-tilelang-cache \
  TRITON_CACHE_DIR=/tmp/flashqla-triton-cache \
  /opt/ydt/bin/python profile/bench_gdn_fp8_training.py --scopes full_layer \
  --fp8-projections q k v z out --trace \
  --output traces/gdn_fp8_training/large_projections/results.json
```

For the isolated projection scope, each projection uses fixed BF16 inputs and
upstream gradients. It includes forward quantization, backward weight
dequantization and dX/dW, but excludes summing different projections' dX for
the shared hidden input. The full-layer benchmark includes that accumulation.
Do not add isolated medians to full-layer medians: activation distributions,
buffer lifetimes and runtime conditions differ.

The subsequent [DeepGEMM blockwise experiment](deepgemm/README.md) uses the
actual YDT block quantization kernels and adds a matched three-way comparison.

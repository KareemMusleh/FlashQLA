# Shared FP8 quantization, direct projection gradients, and convolution fusion

The experimental training entry point is
`flash_qla/ops/fp8_gdn_layer.py::fp8_gdn_layer`. It shares one blockwise FP8
quantization across all six input projections and writes convolution/gate
gradients directly into a packed buffer consumed by two BF16 projection
backward GEMMs. It preserves the preceding benchmark's standard GDN forward,
eight partitions, full history, CP cache, BF16 inverse, and training output
norm/gate caches. Weight quantization remains outside timing; activation
quantization, weight dequantization, all gradient work and packing are timed.

## Full-layer training results

H100 SXM 80GB, B=1, hidden=2560, Hk=16, Hv=48, D=128, convolution width=4.
FP8 forward uses DeepGEMM blockwise E4M3 with direct-layout scales; gradients
are BF16, computed with BF16 GEMMs. Input and every parameter gradient are
included. Loss and optimizer are excluded. The baseline is the preceding FP8
layer, not the committed BF16 FlashQLA layer.

| Variant | 16k forward/backward | 128k forward/backward |
|---|---:|---:|
| Previous FP8 path | 12.420 ms | 99.769 ms |
| Shared activation quantization | 12.211 ms | 99.079 ms |
| Shared quantization + direct combined backward | 11.576 ms | 95.032 ms |
| Above + fused Q/K convolution/normalization | 11.469 ms | 95.591 ms |
| Above + state workspace cleanup | 11.509 ms | 95.354 ms |

[Final samples, checks and memory measurements](final/results.json).
The direct-buffer path reduces latency 6.8% at 16k and 4.7% at 128k in this run.
A preceding repeat measured 12.373 -> 11.571 ms and 99.753 -> 95.698 ms
([direct_repeat/results.json](direct_repeat/results.json)), confirming roughly
6–7% and 4–5% improvements. Convolution fusion has a small/inconsistent
additional benefit; the state change does not establish a latency improvement.
Those two changes should not be credited with a reliable training speedup.

Measurements use CUDA graphs containing both variant orders, 21 alternating
order samples, and workspace reuse within each graph. The tested GPU was free
of other active GPU jobs during final timing. Some other GPUs were busy.

Chrome traces:

- [16k baseline](final/16384_original.json)
- [16k direct-buffer path](final/16384_direct.json)
- [128k baseline](final/131072_original.json)
- [128k direct-buffer path](final/131072_direct.json)
- [128k convolution fusion](final/131072_direct_conv.json)
- [128k all changes](final/131072_all_direct.json)

## What changed

**Shared activation quantization.** Six input projections consume one temporary
FP8 payload/scale pair in forward, rather than quantizing the same hidden
tensor six times. Output projection still quantizes its separate input. The
traces therefore contain two activation quantization kernels instead of seven.
Quantized X is discarded after forward; original BF16 X is retained for
backward, exactly as before. Standalone use is
`shared_deepgemm_projections` in `flash_qla/ops/deepgemm_linear.py`.

**Combined BF16 projection backward.** One GEMM computes shared dX and another
computes all six dW matrices. The original parameters remain independent.
Input projection weights are dequantized during backward and concatenated
temporarily; that work is included. The initial implementation concatenated
activation gradients at runtime. Its 128k trace spent about 5.7 ms in large
copies, negating the GEMM/gradient-add savings. The final implementation uses
DAO convolution backward's destination argument, a pitched Z-gradient output
in the norm/gate kernel, and the existing scalar-gate backward destination
arguments to populate the packed gradient buffer directly. Projection GEMMs
drop from 21 to 11 across full training; six forward projections still execute
separately. The norm/gate keeps the same saved N/S/rstd as the baseline.

**Fused Q/K convolution, SiLU and normalization.**
`flash_qla/ops/conv_norm.py` computes width-2/3/4 zero-history convolution in
FP32, applies SiLU, rounds to BF16, then normalizes 128-element head vectors.
Its ordinary autograd adapter uses the existing convolution backward. The
direct layer supplies its norm statistics to the existing GDN backward so
group reduction and normalization backward stay fused; normalization is not
differentiated twice. V convolution remains DAO's kernel. This feature remains
optional (`fused_conv=False` by default) because its extra full-layer speedup
is inconsistent and forward rounding differs slightly from DAO's SiLU.

**State preparation.** Ordinary state reconstruction does not need the
partition-transfer matrix M. `prepare_h.py` no longer allocates full M/Z
scratch matrices for that specialization (about 48 KiB of source-level shared
workspace per CTA), nor computes their unused zero fragments. The CP
specialization still computes M normally. When a non-CP caller requests the
returned correction matrix, the wrapper explicitly zeros it, preserving the
previous result. The original two-stage pipeline and recurrence math remain.
Existing dispatch uses this workspace cleanup; no cache/truncation advantage
is introduced.

## Numerical and memory tradeoffs

The direct-buffer path's full-layer forward outputs are bitwise identical to
the previous FP8 path at 16k and 128k. Maximum input/parameter gradient relative
RMS differences are 0.367%, on dX, from combining BF16 reductions. Fused Q/K
convolution gives output differences of 0.789%/1.315% and maximum gradient
differences of 0.520%/0.643% relative to the preceding FP8 layer. An operator
diagnostic measured Q/K convolution output differences around 7e-6 relative
RMS, with 38 changed BF16 values per projection in the 16k fixture; later layer
operations amplify these sparse rounding differences. No convergence claim
is made. The underlying FP8-vs-BF16 difference remains separate from these
additional fusion/reduction differences.

The packed buffer is backward workspace, not a retained forward cache. Early
allocation increases peak live memory. After common warmup, measured peak
allocation increments above resident inputs/parameters are:

| Sequence | Previous FP8 path | Direct-buffer path |
|---|---:|---:|
| 16k | 2335 MiB | 2859 MiB |
| 128k | 18042 MiB | 21932 MiB |

Thus the measured latency gain costs about 0.5 GiB/3.8 GiB additional peak
workspace. These are allocator measurements of one warmed step, not minimum
standalone process/device-memory requirements. Consumed saved tensors are
released during ordinary backward; retained graphs preserve them.

## Validation and the 128k indexing fix

All 21 new/affected tests pass under Compute Sanitizer memcheck with zero
errors, including complete small-layer gradients, standalone shared
projections, convolution widths 2/3/4, norm/gate tests, state correction in
both layouts and CP modes, and the long-offset regression. All 12 existing
cached-backward tests pass. Filtered state-kernel racecheck passes all four
state-workspace cases with zero hazards. Logs: [final_memcheck.log](final_memcheck.log),
[cached_backward_regression.log](cached_backward_regression.log),
[state_racecheck.log](state_racecheck.log).

The first direct-buffer experiment uncovered an integer-index overflow at
128k: 131072 tokens * 16480 BF16 elements per packed row exceeds 2^31 elements.
The Z-gradient kernel formerly multiplied the token index by this pitch in
32-bit arithmetic. It now promotes the token index to int64 before that
multiplication. A 17-row, large-pitch regression independently exercises
offsets above 2^31 and checks the known SiLU derivative. This was an indexing
bug in the new pitched store, not a race-condition fix for the older GDN main
kernel. Failed-run logs are retained; their timings are excluded.

## Experiments not selected

`states.json` records a separate value-tiled state reconstruction prototype.
Its states were bitwise identical to the reference but it was slower:
0.516 versus 0.347 ms at 16k, and 4.077 versus 2.641 ms at 128k for its best
tested 64-value tile. It is not used by the training path.

`prepare_tuning.json` compares one/two/three pipeline stages. One stage was
slower; three-stage dh exceeded shared-memory capacity, and three-stage h was
slower even after workspace removal. Reducing X/Y register allowances also
hurt performance and was reverted. Two stages and the original register
allowances remain. An initial runtime-packed full-layer experiment (`results.json`)
also showed no 128k gain. These are rejected experiments, not active speedups.

## Use and reproduce

```python
from flash_qla.ops.fp8_gdn_layer import fp8_gdn_layer

# params: Q/K/V/Z/A/B weights, Q/K/V convolution weights,
#         A_log, dt_bias, norm weight, output weight.
# prepared: prequantized Q/K/V/Z/A/B/output weights; refresh after updates.
# context: the existing equal-P8, single-sequence intra-CP context.
y = fp8_gdn_layer(x, params, prepared, context, fused_conv=False)
```

This is an explicit experimental entry point, not automatic model dispatch.
It supports first-order training, one unpadded sequence, BF16 activations and
projection/convolution/norm weights, FP32 decay parameters, D=128, bias-free
projections, zero convolution history, and no distributed CP, initial/final
state or autocast. Loss and optimizer integration are not implemented here.

```sh
LD_PRELOAD=/usr/local/cuda/compat/lib/libcuda.so.1 CUDA_VISIBLE_DEVICES=0 \
  TILELANG_CACHE_DIR=/tmp/flashqla-tilelang-cache \
  TRITON_CACHE_DIR=/tmp/flashqla-triton-cache \
  /opt/ydt/bin/python profile/bench_gdn_training_next.py \
  --modes original shared direct direct_conv all_direct --trace \
  --output traces/gdn_training_next/final/results.json
```

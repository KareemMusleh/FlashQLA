# Fused output norm/gate forward in direct-buffer GDN training

The experimental `direct_gdn_layer` now uses the existing `norm_gate_fwd`
kernel for forward RMSNorm, SiLU, and multiplication. This runs inside the
layer autograd Function, whose existing recomputing backward handles all
gradients. The forward-only helper is not exposed as a standalone training
operation. No normalized output, activated gate, or norm statistics are saved.

`output_gate_mode="fused"` is the default; `output_gate_mode="eager"` restores
the prior three-operation forward for comparison. Backward, direct Q/K/V/Z/A/B
gradient writes, independent projection parameters, inverse recomputation,
and cache policy are identical in both variants.

## Matched full-layer results

H100, hidden=2560, Hk=16, Hv=48, D=128, convolution width 4, BF16 activations and
weights with FP32 decay parameters. Both variants use full-history optimized
recurrence forward and default uncached recurrence backward. The synthetic
fixture retains dt_max=.01 and norm epsilon 1e-6. These results exclude the
optimizer, distributed CP, state cache, and activation-stat instrumentation.

| Tokens | Forward eager / fused | Forward + backward eager / fused | Step improvement |
| --- | ---: | ---: | ---: |
| 16K | 4.487 / 3.801 ms | 13.001 / 12.297 ms | 5.4% |
| 128K | 36.471 / 30.148 ms | 104.604 / 100.201 ms | 4.2% |

Each scope uses 21 CUDA graph samples with alternating variant execution order
and external events around each variant. Median paired step savings are
0.635 ms at 16K and 3.882 ms at 128K. Individual medians above need not subtract
to the same paired median. Full-step traces have 61 kernels with eager gating
and 59 with fused gating at both lengths. GPU0 was idle outside these runs;
other GPUs are not included in the benchmark.

Unique retained activation storage is identical: 1307 MiB at 16K and
10456 MiB at 128K, with 28 saved tensor references in both variants. Native
allocation-event history also finds unchanged additional live step peaks:
2633 MiB / 20602 MiB, excluding resident input/parameters and validation
reference tensors. These are the same direct-buffer peak-memory tradeoffs
documented in the earlier whole-layer report.

Output relative RMS differences are 0.00400% and 0.00407%. Maximum gradient
RMS differences on the fixed-upstream fixtures are 0.00573% and 0.00952%.
Small forward differences arise from FP32 reduction/nonlinear arithmetic at
BF16 rounding boundaries. This is not a bitwise-equality claim.

## Validation and artifacts

Thirty-nine tests passed: the direct-layer tests now cover both eager/fused
forward with grouped/ungrouped heads, every input/parameter gradient, saved
storage, repeated backward, strided destinations and long offsets; existing
norm/gate tests cover independent FP64 references, BF16/FP16/FP32, FP32 norm
weights, packed strides, small/zero inputs, epsilon values, and compiled
comparisons. The layer entry point retains its existing BF16/first-order
limitations; mixed-dtype standalone helper tests do not expand that contract.

`results.json` contains timing samples, native allocation-history checks,
output/gradient checks, and source hashes. Per-length eager/fused Chrome traces
are included. The earlier standalone norm/gate forward-only contract remains
unchanged.

```sh
CUDA_VISIBLE_DEVICES=2 /opt/ydt/bin/python -m pytest -q tests/test_direct_gdn_layer.py tests/test_norm_gate.py
CUDA_VISIBLE_DEVICES=0 /opt/ydt/bin/python profile/bench_direct_gdn_output_gate.py --trace
```

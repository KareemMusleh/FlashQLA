# Combined GDN input-projection backward

`flash_qla/ops/input_projection.py` supplies an opt-in bias-free projection
function. Forward keeps six independent contiguous outputs and original
parameters. Backward packs the six incoming gradients and current weights as
temporary workspace, computes one dX GEMM and one dW GEMM, and returns slices
of dW for the original parameters. There is no persistent packed weight,
inverse, partition-state cache, or additional saved activation. Parameters can
be updated normally between steps.

Example replacing the six projection calls in GDN:

```python
from flash_qla.ops.input_projection import combined_input_projection

xq, xk, xv, xg, alpha, beta = combined_input_projection(
    x, self.q_proj.weight, self.k_proj.weight, self.v_proj.weight,
    self.g_proj.weight, self.a_proj.weight, self.b_proj.weight,
)
```

The external YDT modules have not been modified. This function assumes matched
input/weight dtypes and devices, supports CUDA BF16 autocast, and does not
support projection biases. `combine_weight_gradients=False` is an ablation
that combines dX but retains the separate dW GEMMs.

## Matched measurements

H100, hidden=2560, Hk=16, Hv=48, D=128, BF16 activations/weights, FP32 gate
parameters, convolution width 4. CUDA graphs include all runtime packing;
nine alternating-order samples per variant, fixed upstream gradients.

| Scope | Separate | Combined backward | Combined dX only | Packed parameter |
| --- | ---: | ---: | ---: | ---: |
| 16K projection backward | 4.508 ms | 4.260 ms | 4.317 ms | 4.191 ms |
| 16K complete-layer backward | 9.921 ms | 9.785 ms | 9.920 ms | 9.725 ms |
| 16K complete-layer forward + backward | 14.746 ms | 14.264 ms | 14.480 ms | 14.467 ms |
| 128K projection backward | 34.696 ms | 35.752 ms | 36.765 ms | 35.731 ms |
| 128K projections forward + backward | 51.207 ms | 52.419 ms | 53.004 ms | 52.204 ms |

The complete-layer benchmark follows YDT's projection, convolution/SiLU,
gate preparation, normalized delta rule, output RMSNorm/SiLU gate, and output
projection operation order. It uses full-history optimized forward and the
default backward, explicitly disabling forward CP caches and inverse caching.
Convolution uses Dao's CUDA implementation. It includes input and every
parameter gradient, but excludes optimizer updates and activation-stat logging.
It is synthetic, single-sequence, and has no distributed context parallelism.
The existing fixture uses dt_max=.01, not YDT's original .1; its documented
optimized-forward numerical limitation at the original decay range remains.

The full-layer 16K improvement is small and varied across runs: approximately
1.4%, 2.0%, and 3.3%. Treat this as a modest gain, not a precise 3.3% promise.
128K measurements cover projections only, and show a regression. The helper
remains opt-in; no existing default has changed.

The packed-parameter variant is a separate control using one authoritative
trainable weight and ordinary autograd, rather than a packed weight cached
alongside independent parameters. Strided projection outputs affect later
operators; projection-only gains do not predict whole-layer gains.

## Activation and workspace accounting

Saved-tensor hooks show identical unique retained activation storage for
separate and combined backward: 80 MiB for 16K projections and
1,779,433,856 bytes for the full 16K layer. The combined helper saves the
original input and six weights; pointer checks establish that no newly
allocated tensor is saved. Repeated input references in separate linear
operations do not imply repeated storage.

Packing still increases temporary backward memory. Eager peak allocation
increments, including returned gradients, are:

| Scope | Separate | Combined backward |
| --- | ---: | ---: |
| 16K projections | 240.47 MiB | 675.47 MiB |
| 128K projections | 1360.47 MiB | 4840.47 MiB |
| 16K complete layer | 1581.00 MiB | 1581.00 MiB |

The full-layer peak is dominated by other backward work, so this fixture's
peak does not rise. This does not make projection packing free. Removing the
large gradient concatenation by having upstream backward kernels produce a
shared GEMM layout is the next experiment; splitting dW alone did not help.

## Correctness and artifacts

Forward outputs match the separate baseline on the measured fixtures.
Parameter-gradient differences are below 0.09% relative RMS; changing GEMM
shapes changes accumulation rounding. Input-gradient relative RMS differs by about 0.38% because
independent BF16 dX products/additions round at different boundaries than the
combined reduction. CUDA tests compare dX to independent FP64 arithmetic.

Twelve tests cover FP64 outputs/gradients, unused outputs, frozen weights,
multiple leading dimensions, numerical gradcheck, original-parameter SGD
updates across three steps, saved-tensor pointers, CUDA BF16, and CUDA autocast.

Final timing/check/memory files are `final_16k.json` and `final_128k.json`.
Each records source hashes; 16K Chrome traces cover all four variants for
projection-only and complete-layer steps. Initial runs are archived in
`results.json` and `projections_128k.json`; their hashes identify older code.

```sh
CUDA_VISIBLE_DEVICES=0 /opt/ydt/bin/python -m pytest -q tests/test_input_projection.py
CUDA_VISIBLE_DEVICES=0 /opt/ydt/bin/python profile/bench_gdn_projection_training.py --trace --output traces/gdn_projection_training/final_16k.json
CUDA_VISIBLE_DEVICES=2 /opt/ydt/bin/python profile/bench_gdn_projection_training.py --seqlens 131072 --projection-only --output traces/gdn_projection_training/final_128k.json
```

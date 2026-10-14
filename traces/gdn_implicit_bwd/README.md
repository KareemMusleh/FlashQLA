# Implicit-solve backward experiment

The algebra and gradients validate, but the current kernel schedule is slower
than the default FlashQLA backward. The public API still selects the default
backward. This is a prototype for further scheduling work, not a production
performance improvement.

## Algebra

For each 64-token chunk, with key-first incoming state S and cumulative gate
matrix Gamma, define:

```text
R = strict_lower(K K.T)
B = diag(beta)
A = (I + B R)^-1
W = Gamma^-1 V - K S
U = A B W
P = causal(Q K.T)
O = scale Gamma (Q S + P U)
S_new = gamma_last (S + K.T U)
```

Given the total dU from the output and state-update paths, differentiate the
triangular system instead of forming an explicit inverse gradient:

```text
Z = A.T dU
dW = B Z
F = -strict_lower(Z U.T)
db = rowsum(Z * W) + rowsum(K * (F K))
dK_solve = B (F K) + F.T (B K)
```

The beta gradient does not divide by beta, and does not require another K K.T
product. Gate gradients come from row contractions for Gamma/Gamma^-1 and a
scalar chunk-end decay contraction, rather than a pairwise gate-ratio matrix.
The main kernel uses 17 GEMMs per head/chunk versus the existing kernel's 19.
It retains the default unfurled inverse and saved chunk states; state
reconstruction, boundary correction, normalization, grouped-head reduction,
and the reverse gate-prefix sum remain in the existing driver.

The prototype requires Hopper, BF16 Q/K/V/inverse/chunk states, FP32 gates,
beta and terminal-state gradients, B=1, Dk=Dv=128, contiguous inputs, and
64-token-aligned partition boundaries. It supports both state layouts and
terminal-state gradients. It does not support ragged tails. The benchmark
temporarily substitutes the experimental main kernel within the existing
training driver; no public backend selection was added.

## Measurements

Final isolated run on H100 GPU 1, B=1, Hk=16, Hv=48, D=128, BF16. Both variants
use the same optimized full-history forward and fixed upstream gradients.
Backward includes default inverse rebuilding, state reconstruction, boundary
correction, grouped-head reduction and Q/K normalization backward.
CUDA-graph medians use 50 alternating warmup rounds and nine alternating-order
timing samples. Partitions are chosen by the default training CP policy.

| Sequence / scope | Default backward | Implicit backward |
|---|---:|---:|
| 16k / main kernel | 1.061 ms | 1.379 ms |
| 16k / complete backward | 1.959 ms | 2.243 ms |
| 16k / forward + backward | 2.620 ms | 2.927 ms |
| 128k / main kernel | 8.150 ms | 10.859 ms |
| 128k / complete backward | 14.076 ms | 16.549 ms |
| 128k / forward + backward | 18.988 ms | 21.622 ms |

Gradient differences from the default backward are approximately 0.59% dQ,
0.61% dK, 0.50% dV, 0.85% dg, and 0.47% db relative RMS on these fixtures.
These are differences from the default implementation, not error bounds for
arbitrary training inputs. Factored gate arithmetic changes intermediate
rounding.

See [final samples and checks](isolated/results.json). The earlier combined
run in results.json contains interference in its 16k backward-only samples;
those medians exceed complete forward/backward timing, so they are not used
in the table. Earlier snapshots and timing files retain the progression from
serial copies, through reordered state updates, to TMA loads and stores.

The first serial prototype took about 2.42 ms for the 16k main kernel. TMA
loads reduced this to about 1.44 ms; TMA gradient stores improved it further.
The final checked 16k kernel uses 255 registers per thread, with zero stack
usage or register spills. Removing spills did not eliminate the performance
gap. Unlike the existing warp-specialized backward, this prototype runs the
matrix products on the same two cooperating math groups and waits between
products. It also retains substantial shared-operand traffic.

The next implementation target is a value-first register-operand pipeline:
two state/value math groups, a separate matrix-preparation group that computes
state-independent QK tiles ahead, and an independent load/store producer.
This should be evaluated while retaining the validated algebra, before
changing checkpointing or the default backward selection.

## Validation and reproduction

Seven gradient tests pass: five independent token-by-token FP64 autograd
comparisons and two full training-driver checks. They cover zero/unit beta,
zero output scale, weak/strong decay, grouped heads, normalization, both
state layouts, initial/final-state gradients, and four-chunk barrier reuse.
Zero-beta gradient accuracy is checked separately. The independent cases
require relative RMS below 3.5%; comparisons with the default training driver
require below 4%.

Exact-name racecheck of the experimental kernels reports zero hazards across
all five independent cases. A substring-filtered full-suite run also checked
legacy reference kernels and reported 56 hazards, all in kkt_solve. That
legacy result is saved separately and is not attributed to the prototype.
See [validation metadata](validation.json), tests.log, racecheck.log, and
resources.log. Compiler race-analysis warnings about the per-thread register
reduction are retained; the annotated ownership assigns each partial sum to
one thread, and Compute Sanitizer reports no hazards in the generated kernel.

```sh
CUDA_VISIBLE_DEVICES=1 /opt/ydt/bin/python -m pytest -q tests/test_implicit_bwd.py
CUDA_VISIBLE_DEVICES=3 /usr/local/cuda/bin/compute-sanitizer \
  --tool racecheck --kernel-name kne=kernel_kernel --error-exitcode 9 \
  /opt/ydt/bin/python -m pytest -q tests/test_implicit_bwd.py -k adjoint
CUDA_VISIBLE_DEVICES=1 /opt/ydt/bin/python profile/bench_implicit_bwd.py \
  --seqlens 16384 131072 --trace \
  --output traces/gdn_implicit_bwd/isolated/results.json
```

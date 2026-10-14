# Faster backward by retaining forward preparation

Use `fwd_backend="optimized", bwd_backend="optimized"` for eligible Hopper
training inputs. The optimized local preparation kernel now optionally emits
the unfolded BF16 inverse as well as its beta-folded forward inverse. Autograd
saves the unfolded inverse and cumulative gates, then feeds them to the
existing FlashQLA backward kernels. Backward no longer recomputes the inverse
or gate prefix sums.

The inverse is already computed during forward; the added work is an output
store, not a second solve moved into forward. Inference does not retain it.
`bwd_backend="default"` preserves the previous training path and precision.
Unsupported optimized-forward layouts fall back to standard forward/backward.

## H100 measurements

Batch 1, Hk=16, Hv=48, Dk=Dv=128, BF16, full-history optimized forward, with Q/K
normalization. Backward includes state reconstruction, boundary correction and
gradient postprocessing. The default training CP policy chooses partitions.
CUDA-graph timings use 50 alternating warmup rounds and nine alternating-order
samples with fixed upstream gradients. Both variants use the public API.

| Sequence / scope | Default backward mode | Reused preparation |
|---|---:|---:|
| 16k / forward | 0.664 ms | 0.678 ms |
| 16k / backward | 1.959 ms | 1.792 ms |
| 16k / complete training step | 2.628 ms | 2.463 ms |
| 128k / forward | 5.004 ms | 5.059 ms |
| 128k / backward | 14.174 ms | 12.851 ms |
| 128k / complete training step | 19.269 ms | 17.983 ms |

Backward improves by 8.5% at 16k and 9.3% at 128k. Forward/backward together
improve by 6.3% and 6.7%, including the added forward stores. Earlier separate
16k and 128k runs reproduce the improvement. Full samples and source hashes
are in [results.json](results.json); profiler traces confirm that inverse
rebuilding and the extra gate-prefix launch disappear, while the main backward
kernel and postprocessing remain.

Forward outputs are bit-for-bit equal. Gradients differ from the default mode
by about 0.328% dQ, 0.322% dK, 0.111% dV, 0.110% dg and 0.171% db relative RMS.
The saved inverse uses the optimized solve's FP16 intermediate precision; this
is an explicit precision tradeoff, not a universal gradient-error bound.
The default backward mode retains its previous intermediate precision.

The saved BF16 inverse occupies `2 * T * Hv * 64` bytes: 96 MiB at 16k and
768 MiB at 128k with 48 value heads. It is retained until backward, unlike the
previous optimized training path which rebuilt the inverse on demand.

## Validation

The final suite passes 88 tests: cached-backward checks, optimized-forward and
default-backward regressions, partition preparation, normalization gradients,
and the previous implicit-adjoint prototype. Cached-backward checks include
independent token-by-token FP64 autograd, zero/unit beta, zero output scale,
both state layouts, normalization/grouped heads, initial/final-state gradients,
fallback behavior, and a check that backward does not call kkt_solve.

Kernel-filtered Compute Sanitizer racecheck reports zero hazards for the six
inverse/training cases that exercise the new output. It checks local
preparation and the optimized forward kernels; the legacy backward kernels are
unchanged. The scale-zero tests exposed a legacy `scale or default` bug in the
Hopper forward/backward wrappers. They now use the default only for `None`,
including the CP state-adjoint helper.

See [validation.json](validation.json), tests.log and racecheck.log.

```sh
CUDA_VISIBLE_DEVICES=1 /opt/ydt/bin/python -m pytest -q \
  tests/test_cached_bwd.py tests/test_fast_gdn.py tests/test_prepare_mn.py \
  tests/test_l2norm_bwd.py tests/test_implicit_bwd.py
CUDA_VISIBLE_DEVICES=3 /usr/local/cuda/bin/compute-sanitizer \
  --tool racecheck --kernel-name kne=kernel_kernel --error-exitcode 9 \
  /opt/ydt/bin/python -m pytest -q tests/test_cached_bwd.py \
  -k 'cached_inverse or cached_backward'
CUDA_VISIBLE_DEVICES=0 /opt/ydt/bin/python profile/bench_cached_bwd.py \
  --seqlens 16384 131072 --trace --output traces/gdn_cached_bwd/results.json
```

## Other experiments

The implicit-solve kernel remains slower than the default main kernel. Two
additional candidates were measured and archived without changing production
dispatch: TMA gradient stores in the existing pipeline were bit-for-bit equal
but did not improve timing; separating the state recurrence and running
token gradients in chunk-parallel CTAs also increased total time. Their source
and measurements are saved under gdn_bwd_tma and gdn_bwd_parallel. The stage
benchmark can reproduce them with `--variant tma` or `--variant parallel`.

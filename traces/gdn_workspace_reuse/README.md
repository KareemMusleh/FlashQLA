# GDN backward workspace reuse

The direct training layer now reuses the already-required packed projection-gradient allocation as temporary recurrence workspace. It writes the uncached inverse into the Q/K/V prefix before those gradient columns are produced. For head layouts with enough unused prefix space, it also writes the recurrence output gradient there. The inverse is rebuilt on each backward call; no activation or inverse is saved. Set `reuse_gradient_workspace=False` on `direct_gdn_layer` to use the separate allocations for comparison.

The eligible workspace condition is checked at runtime. The inverse always fits in the Q/K/V prefix. The output-gradient region is reused only when the inverse plus output-gradient widths fit before the Z-gradient slice; otherwise the code allocates the output-gradient tensor as before.

On an NVIDIA H100, a native CUDA allocator trace of the synthetic 16K-token layer measured these eager forward/backward peak increments:

| Backward path | Peak above resident inputs and parameters |
| --- | ---: |
| Direct gradients, separate recurrence scratch | 2,760,900,864 B (2,633.0 MiB) |
| Direct gradients, reused scratch | 2,716,385,920 B (2,590.5 MiB) |
| Reduction | 44,514,944 B (42.5 MiB, 1.6%) |

Both direct paths saved the same 1,370,488,832 bytes of activation storage. Their outputs matched; their gradients stayed within the existing direct-layer comparison tolerance. The workspace tests also check both head layouts that fit the output-gradient region and a layout that only fits the inverse, sentinel boundaries around strided writes, normal and varlen inverse writes, and repeated backward calls. No timing comparison was collected; this change targets peak memory.

The measurement is recorded in [results.json](results.json). It uses allocator events for this process and excludes resident input, parameters, and reference gradients.

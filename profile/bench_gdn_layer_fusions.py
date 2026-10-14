"""Matched 16k layer comparison: fused output gate and packed projections."""
import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from bench_gdn_layer import Layer
from bench_prepare_mn import benchmark, error
from flash_qla.ops.gated_delta_rule.chunk.hopper.norm_gate import norm_gate_fwd


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seqlen', type=int, default=16384)
    parser.add_argument('--output', type=Path, default=Path('traces/qwen38_flash_next_layer_fusions_16k'))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(42)
    torch.set_num_threads(4)
    torch.cuda.set_stream(torch.cuda.Stream())
    config = SimpleNamespace(seqlen=args.seqlen, hidden_size=2560, key_heads=16,
                             value_heads=48, conv_size=4, norm_eps=1e-6, dt_max=.01,
                             partitions=8, conv_backend='dao', gdn_backend='optimized',
                             projection_mode='separate', norm_gate_mode='eager')
    layer = Layer(config)
    p = layer.project()
    c = layer.conv(p)
    g = layer.gates(p)
    n = layer.normalize(c)
    o = layer.gdn(n, g)
    eager_gate = layer.norm_gate(o, p)
    reference = layer.output_project(eager_gate)
    standard = layer.output_project(layer.norm_gate(layer.gdn(n, g, 'default'), p))
    print('Check and tune norm/gate launch', flush=True)
    tunes = {}
    checks = {}
    for rows in (4, 8, 16, 32, 64):
        for warps in (4, 8):
            name = f'rows{rows}_warps{warps}'
            fn = lambda rows=rows, warps=warps: norm_gate_fwd(o, p[3], layer.norm_weight, config.norm_eps, rows_per_cta=rows, num_warps=warps)
            candidate = fn()
            checks[name] = error(candidate, eager_gate)
            assert checks[name] < .006, checks
            tunes[name] = fn
    norm_timings = benchmark({'eager': lambda: layer.norm_gate(o, p), **tunes})
    best = min(tunes, key=lambda name: norm_timings[name]['median_ms'])
    print('Best norm launch', best, norm_timings[best]['median_ms'], flush=True)
    # Full-layer fused default is measured independently of this exploratory launch sweep.
    variants = {'baseline': ('separate', 'eager'), 'fused_gate': ('separate', 'fused'),
                'packed_fused': ('packed', 'fused'), 'packed_contiguous_fused': ('packed_contiguous', 'fused'),
                'qkv_zab_fused': ('qkv_zab', 'fused')}
    functions = {}
    validation = {}
    for name, (projection, norm) in variants.items():
        fn = lambda projection=projection, norm=norm: layer.forward(projection_mode=projection, norm_gate_mode=norm)
        candidate = fn()
        validation[name] = {'relative_rms_vs_baseline': error(candidate, reference),
                            'relative_rms_vs_standard_gdn': error(candidate, standard),
                            'finite': bool(candidate.isfinite().all())}
        assert validation[name]['finite'] and validation[name]['relative_rms_vs_baseline'] < .008 and validation[name]['relative_rms_vs_standard_gdn'] < .012, validation
        functions[name] = fn
    stages = {}
    for mode in ('separate', 'packed', 'packed_contiguous', 'qkv_zab'):
        pp = layer.project(mode)
        projection_error = max(error(a,b) for a,b in zip(pp,p))
        assert projection_error < .003, (mode, projection_error)
        checks['projection_' + mode] = projection_error
        stages['projections_' + mode] = lambda mode=mode: layer.project(mode)
        stages['projection_and_conv_' + mode] = lambda mode=mode: layer.conv(layer.project(mode))
    stages['norm_gate_eager'] = lambda: layer.norm_gate(o, p, 'eager')
    stages['norm_gate_fused'] = lambda: layer.norm_gate(o, p, 'fused')
    print('Benchmark matched complete forwards and stages', flush=True)
    result = {'config':vars(config), 'gpu':torch.cuda.get_device_name(),
              'scope':'same synthetic weights/input for all variants; direct Dao CUDA conv; full history; dt_max .01',
              'norm_launch_sweep':norm_timings, 'best_launch':best, 'stage_validation':checks,
              'layer_validation':validation, 'timings':benchmark(functions | stages),
              'sources':{}}
    for path in (Path(__file__), Path(__file__).with_name('bench_gdn_layer.py'), Path(__file__).resolve().parents[1]/'flash_qla/ops/gated_delta_rule/chunk/hopper/norm_gate.py'):
        result['sources'][str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    (args.output/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k in ('best_launch','layer_validation','timings')},indent=2),flush=True)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA], record_shapes=True) as prof:
        for name,(projection,norm) in variants.items():
            with torch.profiler.record_function(name):
                layer.forward(projection_mode=projection,norm_gate_mode=norm,labels=True)
        torch.cuda.synchronize()
    prof.export_chrome_trace(str(args.output/'layer_trace.json'))


if __name__ == '__main__':
    main()

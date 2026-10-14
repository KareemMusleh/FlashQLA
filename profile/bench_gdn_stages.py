"""Isolate local preparation and output with identical inputs and partition counts."""
import argparse
import importlib.util
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bench_prepare_mn import benchmark, error
from flash_qla.ops.gated_delta_rule.chunk.hopper.local_precompute import local_precompute
from flash_qla.ops.gated_delta_rule.chunk.hopper.output_fwd import output_fwd


def load_baseline(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seqlen', type=int, default=16384)
    parser.add_argument('--partitions', type=int, default=8)
    parser.add_argument('--flashinfer', type=Path, default=Path('/tmp/flashinfer-gdn-compare'))
    parser.add_argument('--output', type=Path, default=Path('traces/gdn_stages/results.json'))
    parser.add_argument('--baseline-local', type=Path)
    parser.add_argument('--baseline-output', type=Path)
    args = parser.parse_args()
    sys.path.insert(0, str(args.flashinfer))
    from flashinfer.gdn_kernels.delta_rule_dsl.delta_rule_cp_sm90 import (
        cp_delta_rule_t_precompute_dsl_sm90 as fi_local,
        cp_delta_rule_prefill_dsl_sm90 as fi_output,
    )
    torch.manual_seed(42)
    torch.set_num_threads(4)
    torch.cuda.set_stream(torch.cuda.Stream())
    t, parts = args.seqlen, args.partitions
    q, k = [torch.nn.functional.normalize(torch.randn(1, t, 16, 128, device='cuda'), dim=-1).bfloat16() for _ in range(2)]
    v = torch.randn(1, t, 48, 128, device='cuda', dtype=torch.bfloat16)
    g = torch.nn.functional.logsigmoid(torch.randn(1, t, 48, device='cuda')) / 16
    beta = torch.randn_like(g).sigmoid()
    alpha, cu = g[0].exp(), torch.tensor([0, t], device='cuda', dtype=torch.int32)
    ab, gc = local_precompute(k, beta, g)
    fit = fi_local(k[0], beta[0], cu, t, max_seqlen=t)
    h0 = torch.zeros(parts, 48, 128, 128, device='cuda')
    fio = torch.empty_like(v[0])
    def run_fi_output():
        fi_output(fio, None, q[0], k[0], v[0], fit, h0, alpha,
                  128**-.5, cu, t, cp_chunk_len=t//parts, max_seqlen=t)
        return fio[None]
    local = {'optimized': lambda: local_precompute(k, beta, g),
             'flashinfer': lambda: fi_local(k[0], beta[0], cu, t, max_seqlen=t)}
    output = {'optimized': lambda: output_fwd(q, k, v, ab, gc, h0, t//parts)[0],
              'flashinfer': run_fi_output}
    if args.baseline_local:
        old = load_baseline('old_local', args.baseline_local)
        local['baseline'] = lambda: old.local_precompute(k, beta, g)
    if args.baseline_output:
        old_output = load_baseline('old_output', args.baseline_output)
        output['baseline'] = lambda: old_output.output_fwd(q, k, v, ab, gc, h0, t//parts)[0]
    reference = output['flashinfer']().clone()
    checks = {name: error(fn(), reference) for name, fn in output.items()}
    assert max(checks.values()) < .012, checks
    results = {'shape': [1,t,16,48,128], 'partitions':parts,
               'scope':'local includes gate cumsum for FlashQLA; output uses zero incoming states for each partition',
               'output_rms_vs_flashinfer':checks}
    for name, functions in [('local',local),('output',output)]:
        print('TIMING', name, flush=True)
        results[name] = benchmark(functions, inner_repeats=10 if t<=16384 else 2)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2)+'\n')
    print(json.dumps(results,indent=2), flush=True)


if __name__ == '__main__':
    main()

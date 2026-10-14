"""Compare M/N pipeline schedules using the same prepared inputs."""
import argparse
import importlib.util
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bench_prepare_mn import benchmark, error
from flash_qla.ops.gated_delta_rule.chunk.hopper.local_precompute import local_precompute


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seqlen', type=int, default=16384)
    parser.add_argument('--partitions', type=int, nargs='+', default=[8])
    parser.add_argument('--kernel', type=Path, required=True)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--variants', nargs='+', default=['2:0','2:1','3:0','3:1'],
                        help="Pipeline-stage:overlap pairs; use 2:1 for a fixed two-stage candidate")
    args = parser.parse_args()
    new, old = load_module('mn_candidate', args.kernel), load_module('mn_baseline', args.baseline)
    torch.manual_seed(42)
    torch.set_num_threads(4)
    torch.cuda.set_stream(torch.cuda.Stream())
    t = args.seqlen
    k = torch.nn.functional.normalize(torch.randn(1, t, 16, 128, device='cuda'), dim=-1).bfloat16()
    v = torch.randn(1, t, 48, 128, device='cuda', dtype=torch.bfloat16)
    g = torch.nn.functional.logsigmoid(torch.randn(1, t, 48, device='cuda')) / 16
    beta = torch.randn_like(g).sigmoid()
    a, gc = local_precompute(k, beta, g)
    configs = []
    for value in args.variants:
        stages, overlap = map(int, value.split(':'))
        if stages not in (2,3) or overlap not in (0,1):
            parser.error('variants must be 2:0, 2:1, 3:0, or 3:1')
        configs.append((stages, bool(overlap)))
    results = {'shape':[1,t,16,48,128], 'scope':'M/N only, full history, folded beta, BF16 summaries, skip last',
               'kernel_file':str(args.kernel), 'baseline_file':str(args.baseline), 'partitions':{}}
    for parts in args.partitions:
        common = dict(skip_last=True, beta_folded=True)
        def baseline():
            return old.prepare_mn(k,v,a,gc,beta,t//parts,**common)
        functions = {'baseline':baseline}
        for stages, overlap in configs:
            def run(stages=stages, overlap=overlap):
                return new.prepare_mn(k,v,a,gc,beta,t//parts,
                    pipeline_stages=stages, overlap_gemm=overlap, **common)
            functions[f's{stages}_overlap{int(overlap)}'] = run
        rn, rm = baseline()
        checks = {}
        for name, fn in functions.items():
            print('CHECK',parts,name,flush=True)
            n,m = fn()
            checks[name] = {'N_rms':error(n,rn),'M_rms':error(m,rm),'finite':bool(n.isfinite().all() and m.isfinite().all())}
            assert checks[name]['finite'] and max(checks[name]['N_rms'],checks[name]['M_rms']) < .003, checks[name]
        results['partitions'][parts] = {'checks':checks,**benchmark(functions,inner_repeats=5 if t<=16384 else 1)}
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(results,indent=2)+'\n')
        print(json.dumps(results['partitions'][parts],indent=2),flush=True)


if __name__ == '__main__':
    main()

"""Compare shared scratch lifetimes and compile-time N-only suffix preparation."""
import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bench_prepare_mn import benchmark, error
from flash_qla.ops.gated_delta_rule.chunk.hopper.prepare_mn import prepare_mn
from flash_qla.ops.gated_delta_rule.chunk.hopper.local_precompute import local_precompute
from flash_qla.ops.gated_delta_rule.chunk.hopper.cp_fwd import get_warmup_chunks


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seqlen', type=int, default=16384)
    parser.add_argument('--partitions', type=int, default=8)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--retained-chunks', type=int,
                        help="Optional explicit suffix length to measure sustained N-only throughput")
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location('mn_before_lifetimes', args.baseline)
    old = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old)
    torch.manual_seed(42)
    torch.set_num_threads(4)
    torch.cuda.set_stream(torch.cuda.Stream())
    t, p = args.seqlen, args.partitions
    k = torch.nn.functional.normalize(torch.randn(1,t,16,128,device='cuda'),dim=-1).bfloat16()
    v = torch.randn(1,t,48,128,device='cuda',dtype=torch.bfloat16)
    g = torch.nn.functional.logsigmoid(torch.randn(1,t,48,device='cuda'))/16
    beta = torch.randn_like(g).sigmoid()
    a,gc = local_precompute(k,beta,g)
    boundaries = torch.arange(p+1,device='cuda',dtype=torch.int32)*(t//p)
    counts,mask = get_warmup_chunks(gc,boundaries,torch.zeros(p,device='cuda',dtype=torch.bool),threshold=math.log(5e-5))
    if args.retained_chunks is not None:
        assert 1 <= args.retained_chunks <= t // p // 64
        counts = torch.full_like(counts,args.retained_chunks)
        mask = torch.zeros_like(mask)
    base_args = dict(skip_last=True,beta_folded=True)
    suffix_args = dict(num_warmup_chunks=counts,transfer_mask=mask)
    results = {'shape':[1,t,16,48,128],'partitions':p,'threshold':5e-5,
               'explicit_retained_chunks':args.retained_chunks,'measurements':{}}
    for scope,extra in [('full',{}),('suffix',suffix_args)]:
        def baseline():return old.prepare_mn(k,v,a,gc,beta,t//p,**base_args,**extra)
        functions = {'baseline':baseline,
            'reuse':lambda:prepare_mn(k,v,a,gc,beta,t//p,**base_args,**extra),
            'lifetime':lambda:prepare_mn(k,v,a,gc,beta,t//p,lifetime_control=True,**base_args,**extra)}
        if scope=='suffix':
            functions['n_only'] = lambda:prepare_mn(k,v,a,gc,beta,t//p,split_n_only=True,**base_args,**extra)
        rn,rm = baseline()
        checks={}
        for name,fn in functions.items():
            print('CHECK',scope,name,flush=True)
            n,m=fn();checks[name]={'N_rms':error(n,rn),'M_rms':error(m,rm),'finite':bool(n.isfinite().all() and m.isfinite().all())}
            assert checks[name]['finite'] and max(checks[name]['N_rms'],checks[name]['M_rms'])<.003,checks[name]
        results['measurements'][scope]={'checks':checks,**benchmark(functions,inner_repeats=5 if t<=16384 else 1)}
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(results,indent=2)+'\n')
        print(json.dumps(results['measurements'][scope],indent=2),flush=True)


if __name__=='__main__':main()

"""Matched BF16 state reconstruction; same inverse, partitions, and incoming state."""
import argparse,json,sys
from pathlib import Path
import torch
import torch.nn.functional as F

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from bench_implicit_bwd import interleaved,rms
from flash_qla.ops.gated_delta_rule.chunk.hopper.state_reconstruct import reconstruct_states
from flash_qla.ops.gated_delta_rule.chunk.hopper.prepare_h import fused_gdr_h
from flash_qla.ops.gated_delta_rule.chunk.hopper.kkt_solve import kkt_solve

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seqlens',nargs='+',type=int,default=[16384,131072])
    parser.add_argument('--output',type=Path,default=Path('traces/gdn_training_next/states.json'))
    args=parser.parse_args()
    torch.set_num_threads(4);torch.cuda.set_stream(torch.cuda.Stream())
    args.output.parent.mkdir(parents=True,exist_ok=True)
    results={}
    for tokens in args.seqlens:
        torch.manual_seed(907)
        hk,hv,parts=16,48,8
        k=F.normalize(torch.randn(1,tokens,hk,128,device='cuda'),dim=-1).bfloat16()
        v=torch.randn(1,tokens,hv,128,device='cuda',dtype=torch.bfloat16)
        b=torch.rand(1,tokens,hv,device='cuda')
        g=torch.full_like(b,-.001).reshape(1,tokens//64,64,hv).cumsum(2).reshape_as(b)
        a=kkt_solve(k,b)
        h0=torch.randn(parts,hv,128,128,device='cuda')*.01
        cu=torch.arange(parts+1,device='cuda',dtype=torch.int32)*(tokens//parts)
        def baseline():
            return fused_gdr_h(k,v,a,g,b,initial_state=h0,output_final_state=False,output_h=True,cu_seqlens=cu)[0]
        expected=baseline()
        functions={'original':baseline}
        checks={}
        for tile,stages in ((64,1),(64,2),(64,3),(32,2)):
            name=f'v{tile}_stages{stages}'
            fn=lambda tile=tile,stages=stages:reconstruct_states(k,v,a,g,b,h0,parts=parts,stages=stages,value_tile=tile)
            actual=fn();error=rms(actual,expected)
            assert torch.isfinite(actual).all() and error<.003,error
            checks[name]=error;functions[name]=fn
            del actual
        del expected
        timing=interleaved(functions,5 if tokens==16384 else 2)
        results[str(tokens)]={'gradient_input_state_rms':checks,'timing':timing}
        args.output.write_text(json.dumps(results,indent=2)+'\n')
        print(tokens,{n:v['median_ms'] for n,v in timing.items()},checks,flush=True)
        del k,v,b,g,a,h0,cu,baseline,functions,fn;torch.cuda.empty_cache()

if __name__=='__main__':main()

"""State preparation pipeline-depth ablations with unchanged math and caches."""
import argparse,functools,importlib,json,sys
from pathlib import Path
import torch
import torch.nn.functional as F

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from bench_implicit_bwd import interleaved,rms
from bench_fair_gdn_backward import load_module
from flash_qla.ops.gated_delta_rule.chunk.hopper.kkt_solve import kkt_solve
hm=importlib.import_module('flash_qla.ops.gated_delta_rule.chunk.hopper.prepare_h')
dm=importlib.import_module('flash_qla.ops.gated_delta_rule.chunk.hopper.cp_bwd')

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seqlens',nargs='+',type=int,default=[16384,131072])
    parser.add_argument('--output',type=Path,default=Path('traces/gdn_training_next/prepare_tuning.json'))
    args=parser.parse_args();args.output.parent.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4);torch.cuda.set_stream(torch.cuda.Stream());result={}
    reference=load_module(ROOT/'traces/gdn_training_next/prepare_h_reference.py','flash_qla.ops.gated_delta_rule.chunk.hopper._prepare_h_reference')
    for tokens in args.seqlens:
        torch.manual_seed(908);hk,hv,p=16,48,8
        q,k=[F.normalize(torch.randn(1,tokens,hk,128,device='cuda'),dim=-1).bfloat16() for _ in range(2)]
        v=torch.randn(1,tokens,hv,128,device='cuda',dtype=torch.bfloat16);do=torch.randn_like(v)
        b=torch.rand(1,tokens,hv,device='cuda');g=torch.full_like(b,-.001).reshape(1,tokens//64,64,hv).cumsum(2).reshape_as(b)
        a=kkt_solve(k,b);h0=torch.randn(p,hv,128,128,device='cuda')*.01
        cu=torch.arange(p+1,device='cuda',dtype=torch.int32)*(tokens//p)
        warm=torch.full((p,hv),tokens//p//64,device='cuda',dtype=torch.int32)
        entry=result[str(tokens)]={}
        for module,factory_name,label in ((hm,'tilelang_prepare_h','h'),(dm,'tilelang_prepare_dh_ws','dh')):
            factory=getattr(module,factory_name)
            def call(stages):
                setattr(module,factory_name,functools.partial(factory,num_stages=stages))
                try:
                    if label=='h':return hm.fused_gdr_h(k,v,a,g,b,initial_state=h0,output_final_state=False,output_h=True,cu_seqlens=cu,num_stages=stages)[0]
                    return dm.fused_gdr_dh_ws(q,k,a,g,b,do,output_dh=False,cu_seqlens=cu,num_warmup_chunks=warm)[1]
                finally:setattr(module,factory_name,factory)
            def original_h():return reference.fused_gdr_h(k,v,a,g,b,initial_state=h0,output_final_state=False,output_h=True,cu_seqlens=cu)[0]
            expected=original_h() if label=='h' else call(2)
            checks={};valid=[]
            for s in (1,2,3):
                try:actual=call(s)
                except Exception as exc:
                    checks[str(s)]={'unsupported':str(exc)}
                    continue
                checks[str(s)]=rms(actual,expected)
                assert checks[str(s)]<.001 and torch.isfinite(actual).all()
                valid.append(s)
                del actual
            del expected
            functions={str(s):lambda s=s:call(s) for s in valid}
            if label=='h':functions['reference']=original_h
            timing=interleaved(functions,5 if tokens==16384 else 2)
            entry[label]={'relative_rms':checks,'timing':timing}
            print(tokens,label,{s:v['median_ms'] for s,v in timing.items()},flush=True)
            del call,functions,original_h;torch.cuda.empty_cache()
        args.output.write_text(json.dumps(result,indent=2)+'\n')
        del q,k,v,do,b,g,a,h0,cu,warm;torch.cuda.empty_cache()

if __name__=='__main__':main()

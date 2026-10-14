"""Tune explicit cuBLASLt BF16 input/output projections for the 16k GDN layer."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import torch
import torch.nn.functional as F
from bench_gdn_layer import Layer
from bench_prepare_mn import benchmark,error
from bench_gdn_preparation import paired_benchmark
from flash_qla.ops.gated_delta_rule.chunk.hopper.projection import ProjectionPlan


def torch_lt(x,w):
    previous=torch.backends.cuda.preferred_blas_library()
    torch.backends.cuda.preferred_blas_library("cublaslt")
    try:return F.linear(x,w)
    finally:torch.backends.cuda.preferred_blas_library(previous)


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--algorithms',type=int,default=24)
    parser.add_argument('--output',type=Path,default=Path('traces/qwen38_projection_tuning_16k'))
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    torch.manual_seed(42);torch.set_num_threads(4);torch.cuda.set_stream(torch.cuda.Stream())
    config=SimpleNamespace(seqlen=16384,hidden_size=2560,key_heads=16,value_heads=48,
        conv_size=4,norm_eps=1e-6,dt_max=.01,partitions=8,conv_backend='dao',
        gdn_backend='optimized',projection_mode='packed',norm_gate_mode='fused',preparation_mode='fused_gates')
    layer=Layer(config)
    p=layer.project();n,g=layer.prepare_qkv(p);o=layer.gdn(n,g);gated=layer.norm_gate(o,p)
    inputs={'input':(layer.x,layer.packed_weight),'output':(gated.view(16384,-1),layer.wo)}
    result={'config':vars(config),'gpu':torch.cuda.get_device_name(),'scope':'same synthetic weights; BF16 operands/outputs, FP32 accumulation; full history, dt_max .01','projections':{}}
    winners={}
    for name,(x,w) in inputs.items():
        reference=F.linear(x,w)
        functions={'torch_cublas':lambda x=x,w=w:F.linear(x,w), 'torch_lt':lambda x=x,w=w:torch_lt(x,w)}
        plans={};weights={};checks={}
        for column in (False,True):
            for transpose in (False,True):
                plan=ProjectionPlan(x.shape[0],w.shape[0],w.shape[1],x.device,weight_transposed=transpose,column_order=column,algorithms=args.algorithms)
                pw=w.T.contiguous() if transpose else w
                buf=torch.empty_like(reference)
                prefix=('lt_col_' if column else 'lt_row_')+('nn' if transpose else 'nt')
                plans[prefix]=plan;weights[prefix]=pw
                for i in range(len(plan.results)):
                    key=f'{prefix}_{i}'
                    fn=lambda plan=plan,pw=pw,i=i,buf=buf,x=x:plan.run(x,pw,i,out=buf)
                    try:
                        y=fn();torch.cuda.synchronize();checks[key]=error(y,reference)
                        assert checks[key]<.003,(key,checks[key])
                        functions[key]=fn
                    except RuntimeError as exc:checks[key]=str(exc)
        print('Tune',name,len(functions),flush=True)
        times=benchmark(functions)
        best=min(times,key=lambda key:times[key]['median_ms'])
        if best=='torch_cublas':winners[name]=None
        elif best=='torch_lt':winners[name]='torch_lt'
        else:
            prefix,index=best.rsplit('_',1);plan=plans[prefix];plan.best=int(index)
            winners[name]=(plan,weights[prefix])
        result['projections'][name]={'shape':[x.shape[0],w.shape[0],w.shape[1]],'best':best,'timings':times,'validation':checks,
            'algorithm_metadata':{prefix:[{'workspace_bytes':r.workspace,'waves':r.waves,'opaque_data':list(r.algo.data)} for r in plan.results] for prefix,plan in plans.items()}}
        print(name,best,times[best],flush=True)
    baseline=layer.forward()
    original_project=layer.project;original_output=layer.output_project
    def tuned_project(mode=None):
        if winners['input'] is None:return original_project(mode)
        if winners['input']=='torch_lt':return torch_lt(layer.x,layer.packed_weight).split(layer.projection_sizes,dim=-1)
        plan,w=winners['input'];return plan.run(layer.x,w).split(layer.projection_sizes,dim=-1)
    def tuned_output(gated):
        if winners['output'] is None:return original_output(gated)
        if winners['output']=='torch_lt':return torch_lt(gated.view(16384,-1),layer.wo)
        plan,w=winners['output'];return plan.run(gated.view(16384,-1),w)
    def tuned_layer():
        layer.project=tuned_project;layer.output_project=tuned_output
        try:return layer.forward()
        finally:layer.project=original_project;layer.output_project=original_output
    candidate=tuned_layer();result['layer_relative_rms']=error(candidate,baseline)
    assert candidate.isfinite().all() and result['layer_relative_rms']<.012,result['layer_relative_rms']
    result['layer_comparison']=paired_benchmark({'torch':layer.forward,'tuned':tuned_layer})
    (args.output/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    print('Layer',result['layer_relative_rms'],{k:v['median_ms'] for k,v in result['layer_comparison']['timings'].items()},result['layer_comparison']['median_paired_saving_ms'],flush=True)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
        for name,fn in [('torch',layer.forward),('tuned',tuned_layer)]:
            with torch.profiler.record_function(name):fn()
        torch.cuda.synchronize()
    prof.export_chrome_trace(str(args.output/'layer_trace.json'))


if __name__=='__main__':main()

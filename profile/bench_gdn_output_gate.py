"""Prototype GDN output epilogue vs existing output plus standalone norm/gate."""
import argparse
import importlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import torch
from bench_gdn_layer import Layer
from bench_prepare_mn import benchmark,error
from bench_gdn_preparation import paired_benchmark
import importlib.util


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seqlen',type=int,default=16384)
    parser.add_argument('--output',type=Path,default=Path('traces/qwen38_output_gate_experiment_16k'))
    parser.add_argument('--kernel-file',type=Path,default=Path(__file__).with_name('output_fwd_gate_parallel_experiment.py'))
    args=parser.parse_args()
    spec=importlib.util.spec_from_file_location('output_gate_candidate',args.kernel_file)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    fused_output=module.output_fwd
    args.output.mkdir(parents=True,exist_ok=True)
    torch.manual_seed(42);torch.set_num_threads(4);torch.cuda.set_stream(torch.cuda.Stream())
    config=SimpleNamespace(seqlen=args.seqlen,hidden_size=2560,key_heads=16,value_heads=48,
        conv_size=4,norm_eps=1e-6,dt_max=.01,partitions=8,conv_backend='dao',
        gdn_backend='optimized',projection_mode='packed',norm_gate_mode='fused',preparation_mode='fused_gates')
    layer=Layer(config)
    fast=importlib.import_module('flash_qla.ops.gated_delta_rule.chunk.hopper.fast_fwd')
    original_output=fast.output_fwd
    p=layer.project();n,g=layer.prepare_qkv(p)
    captured={}
    def save_output(*a,**kw):
        captured['args']=a;captured['kw']=kw
        return original_output(*a,**kw)
    with patch.object(fast,'output_fwd',save_output):o=layer.gdn(n,g)
    reference=layer.norm_gate(o,p)
    def fused_stage():
        return fused_output(*captured['args'],**captured['kw'],gate=p[3],norm_weight=layer.norm_weight,norm_eps=config.norm_eps)[0]
    print('Compile/check fused epilogue',flush=True)
    candidate=fused_stage()
    rms=error(candidate,reference);assert candidate.isfinite().all() and rms<.006,rms
    print('Stage error',rms,flush=True)
    baseline_stage=lambda:layer.norm_gate(original_output(*captured['args'],**captured['kw'])[0],p)
    timing=benchmark({'separate':baseline_stage,'fused':fused_stage})
    original_project=layer.project
    original_norm=layer.norm_gate
    current={}
    def project(*a,**kw):
        pp=original_project(*a,**kw);current['gate']=pp[3];return pp
    def output(*a,**kw):return fused_output(*a,**kw,gate=current['gate'],norm_weight=layer.norm_weight,norm_eps=config.norm_eps)
    def fused_layer():
        with patch.object(layer,'project',project),patch.object(layer,'norm_gate',lambda o,p,mode=None:o),patch.object(fast,'output_fwd',output):
            return layer.forward()
    baseline=layer.forward();final=fused_layer();layer_error=error(final,baseline)
    assert final.isfinite().all() and layer_error<.008,layer_error
    comparisons=paired_benchmark({'separate':layer.forward,'fused':fused_layer})
    results={'stage_relative_rms':rms,'layer_relative_rms':layer_error,'stage_timings':timing,'paired_layer':comparisons,'scope':'same BF16 fixture; full history; dt_max .01; original GDN math'}
    (args.output/'results.json').write_text(json.dumps(results,indent=2)+'\n')
    print(json.dumps({k:v for k,v in results.items() if k!='paired_layer'},indent=2),flush=True)
    print('Layer',{k:v['median_ms'] for k,v in comparisons['timings'].items()},comparisons['median_paired_saving_ms'],flush=True)


if __name__=='__main__':main()

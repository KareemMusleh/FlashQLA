"""Compare eager, torch.compile and existing Triton output norm/SiLU gating."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import torch
from bench_gdn_layer import Layer
from bench_prepare_mn import benchmark,error
from bench_gdn_preparation import paired_benchmark


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('traces/qwen38_compiled_output_gate_16k'))
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    torch.manual_seed(42);torch.set_num_threads(4);torch.cuda.set_stream(torch.cuda.Stream())
    config=SimpleNamespace(seqlen=16384,hidden_size=2560,key_heads=16,value_heads=48,
        conv_size=4,norm_eps=1e-6,dt_max=.01,partitions=8,conv_backend='dao',
        gdn_backend='optimized',projection_mode='packed',norm_gate_mode='fused',preparation_mode='fused_gates')
    layer=Layer(config)
    p=layer.project();n,g=layer.prepare_qkv(p);o=layer.gdn(n,g)
    reference=layer.norm_gate(o,p,'eager')
    functions={mode:lambda mode=mode:layer.norm_gate(o,p,mode) for mode in ('eager','fused','compiled')}
    print('Compile and validate output gate',flush=True)
    checks={mode:error(fn(),reference) for mode,fn in functions.items()}
    assert max(checks.values())<.006,checks
    timings=benchmark(functions)
    baseline=layer.forward()
    compiled=lambda:layer.forward(norm_gate_mode='compiled')
    layer_error=error(compiled(),baseline)
    assert layer_error<.008,layer_error
    comparison=paired_benchmark({'triton':layer.forward,'compiled':compiled})
    result={'config':vars(config),'stage_relative_rms_vs_eager':checks,'stage_timings':timings,
            'layer_relative_rms_vs_triton':layer_error,'paired_layer':comparison,
            'scope':'same 16k synthetic BF16 fixture; full history; dt_max .01; rounding preserved'}
    print('stages',{k:v['median_ms'] for k,v in timings.items()},flush=True)
    print('layer',{k:v['median_ms'] for k,v in comparison['timings'].items()},comparison['median_paired_saving_ms'],flush=True)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
        for mode,fn in functions.items():
            with torch.profiler.record_function('norm_gate_'+mode):fn()
        torch.cuda.synchronize()
    path=args.output/'gate_trace.json';prof.export_chrome_trace(str(path))
    events=json.loads(path.read_text())['traceEvents']
    launches={e['args']['correlation']:e for e in events if e.get('cat') in ('cuda_runtime','cuda_driver') and 'correlation' in e.get('args',{})}
    result['stage_kernel_counts']={}
    for mode in functions:
        root=next(e for e in events if e.get('name')=='norm_gate_'+mode and e.get('ph')=='X')
        result['stage_kernel_counts'][mode]=sum(e.get('cat')=='kernel' and e.get('args',{}).get('correlation') in launches and root['ts']<=launches[e['args']['correlation']]['ts']<root['ts']+root['dur'] for e in events)
    (args.output/'results.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()

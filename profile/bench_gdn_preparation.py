"""Matched Qwen3.8 16k comparison of CUDA conv/L2 normalization and fused prep."""
import argparse
import hashlib
import json
import statistics
from pathlib import Path
from types import SimpleNamespace

import torch
from bench_gdn_layer import Layer
from bench_prepare_mn import benchmark, error
from flash_qla.ops.gated_delta_rule.chunk.hopper.prepare_qkv import prepare_qkv_fwd


def paired_benchmark(functions, samples=120):
    """Record both paths back-to-back in one graph, alternating their order.

    Within-graph CUDA events keep both measurements close in time when GPU
    clocks drift during long GEMM-heavy replay bursts.
    """
    captures=[]
    names=list(functions)
    for order in (names,list(reversed(names))):
        events={name:(torch.cuda.Event(enable_timing=True,external=True),torch.cuda.Event(enable_timing=True,external=True)) for name in order}
        graph=torch.cuda.CUDAGraph()
        outputs={}
        with torch.cuda.graph(graph):
            for name in order:
                events[name][0].record()
                outputs[name]=functions[name]()
                events[name][1].record()
        captures.append((graph,events,outputs))
    for _ in range(10):
        for graph,_,_ in captures:graph.replay()
    torch.cuda.synchronize()
    times={name:[] for name in names}
    differences=[]
    for i in range(samples):
        graph,events,_=captures[i%2]
        graph.replay()
        torch.cuda.synchronize()
        for name,(start,end) in events.items():times[name].append(start.elapsed_time(end))
        differences.append(times[names[0]][-1]-times[names[1]][-1])
    return {'timings':{name:{'median_ms':statistics.median(values),'samples_ms':values} for name,values in times.items()},
            'median_paired_saving_ms':statistics.median(differences),
            'mean_paired_saving_ms':statistics.mean(differences)}


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seqlen',type=int,default=16384)
    parser.add_argument('--output',type=Path,default=Path('traces/qwen38_flash_next_preparation_16k'))
    parser.add_argument('--final-only',action='store_true',help='Compare only CUDA baseline and fully fused preparation')
    parser.add_argument('--layer-inner-repeats',type=int,default=1)
    parser.add_argument('--paired-timing',action='store_true')
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    torch.manual_seed(42)
    torch.set_num_threads(4)
    torch.cuda.set_stream(torch.cuda.Stream())
    config=SimpleNamespace(seqlen=args.seqlen,hidden_size=2560,key_heads=16,value_heads=48,
                           conv_size=4,norm_eps=1e-6,dt_max=.01,partitions=8,
                           conv_backend='dao',gdn_backend='optimized',projection_mode='packed',
                           norm_gate_mode='fused',preparation_mode='separate')
    layer=Layer(config)
    p=layer.project()
    reference_prep=layer.prepare_qkv(p,'separate')
    baseline=lambda:layer.prepare_qkv(p,'separate')
    functions={'cuda_conv_and_l2':baseline}
    checks={}
    print('Validate and tune fused preparation',flush=True)
    for bt in ((32,) if args.final_only else (8,16,32,64)):
        for warps in ((4,) if args.final_only else (4,8)):
            name=f'tokens{bt}_warps{warps}'
            fn=lambda bt=bt,warps=warps:prepare_qkv_fwd(p,layer.conv_weights,16,48,tokens_per_cta=bt,num_warps=warps)
            candidate=fn()
            checks[name]=[error(a,b) for a,b in zip(candidate,reference_prep)]
            assert max(checks[name])<.004,checks
            functions[name]=fn
    timings=benchmark(functions)
    best=min((n for n in timings if n!='cuda_conv_and_l2'),key=lambda n:timings[n]['median_ms'])
    print('Best',best,timings[best],flush=True)
    print('Validate and benchmark complete layer',flush=True)
    reference_gates=layer.gates(p)
    qkv_fused,gates_fused=layer.prepare_qkv(p,'fused_gates')
    checks['fused_gates']=[error(a,b) for a,b in zip(gates_fused,reference_gates)]
    assert max(checks['fused_gates'])<1e-6,checks
    timings.update(benchmark({'cuda_conv_l2_and_gates':lambda:(baseline(),layer.gates(p)),
                              'fused_qkv_and_gates':lambda:layer.prepare_qkv(p,'fused_gates')}))
    variants={'packed_cuda':('packed','separate'),'packed_fused':('packed','fused'),
              'packed_fused_gates':('packed','fused_gates'),
              'separate_cuda':('separate','separate'),'separate_fused':('separate','fused')}
    if args.final_only:
        variants={name:variant for name,variant in variants.items() if name in ('packed_cuda','packed_fused_gates')}
    reference=layer.forward()
    standard=layer.forward(backend='default')
    layers={}
    validation={}
    for name,(projection,prep) in variants.items():
        fn=lambda projection=projection,prep=prep:layer.forward(projection_mode=projection,preparation_mode=prep)
        result=fn()
        validation[name]={'relative_rms_vs_cuda':error(result,reference),
                          'relative_rms_vs_standard_gdn':error(result,standard),'finite':bool(result.isfinite().all())}
        assert validation[name]['finite'] and validation[name]['relative_rms_vs_cuda']<.008 and validation[name]['relative_rms_vs_standard_gdn']<.012,validation
        layers[name]=fn
    output={'config':vars(config),'gpu':torch.cuda.get_device_name(),'scope':'same weights/inputs; full history; mild synthetic decay; fused output gate in every variant',
            'stage_validation':checks,'stage_timings':timings,'best_launch':best,
            'layer_validation':validation,'layer_timings':benchmark(layers,inner_repeats=args.layer_inner_repeats),
            'layer_inner_repeats':args.layer_inner_repeats,'sources':{}}
    if args.paired_timing:
        output['paired_comparison']=paired_benchmark({name:layers[name] for name in ('packed_cuda','packed_fused_gates')})
    for path in [Path(__file__),Path(__file__).with_name('bench_gdn_layer.py'),Path(__file__).resolve().parents[1]/'flash_qla/ops/gated_delta_rule/chunk/hopper/prepare_qkv.py']:
        output['sources'][str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
    print(json.dumps(output,indent=2),flush=True)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],record_shapes=True) as prof:
        for name,(projection,prep) in variants.items():
            with torch.profiler.record_function(name):
                layer.forward(projection_mode=projection,preparation_mode=prep,labels=True)
        torch.cuda.synchronize()
    trace_path=args.output/'layer_trace.json'
    prof.export_chrome_trace(str(trace_path))
    events=json.loads(trace_path.read_text())['traceEvents']
    launches={e['args']['correlation']:e for e in events if e.get('cat') in ('cuda_runtime','cuda_driver') and 'correlation' in e.get('args',{})}
    output['profile_kernel_counts']={}
    for name in variants:
        root=next(e for e in events if e.get('name')==name and e.get('ph')=='X')
        output['profile_kernel_counts'][name]=sum(e.get('cat')=='kernel' and e.get('args',{}).get('correlation') in launches and root['ts']<=launches[e['args']['correlation']]['ts']<root['ts']+root['dur'] for e in events)
    (args.output/'results.json').write_text(json.dumps(output,indent=2)+'\n')


if __name__=='__main__':
    main()

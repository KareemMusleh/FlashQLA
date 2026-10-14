import os,sys,json,statistics,subprocess
from pathlib import Path
sys.path.insert(0,'/tmp/flashinfer-gdn-compare')
sys.path.insert(0,'/home/kareemm/forks/FlashQLA')
import torch
import importlib
from functools import partial
from flash_qla.ops.gated_delta_rule.chunk.cp.context import _build_intra_cp_context
chunk_module=importlib.import_module("flash_qla.ops.gated_delta_rule.chunk")
original_warmup=chunk_module.get_warmup_chunks
import flashinfer
import flash_qla
from flashinfer.gdn_prefill import chunk_gated_delta_rule as fi
from flash_qla import chunk_gated_delta_rule as qla
from flash_qla.utils.math import l2norm_fwd
out=Path('/home/kareemm/forks/FlashQLA/traces/flashinfer_matched_partitions_16k');out.mkdir(parents=True,exist_ok=True)
torch.manual_seed(42);torch.set_num_threads(4);torch.cuda.set_stream(torch.cuda.Stream())
T=16384
q,k=[torch.randn(1,T,16,128,device='cuda',dtype=torch.bfloat16) for _ in range(2)]
v=torch.randn(1,T,48,128,device='cuda',dtype=torch.bfloat16)
g=torch.nn.functional.logsigmoid(torch.randn(1,T,48,device='cuda'))/16
beta=torch.randn_like(g).sigmoid();alpha=g.exp();cu=torch.tensor([0,T],device='cuda',dtype=torch.int32)
qn,kn=[l2norm_fwd(x)[0] for x in (q,k)]
contexts={p:_build_intra_cp_context(cu,64,[T//64],T//p//64) for p in [2,8]}
results={'flashinfer_commit':subprocess.check_output(['git','-C','/tmp/flashinfer-gdn-compare','rev-parse','HEAD'],text=True).strip(),'flashinfer_path':flashinfer.__file__,'flashqla_commit':subprocess.check_output(['git','-C','/home/kareemm/forks/FlashQLA','rev-parse','HEAD'],text=True).strip(),'torch':torch.__version__,'gpu':torch.cuda.get_device_name(),'visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),'shape':{'batch':1,'tokens':T,'key_heads':16,'value_heads':48,'head_dim':128},'dtype':'BF16 QKV; FP32 gates','scope':'forward-only; no initial/final state; scale=128**-0.5','gate':'FlashQLA logsigmoid(N(0,1))/16; FlashInfer exp(log_gate), conversion outside timing','method':'CUDA graphs; nine rotating-order samples of 30 replays, warmed before capture'}
with torch.no_grad():
 for norm in [True,False]:
  section='including_qk_norm' if norm else 'prenormalized_qk'
  qq,kk=(q,k) if norm else (qn,kn)
  def qlafn(partitions=2,truncate=False):
   chunk_module.get_warmup_chunks=original_warmup if truncate else partial(original_warmup,threshold=float('-inf'))
   return qla(qq,kk,v,g,beta,scale=128**-.5,use_qk_l2norm_in_kernel=norm,output_final_state=False,cp_context=contexts[partitions])[0]
  def fifn(partitions=2):
   return fi(qq[0],kk[0],v[0],alpha[0],beta[0],scale=128**-.5,cu_seqlens=cu,output_final_state=False,use_qk_l2norm_in_kernel=norm,use_cp=True,_cp_chunk_len=T//partitions)[None]
  fns={}
  for partitions in [2,8]:
   fns[f'flashqla_full_p{partitions}']=partial(qlafn,partitions,False)
   fns[f'flashinfer_p{partitions}']=partial(fifn,partitions)
   fns[f'flashqla_truncated_p{partitions}']=partial(qlafn,partitions,True)
  graphs={};outputs={};entry=results[section]={}
  ref=qla(qq,kk,v,g,beta,scale=128**-.5,use_qk_l2norm_in_kernel=norm,output_final_state=False,auto_cp=False)[0].clone()
  for name,fn in fns.items():
   print('WARMUP',section,name,flush=True)
   y=fn();torch.cuda.synchronize()
   a,b=ref.float(),y.float()
   err={'relative_rms':((a-b).square().mean()/a.square().mean()).sqrt().item(),'max_abs':(a-b).abs().max().item(),'finite':bool(y.isfinite().all())}
   print('CHECK',err,flush=True)
   assert err['finite'] and err['relative_rms']<.01,err
   entry[name]={'correctness_vs_flashqla_no_cp':err}
   for _ in range(5):fn()
   torch.cuda.synchronize()
   graph=torch.cuda.CUDAGraph()
   with torch.cuda.graph(graph):outputs[name]=fn()
   graphs[name]=graph
  for _ in range(20):
   for graph in graphs.values():graph.replay()
  samples={name:[] for name in fns};names=list(fns)
  for cycle in range(9):
   order=names[cycle%len(names):]+names[:cycle%len(names)]
   if cycle%2:order=order[::-1]
   for name in order:
    start,end=[torch.cuda.Event(enable_timing=True) for _ in range(2)]
    start.record()
    for _ in range(30):graphs[name].replay()
    end.record();end.synchronize();samples[name].append(start.elapsed_time(end)/30)
  for name,fn in fns.items():
   entry[name].update(median_ms=statistics.median(samples[name]),samples_ms=samples[name])
   print('RESULT',section,name,entry[name],flush=True)
   with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
    for _ in range(3):
     with torch.profiler.record_function(name):fn()
    torch.cuda.synchronize()
   trace=out/f'{section}_{name}.json';prof.export_chrome_trace(str(trace))
   kernels=[e for e in json.loads(trace.read_text())['traceEvents'] if e.get('cat')=='kernel']
   entry[name]['kernels_per_iteration']=len(kernels)//3
   entry[name]['kernel_names']=sorted(set(e['name'] for e in kernels))
  (out/'results.json').write_text(json.dumps(results,indent=2)+'\n')
  del graphs,outputs
results['partition_boundaries']={p:c.intra_cp_cu_seqlens.tolist() for p,c in contexts.items()}
results['full_boundary_method']='Benchmark-local get_warmup_chunks threshold=-inf; all needed partition chunks processed, no decay truncation'
chunk_module.get_warmup_chunks=original_warmup
(out/'results.json').write_text(json.dumps(results,indent=2)+'\n')
print('DONE',flush=True)

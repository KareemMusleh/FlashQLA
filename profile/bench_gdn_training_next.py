"""Separate and cumulative full-training ablations for the next GDN optimizations."""
import argparse,gc,hashlib,json,os,sys
from contextlib import contextmanager
from pathlib import Path
import torch
from causal_conv1d import causal_conv1d_fn

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from bench_gdn_full_layer_training import Layer
from bench_gdn_fp8_training import PreparedLinear
from bench_direct_gdn_training import paired_graphs
from bench_fair_gdn_backward import chunk,truncation,load_module
from bench_implicit_bwd import rms
from flash_qla.ops.deepgemm_linear import load_ydt_blockwise,shared_deepgemm_projections
from flash_qla.ops.conv_norm import causal_conv_silu_l2norm
from flash_qla.ops.fp8_gdn_layer import fp8_gdn_layer
from flash_qla.ops.gated_delta_rule.chunk.hopper.fast_fwd import _training_cp_context
from flash_qla.ops.gated_delta_rule.chunk.hopper.prepare_h import fused_gdr_h as workspace_reduced_h

reference_h=load_module(ROOT/'traces/gdn_training_next/prepare_h_reference.py','flash_qla.ops.gated_delta_rule.chunk.hopper._training_next_reference_h').fused_gdr_h

class CandidateLayer(Layer):
    def __init__(self,context,quantizer):
        super().__init__(2560,.001,context)
        self.mode='original'
        self.fused_norm_gate=True
        for name in ('q','k','v','z','a','b','out'):
            module=PreparedLinear(getattr(self,name),quantizer)
            module.enabled=True
            setattr(self,name,module)

    def forward(self,x):
        if self.mode in ('direct','direct_conv','all_direct'):
            linears=(self.q,self.k,self.v,self.z,self.a,self.b,self.out)
            params=[m.weight for m in linears[:6]]+list(self.conv)+[self.A_log,self.dt_bias,self.norm,self.out.weight]
            return fp8_gdn_layer(x.reshape(-1,2560),params,[m.prepared for m in linears],self.context,
                                 fused_conv=self.mode!='direct').reshape_as(x)
        return super().forward(x)

    def input_projections(self,x):
        if self.mode in ('original','conv_only','state'):
            return super().input_projections(x)
        linears=(self.q,self.k,self.v,self.z,self.a,self.b)
        return shared_deepgemm_projections(x,[m.weight for m in linears],[m.prepared for m in linears],
                                            combined_backward=self.mode!='shared')

    def convolution_outputs(self,q,k,v):
        if self.mode in ('conv_only','conv','all'):
            q=causal_conv_silu_l2norm(q,self.conv[0])
            k=causal_conv_silu_l2norm(k,self.conv[1])
            v=causal_conv1d_fn(v.transpose(1,2),self.conv[2],activation='silu').transpose(1,2).contiguous()
            return q,k,v
        return super().convolution_outputs(q,k,v)

@contextmanager
def state_backend(enabled):
    original=chunk.fused_gdr_h
    chunk.fused_gdr_h=workspace_reduced_h if enabled else reference_h
    try:yield
    finally:chunk.fused_gdr_h=original

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seqlens',nargs='+',type=int,default=[16384,131072])
    parser.add_argument('--modes',nargs='+',default=['original','shared','combined','conv_only','conv','state','all'])
    parser.add_argument('--trace',action='store_true')
    parser.add_argument('--output',type=Path,default=Path('traces/gdn_training_next/results.json'))
    args=parser.parse_args();args.output.parent.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4);torch.cuda.set_stream(torch.cuda.Stream())
    qpath=Path('/home/kareemm/arcadia/ml/torch/ydt/core/ydt_core/fp8/quantization/blockwise.py')
    quantizer=load_ydt_blockwise(qpath)
    paths=['flash_qla/ops/deepgemm_linear.py','flash_qla/ops/conv_norm.py','flash_qla/ops/fp8_gdn_layer.py','flash_qla/ops/gated_delta_rule/chunk/hopper/prepare_h.py',str(Path(__file__).relative_to(ROOT))]
    results={'gpu':torch.cuda.get_device_name(),'visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),
             'scope':'Full GDN layer forward/backward, all input/parameter gradients. Same P8, standard GDN forward/cache policy, no decay truncation; prequantized blockwise FP8 weights, runtime quantization/dequantization and BF16 backward GEMMs. Loss/optimizer excluded.',
             'source_sha256':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths},
             'modes':args.modes,'state_backend':'Remove unused transition-matrix workspace; original two-stage recurrence math'}
    for t in args.seqlens:
        torch.manual_seed(442)
        layer=CandidateLayer(_training_cp_context(torch.device('cuda'),t,8),quantizer)
        x=torch.randn(1,t,2560,device='cuda',dtype=torch.bfloat16,requires_grad=True)
        upstream=torch.randn_like(x);inputs=(x,*layer.parameters())
        def forward(mode):
            layer.mode=mode;layer.qk_normalized=mode in ('conv_only','conv','all')
            with truncation(False),state_backend(mode in ('all','all_direct')):
                return layer(x)
        def step(mode):return torch.autograd.grad(forward(mode),inputs,upstream)
        print('SEQLEN',t,flush=True)
        expected_output=forward('original').detach();expected=step('original')
        checks={}
        for mode in args.modes:
            output=forward(mode).detach();actual=step(mode)
            errors={name:rms(a,b) for name,a,b in zip(['input',*[n for n,_ in layer.named_parameters()]],actual,expected)}
            assert all(torch.isfinite(a).all() for a in actual)
            assert max(errors.values())<.02,(mode,errors)
            checks[mode]={'output_relative_rms':rms(output,expected_output),'gradient_relative_rms':errors}
            print('check',mode,checks[mode]['output_relative_rms'],max(errors.values()),flush=True)
            del actual,output
        del expected,expected_output;gc.collect();torch.cuda.empty_cache()
        memory={}
        for mode in args.modes:
            gc.collect();torch.cuda.synchronize();torch.cuda.empty_cache()
            baseline=torch.cuda.memory_allocated();torch.cuda.reset_peak_memory_stats()
            measured=step(mode);torch.cuda.synchronize()
            memory[mode]={'peak_increment_bytes':torch.cuda.max_memory_allocated()-baseline}
            del measured
        timings=paired_graphs({mode:lambda mode=mode:step(mode) for mode in args.modes},samples=21)
        results[str(t)]={'checks':checks,'timing':timings,'step_memory':memory}
        print('timing',{mode:v['median_ms'] for mode,v in timings.items()},flush=True)
        if args.trace:
            gc.collect();torch.cuda.empty_cache()
            for mode in args.modes:
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
                    step(mode);torch.cuda.synchronize()
                prof.export_chrome_trace(str(args.output.parent/f'{t}_{mode}.json'))
        args.output.write_text(json.dumps(results,indent=2)+'\n')
        del layer,x,upstream,inputs,forward,step;gc.collect();torch.cuda.empty_cache()

if __name__=='__main__':main()

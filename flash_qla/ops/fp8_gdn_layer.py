"""Matched-cache blockwise FP8 GDN training with direct projection gradients."""
import copy
import torch
import torch.nn.functional as F
from torch.autograd.function import once_differentiable
import triton

from flash_qla.ops.gdn_layer import _GDRContext,gate_backward_into
from flash_qla.ops.deepgemm_linear import shared_deepgemm_projections,deepgemm_linear
from flash_qla.ops.conv_norm import _conv_norm
from flash_qla.ops.gated_delta_rule.chunk import ChunkGatedDeltaRuleFunction
from flash_qla.ops.gated_delta_rule.chunk.hopper.training_norm_gate import _NormGate,_backward as norm_backward


class _FP8GDN(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,prepared,cp_context,fused_conv,*parameters):
        from causal_conv1d.cpp_functions import causal_conv1d_fwd_function
        wq,wk,wv,wz,wa,wb,cq,ck,cv,al,dt,nw,wo=parameters
        weights=(wq,wk,wv,wz,wa,wb)
        hk,hv=wq.shape[0]//128,wv.shape[0]//128
        p=shared_deepgemm_projections(x[None],weights,prepared[:6])
        p=tuple(t[0] for t in p)
        conv=[];stats=[]
        for i,(z,w) in enumerate(zip(p[:3],(cq,ck,cv))):
            if fused_conv and i<2:
                out=torch.empty_like(z)
                r=torch.empty((1,z.shape[0],hk),device=z.device,dtype=torch.float32)
                _conv_norm[(triton.cdiv(z.shape[0],16),hk)](z,w,out,r,z.shape[0],hk,w.shape[-1],1e-6,16,num_warps=4,enable_fp_fusion=False)
                stats.append(r)
            else:
                out=causal_conv1d_fwd_function(z[None].transpose(1,2),w,None,None,None,None,True).transpose(1,2)[0]
            conv.append(out)
        q=conv[0].view(1,-1,hk,128);k=conv[1].view_as(q);v=conv[2].view(1,-1,hv,128)
        gate_input=p[4].float()+dt
        g=(-al.exp()*F.softplus(gate_input))[None]
        beta=p[5].float().sigmoid()[None]
        inner=_GDRContext()
        o,_=ChunkGatedDeltaRuleFunction.forward(inner,q,k,v,g,beta,
            output_final_state=False,use_qk_l2norm_in_kernel=not fused_conv,
            cp_context=cp_context,enable_fwd_cp_cache=True,is_train=True)
        if fused_conv:
            saved=list(inner.saved_tensors);saved[2:4]=stats;inner.saved_tensors=tuple(saved)
        normctx=_GDRContext()
        gated=_NormGate.forward(normctx,o,p[3].view_as(o),nw,1e-6)
        result=deepgemm_linear(gated.reshape(x.shape[0],-1),wo,prepared[6])
        n,s,r=normctx.saved_tensors[3:]
        ctx.save_for_backward(x,*parameters,*p[:4],gate_input,o,gated,n,s,r,*inner.saved_tensors)
        inner.saved_tensors=None
        ctx.inner=inner;ctx.prepared=prepared
        return result

    @staticmethod
    @once_differentiable
    def backward(ctx,dy):
        from causal_conv1d.cpp_functions import causal_conv1d_bwd_function
        saved=ctx.saved_tensors
        x,wq,wk,wv,wz,wa,wb,cq,ck,cv,al,dt,nw,wo=saved[:14]
        p=saved[14:18]
        gate_input,o,gated,n,s,r=saved[18:24]
        inner=copy.copy(ctx.inner);inner.saved_tensors=saved[24:]
        ctx.maybe_clear_saved_tensors()
        del saved
        beta=inner.saved_tensors[6].reshape(x.shape[0],-1)
        weights=(wq,wk,wv,wz,wa,wb)
        sizes=[w.shape[0] for w in weights]
        pitch=sum(sizes)
        packed=torch.empty((x.shape[0],pitch),device=x.device,dtype=x.dtype)
        destinations=packed.split(sizes,dim=-1)
        decoded_out=ctx.prepared[6].quantizer.dequant(ctx.prepared[6].data,ctx.prepared[6].scale,(128,128))
        dwo=dy.T@gated.reshape(x.shape[0],-1)
        dgate=(dy@decoded_out).view_as(o)
        do=torch.empty_like(o)
        rows=o.numel()//128;blocks=triton.cdiv(rows,16)
        partial=torch.empty((blocks,128),device=x.device,dtype=torch.float32)
        norm_backward[(blocks,)](o,p[3],nw,n,s,r,dgate,do,destinations[3],partial,
            rows,128,128,16,HEADS=o.shape[2],DZ_PITCH=pitch,num_warps=4,enable_fp_fusion=False)
        dnw=partial.sum(0).to(nw.dtype)
        del dgate,decoded_out,partial,gated,n,s,r,o
        p=p[:3]
        dq,dk,dv,dg,db=ChunkGatedDeltaRuleFunction.backward(inner,do,None)[:5]
        inner.saved_tensors=None
        del inner,do
        conv_grads=[]
        for z,w,dz,destination in zip(p[:3],(cq,ck,cv),(dq,dk,dv),destinations[:3]):
            _,dw,_,_=causal_conv1d_bwd_function(z[None].transpose(1,2),w,None,
                dz.reshape(1,x.shape[0],-1).transpose(1,2),None,None,None,
                destination[None].transpose(1,2),False,True)
            conv_grads.append(dw)
        dal,ddt=gate_backward_into(gate_input,beta,al,dt,dg,db,destinations[4],destinations[5],prepared=True)
        del p,dq,dk,dv,dg,db,gate_input,beta
        decoded=[pi.quantizer.dequant(pi.data,pi.scale,(128,128)) for pi in ctx.prepared[:6]]
        dx=packed@torch.cat(decoded)
        dw=(packed.T@x).split(sizes)
        return dx,None,None,None,*dw,*conv_grads,dal,ddt,dnw,dwo


def fp8_gdn_layer(x,parameters,prepared,cp_context,*,fused_conv=False):
    """One BF16 sequence, standard GDN/cache policy, BF16 gradients.

    Parameters: Q,K,V,Z,A,B projection weights; Q,K,V convolution weights;
    A_log,dt_bias,norm weight,output projection weight. Prepared weights are
    prequantized Q,K,V,Z,A,B,output weights; refresh after optimizer updates.
    All dequantization and BF16 backward GEMMs are runtime work.
    """
    if len(parameters)!=13 or len(prepared)!=7 or x.ndim!=2 or not x.is_contiguous():
        raise ValueError("expected contiguous [tokens,hidden], 13 parameters and 7 FP8 weights")
    if not x.is_cuda or x.dtype!=torch.bfloat16 or torch.is_autocast_enabled('cuda'):
        raise ValueError("requires explicit CUDA BF16 tensors, without autocast")
    if not cp_context.is_intra or cp_context.is_inter or x.shape[0]%64:
        raise ValueError("requires one unpadded intra-CP sequence")
    if any(t.device!=x.device for t in parameters):
        raise ValueError("all parameters must be on the input device")
    if any(t.dtype!=x.dtype for i,t in enumerate(parameters) if i not in (9,10)):
        raise ValueError("requires BF16 projection/convolution/norm weights")
    if parameters[9].dtype!=torch.float32 or parameters[10].dtype!=torch.float32:
        raise ValueError("requires FP32 decay parameters")
    for w,pi in zip((*parameters[:6],parameters[-1]),prepared):
        if pi.parameter is not w or pi.version!=w._version:
            raise ValueError("stale FP8 weight: re-quantize after optimizer update")
    return _FP8GDN.apply(x,tuple(prepared),cp_context,fused_conv,*parameters)

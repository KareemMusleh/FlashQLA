"""Zero-history causal convolution/SiLU/L2 norm for BF16 GDN Q/K training."""
import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

from flash_qla.utils.math import l2norm_bwd


@triton.jit
def _conv_norm(X,W,Y,R,TOKENS:tl.constexpr,HEADS:tl.constexpr,WIDTH:tl.constexpr,
               EPS:tl.constexpr,BT:tl.constexpr,NORMALIZE:tl.constexpr=True):
    t=tl.program_id(0)*BT+tl.arange(0,BT)
    h=tl.program_id(1)
    d=tl.arange(0,128)
    channel=h*128+d
    acc=tl.full((BT,128),0,tl.float32)
    for tap in tl.static_range(WIDTH):
        ti=t-WIDTH+1+tap
        x=tl.load(X+ti[:,None]*HEADS*128+channel[None,:],(ti[:,None]>=0)&(ti[:,None]<TOKENS),0).to(tl.float32)
        w=tl.load(W+channel*WIDTH+tap).to(tl.float32)
        acc=acc+x*w[None,:]
    activated=(acc/(1+libdevice.exp(-acc))).to(Y.dtype.element_ty).to(tl.float32)
    r=tl.rsqrt(tl.sum(activated*activated,1)+EPS)
    off=t[:,None]*HEADS*128+channel[None,:]
    if NORMALIZE:
        tl.store(Y+off,activated*r[:,None],t[:,None]<TOKENS)
    else:
        tl.store(Y+off,activated,t[:,None]<TOKENS)
    tl.store(R+t*HEADS+h,r,t<TOKENS)


class _ConvNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,w,eps):
        tokens,channels=x.shape[-2:]
        heads=channels//128
        y=torch.empty_like(x)
        r=torch.empty((*x.shape[:-1],heads),device=x.device,dtype=torch.float32)
        _conv_norm[(triton.cdiv(tokens,16),heads)](x,w,y,r,tokens,heads,w.shape[-1],eps,16,num_warps=4,enable_fp_fusion=False)
        ctx.save_for_backward(x,w,y,r)
        return y

    @staticmethod
    def backward(ctx,dy):
        from causal_conv1d.cpp_functions import causal_conv1d_bwd_function

        x,w,y,r=ctx.saved_tensors
        shaped=y.reshape(1,-1,y.shape[-1]//128,128)
        dc=l2norm_bwd(shaped,r,dy.contiguous().reshape_as(shaped),output_dtype=x.dtype).reshape_as(x)
        dx,dw,_,_=causal_conv1d_bwd_function(x.transpose(1,2),w,None,dc.transpose(1,2),None,None,None,None,False,True)
        return dx.transpose(1,2).contiguous(),dw,None


def causal_conv_silu_l2norm(x,weight,eps=1e-6):
    """One unpadded sequence; D=128, BF16, no bias/history; first-order gradients."""
    if x.ndim!=3 or x.shape[0]!=1 or x.shape[-1]%128 or not x.is_contiguous():
        raise ValueError("expected contiguous [1,tokens,heads*128] input")
    if not x.is_cuda or x.dtype!=torch.bfloat16 or weight.dtype!=x.dtype or weight.device!=x.device:
        raise ValueError("expected matching CUDA BF16 input and weights")
    if weight.shape[0]!=x.shape[-1] or weight.shape[-1] not in (2,3,4) or not weight.is_contiguous():
        raise ValueError("expected contiguous [channels,2/3/4] convolution weight")
    return _ConvNorm.apply(x,weight,eps)

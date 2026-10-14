"""Experimental state reconstruction split across the independent V dimension."""
import torch
import tilelang
import tilelang.language as T


@tilelang.jit(out_idx=[6],pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH:True})
def _state_reconstruct(tokens,hk,hv,parts,initial,stages,value_tile):
    C,D=64,128
    span=tokens//parts
    chunks=span//C

    @T.prim_func
    def kernel(K:T.Tensor((1,tokens,hk,D),'bfloat16'),
               V:T.Tensor((1,tokens,hv,D),'bfloat16'),
               A:T.Tensor((1,tokens,hv,C),'bfloat16'),
               G:T.Tensor((1,tokens,hv),'float32'),
               B:T.Tensor((1,tokens,hv),'float32'),
               H0:T.Tensor((parts,hv,D,D),'float32'),
               H:T.Tensor((1,tokens//C,hv,D,D),'bfloat16')):
        with T.Kernel(parts*hv*(D//value_tile),threads=128) as block:
            vb=block%(D//value_tile)
            ph=block//(D//value_tile)
            p,h=ph//hv,ph%hv
            lo=vb*value_tile
            ks=T.alloc_shared((C,D),'bfloat16')
            vs=T.alloc_shared((C,value_tile),'bfloat16')
            a=T.alloc_shared((C,C),'bfloat16')
            g=T.alloc_shared((C,),'float32')
            b=T.alloc_shared((C,),'float32')
            hs=T.alloc_shared((D,value_tile),'bfloat16')
            xs=T.alloc_shared((C,D),'bfloat16')
            ys=T.alloc_shared((C,value_tile),'bfloat16')
            st=T.alloc_fragment((D,value_tile),'float32')
            xf=T.alloc_fragment((C,D),'float32')
            yf=T.alloc_fragment((C,value_tile),'float32')
            T.use_swizzle(10)
            if initial:
                T.copy(H0[p,h,0:D,lo:lo+value_tile],st)
            else:
                T.clear(st)
            for c in T.Pipelined(chunks,num_stages=stages):
                start=p*span+c*C
                T.copy(K[0,start:start+C,h//(hv//hk),0:D],ks)
                T.copy(V[0,start:start+C,h,lo:lo+value_tile],vs)
                T.copy(A[0,start:start+C,h,0:C],a)
                T.copy(G[0,start:start+C,h],g)
                T.copy(B[0,start:start+C,h],b)
                T.copy(st,H[0,p*chunks+c,h,0:D,lo:lo+value_tile])
                T.copy(st,hs)
                T.gemm(a,ks,xf,transpose_A=True,clear_accum=True)
                for i,j in T.Parallel(C,D):
                    xf[i,j]*=-b[i]
                T.copy(xf,xs)
                T.gemm(ks,hs,yf,clear_accum=True)
                for i,j in T.Parallel(C,value_tile):
                    yf[i,j]=yf[i,j]*T.exp2(g[C-1]*1.442695)-vs[i,j]*T.exp2((g[C-1]-g[i])*1.442695)
                T.copy(yf,ys)
                for i,j in T.Parallel(D,value_tile):
                    st[i,j]*=T.exp2(g[C-1]*1.442695)
                T.gemm(xs,ys,st,transpose_A=True,clear_accum=False)
    return kernel


def reconstruct_states(k,v,a,g,b,initial_state=None,*,parts=8,stages=2,value_tile=64):
    if k.shape[0]!=1 or k.shape[-1]!=128 or v.shape[-1]!=128 or k.shape[1]%(parts*64):
        raise ValueError("requires one equal-partition sequence with D=128")
    if k.dtype!=torch.bfloat16 or v.dtype!=k.dtype or a.dtype!=k.dtype:
        raise ValueError("requires BF16 K/V/inverse")
    if value_tile not in (32,64,128):
        raise ValueError("value_tile must divide 128")
    if initial_state is None:
        h0=torch.empty((parts,v.shape[-2],128,128),device=k.device,dtype=torch.float32)
    else:
        h0=initial_state.float().contiguous()
    return _state_reconstruct(k.shape[1],k.shape[-2],v.shape[-2],parts,initial_state is not None,stages,value_tile)(k,v,a,g,b,h0)

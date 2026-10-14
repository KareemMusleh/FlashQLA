"""Explicit cuBLASLt BF16 projection plans for forward benchmarking.

Descriptors and workspace are created before capture. Algorithms must be tuned
for the actual shape/library/GPU; no opaque algorithm is persisted across versions.
"""
import ctypes as ct
from pathlib import Path
import torch


class Algo(ct.Structure):
    _fields_=[('data',ct.c_uint64*8)]


class Heuristic(ct.Structure):
    _fields_=[('algo',Algo),('workspace',ct.c_size_t),('state',ct.c_int),
              ('waves',ct.c_float),('reserved',ct.c_int*4)]


def _check(status):
    if status:raise RuntimeError(f'cuBLASLt status {status}')


def _library():
    paths=list((Path(torch.__file__).resolve().parent.parent/'nvidia').glob('*/lib/libcublasLt.so.*'))
    if not paths:paths=list(Path('/usr/local/cuda/targets/x86_64-linux/lib').glob('libcublasLt.so.*'))
    if not paths:raise RuntimeError('cuBLASLt shared library not found')
    lib=ct.CDLL(str(paths[0]))
    p=ct.c_void_p
    signatures={
        'cublasLtCreate':[ct.POINTER(p)],'cublasLtDestroy':[p],
        'cublasLtMatmulDescCreate':[ct.POINTER(p),ct.c_int,ct.c_int],
        'cublasLtMatmulDescDestroy':[p],
        'cublasLtMatmulDescSetAttribute':[p,ct.c_int,p,ct.c_size_t],
        'cublasLtMatrixLayoutCreate':[ct.POINTER(p),ct.c_int,ct.c_uint64,ct.c_uint64,ct.c_int64],
        'cublasLtMatrixLayoutSetAttribute':[p,ct.c_int,p,ct.c_size_t],
        'cublasLtMatrixLayoutDestroy':[p],
        'cublasLtMatmulPreferenceCreate':[ct.POINTER(p)],'cublasLtMatmulPreferenceDestroy':[p],
        'cublasLtMatmulPreferenceSetAttribute':[p,ct.c_int,p,ct.c_size_t],
        'cublasLtMatmulAlgoGetHeuristic':[p,p,p,p,p,p,p,ct.c_int,ct.POINTER(Heuristic),ct.POINTER(ct.c_int)],
        'cublasLtMatmul':[p]*14+[ct.c_size_t,p],
    }
    for name,args in signatures.items():
        getattr(lib,name).argtypes=args
        getattr(lib,name).restype=ct.c_int
    return lib


class ProjectionPlan:
    """Row-major BF16 X @ W.T, FP32 accumulation, caller's CUDA stream."""
    def __init__(self,m,n,k,device,*,weight_transposed=False,column_order=False,workspace_bytes=64<<20,algorithms=32):
        self.lib=_library()
        self.m,self.n,self.k=m,n,k
        self.device=torch.device(device)
        self.transposed=weight_transposed
        self.column_order=column_order
        self.handle=ct.c_void_p();self.desc=ct.c_void_p();self.layouts=[]
        _check(self.lib.cublasLtCreate(ct.byref(self.handle)))
        _check(self.lib.cublasLtMatmulDescCreate(ct.byref(self.desc),68,0)) # COMPUTE_32F, CUDA_R_32F
        trans=ct.c_int(0 if weight_transposed else 1)
        _check(self.lib.cublasLtMatmulDescSetAttribute(self.desc,3 if column_order else 4,ct.byref(trans),ct.sizeof(trans)))
        shapes=[(n,k) if weight_transposed else (k,n),(k,m),(n,m)] if column_order else [(m,k),(k,n) if weight_transposed else (n,k),(m,n)]
        for rows,cols in shapes:
            layout=ct.c_void_p()
            _check(self.lib.cublasLtMatrixLayoutCreate(ct.byref(layout),14,rows,cols,rows if column_order else cols)) # CUDA_R_16BF
            order=ct.c_int(0 if column_order else 1)
            _check(self.lib.cublasLtMatrixLayoutSetAttribute(layout,1,ct.byref(order),ct.sizeof(order)))
            self.layouts.append(layout)
        self.workspace=torch.empty(workspace_bytes,device=self.device,dtype=torch.uint8)
        pref=ct.c_void_p()
        _check(self.lib.cublasLtMatmulPreferenceCreate(ct.byref(pref)))
        size=ct.c_size_t(workspace_bytes)
        _check(self.lib.cublasLtMatmulPreferenceSetAttribute(pref,1,ct.byref(size),ct.sizeof(size)))
        results=(Heuristic*algorithms)();count=ct.c_int()
        a,b,c=self.layouts
        _check(self.lib.cublasLtMatmulAlgoGetHeuristic(self.handle,self.desc,a,b,c,c,pref,algorithms,results,ct.byref(count)))
        self.results=[results[i] for i in range(count.value) if results[i].state==0]
        _check(self.lib.cublasLtMatmulPreferenceDestroy(pref))
        if not self.results:raise RuntimeError('No cuBLASLt heuristic algorithms')
        self.alpha=ct.c_float(1);self.beta=ct.c_float(0)
        self.best=0

    def run(self,x,weight,index=None,out=None):
        if torch.is_grad_enabled() and (x.requires_grad or weight.requires_grad):
            raise ValueError('ProjectionPlan is forward-only')
        expected=(self.k,self.n) if self.transposed else (self.n,self.k)
        if x.shape!=(self.m,self.k) or weight.shape!=expected:
            raise ValueError('projection input shape differs from plan')
        if any(t.dtype!=torch.bfloat16 or t.device!=self.device or not t.is_contiguous() for t in (x,weight)):
            raise ValueError('plan requires contiguous BF16 tensors on its device')
        if out is None:out=torch.empty((self.m,self.n),device=x.device,dtype=x.dtype)
        if out.shape!=(self.m,self.n) or out.dtype!=x.dtype or out.device!=x.device or not out.is_contiguous():
            raise ValueError('invalid output buffer')
        a,b,c=self.layouts
        result=self.results[self.best if index is None else index]
        ap,bp=(weight.data_ptr(),x.data_ptr()) if self.column_order else (x.data_ptr(),weight.data_ptr())
        _check(self.lib.cublasLtMatmul(self.handle,self.desc,ct.byref(self.alpha),ap,a,
            bp,b,ct.byref(self.beta),out.data_ptr(),c,out.data_ptr(),c,
            ct.byref(result.algo),self.workspace.data_ptr(),self.workspace.numel(),torch.cuda.current_stream(x.device).cuda_stream))
        return out

    def close(self):
        if self.handle.value:
            for layout in self.layouts:_check(self.lib.cublasLtMatrixLayoutDestroy(layout))
            _check(self.lib.cublasLtMatmulDescDestroy(self.desc))
            _check(self.lib.cublasLtDestroy(self.handle))
            self.handle=ct.c_void_p()

#if defined(_MSC_VER) && !defined(__clang__) && _MSC_VER < 1940
#define _tl_orig_alignas alignas
#define alignas(N) _tl_orig_alignas((N) <= 64 ? (N) : 64)
#include <cuda.h>
#undef alignas
#define alignas _tl_orig_alignas
#endif
#include <tl_templates/cuda/instruction/wgmma.h>
#include <tl_templates/cuda/intrin.h>
#include <tl_templates/cuda/barrier.h>
#include <tl_templates/cuda/reduce.h>
#include <tl_templates/cuda/scan.h>
#include <tl_templates/cuda/ldsm.h>
#include <tl_templates/cuda/threadblock_swizzle.h>
#include <tl_templates/cuda/debug.h>
#ifdef ENABLE_BF16
#include <tl_templates/cuda/cuda_bf16_fallbacks.cuh>
#endif

extern "C" __global__ void kernel_kernel(const bfloat16_t* __restrict__ A, const float* __restrict__ Beta, const float* __restrict__ G, const bfloat16_t* __restrict__ K, bfloat16_t* __restrict__ Mout, bfloat16_t* __restrict__ Nout, const bfloat16_t* __restrict__ V);
extern "C" __global__ void __launch_bounds__(256, 1) kernel_kernel(const bfloat16_t* __restrict__ A, const float* __restrict__ Beta, const float* __restrict__ G, const bfloat16_t* __restrict__ K, bfloat16_t* __restrict__ Mout, bfloat16_t* __restrict__ Nout, const bfloat16_t* __restrict__ V) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  void* ks = ((void*)((char*)buf_dyn_shmem + 0));
  void* vs = ((void*)((char*)buf_dyn_shmem + 16384));
  void* x = ((void*)((char*)buf_dyn_shmem + 32768));
  void* a = ((void*)((char*)buf_dyn_shmem + 49152));
  void* beta = ((void*)((char*)buf_dyn_shmem + 57344));
  void* g = ((void*)((char*)buf_dyn_shmem + 57600));
  float nt[64];
  float mt[64];
  float xf[32];
  bfloat16_t operand[64];
  float yz[32];
  bfloat16_t yz_operand[32];
  #pragma unroll
  for (int i = 0; i < 16; ++i) {
    float broadcast_var = 0x0p+0f/*0.000000e+00*/;
    *(float4*)(nt + (i * 4)) = make_float4(broadcast_var, broadcast_var, broadcast_var, broadcast_var);
  }
  #pragma unroll
  for (int i_1 = 0; i_1 < 64; ++i_1) {
    float condval;
    if ((((((((int)threadIdx.x) >> 5) * 16) + (((i_1 & 3) >> 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2)) == ((((i_1 >> 2) * 8) + ((((int)threadIdx.x) & 3) * 2)) + (i_1 & 1)))) {
      condval = 0x1p+0f/*1.000000e+00*/;
    } else {
      condval = 0x0p+0f/*0.000000e+00*/;
    }
    mt[i_1] = condval;
  }
  int condval_1;
  if ((((int)blockIdx.x) == 7)) {
    condval_1 = 0;
  } else {
    condval_1 = 32;
  }
  for (int c = 0; c < condval_1; ++c) {
    __syncthreads();
    #pragma unroll
    for (int i_2 = 0; i_2 < 4; ++i_2) {
      bfloat16_t broadcast_var_1 = bfloat16_t(0x0p+0f/*0.000000e+00*/);
      uint4 condval_2;
      if ((((c >> 5) + ((int)blockIdx.x)) < 8)) {
        condval_2 = *(uint4*)(K + ((((((((int64_t)((int)blockIdx.x)) * (int64_t)4194304) + (((int64_t)c) * (int64_t)131072)) + (((int64_t)i_2) * (int64_t)32768)) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)4) * (int64_t)2048)) + ((((int64_t)((int)blockIdx.y)) / (int64_t)3) * (int64_t)128)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)15) * (int64_t)8)));
      } else {
        condval_2 = make_uint4(__pack_nv_bfloat162(broadcast_var_1, broadcast_var_1), __pack_nv_bfloat162(broadcast_var_1, broadcast_var_1), __pack_nv_bfloat162(broadcast_var_1, broadcast_var_1), __pack_nv_bfloat162(broadcast_var_1, broadcast_var_1));
      }
      *(uint4*)(((bfloat16_t*)ks) + ((((((((((int)threadIdx.x) & 15) >> 3) * 4096) + (i_2 * 1024)) + ((((int)threadIdx.x) >> 4) * 64)) + (((((((int)threadIdx.x) & 127) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))) = condval_2;
    }
    #pragma unroll
    for (int i_3 = 0; i_3 < 4; ++i_3) {
      bfloat16_t broadcast_var_2 = bfloat16_t(0x0p+0f/*0.000000e+00*/);
      uint4 condval_3;
      if ((((c >> 5) + ((int)blockIdx.x)) < 8)) {
        condval_3 = *(uint4*)(V + ((((((((int64_t)((int)blockIdx.x)) * (int64_t)12582912) + (((int64_t)c) * (int64_t)393216)) + (((int64_t)i_3) * (int64_t)98304)) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)4) * (int64_t)6144)) + (((int64_t)((int)blockIdx.y)) * (int64_t)128)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)15) * (int64_t)8)));
      } else {
        condval_3 = make_uint4(__pack_nv_bfloat162(broadcast_var_2, broadcast_var_2), __pack_nv_bfloat162(broadcast_var_2, broadcast_var_2), __pack_nv_bfloat162(broadcast_var_2, broadcast_var_2), __pack_nv_bfloat162(broadcast_var_2, broadcast_var_2));
      }
      *(uint4*)(((bfloat16_t*)vs) + ((i_3 * 2048) + (((int)threadIdx.x) * 8))) = condval_3;
    }
    #pragma unroll
    for (int i_4 = 0; i_4 < 2; ++i_4) {
      bfloat16_t broadcast_var_3 = bfloat16_t(0x0p+0f/*0.000000e+00*/);
      uint4 condval_4;
      if ((((c >> 5) + ((int)blockIdx.x)) < 8)) {
        condval_4 = *(uint4*)(A + ((((((((int64_t)((int)blockIdx.x)) * (int64_t)6291456) + (((int64_t)c) * (int64_t)196608)) + (((int64_t)i_4) * (int64_t)98304)) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)3) * (int64_t)3072)) + (((int64_t)((int)blockIdx.y)) * (int64_t)64)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)7) * (int64_t)8)));
      } else {
        condval_4 = make_uint4(__pack_nv_bfloat162(broadcast_var_3, broadcast_var_3), __pack_nv_bfloat162(broadcast_var_3, broadcast_var_3), __pack_nv_bfloat162(broadcast_var_3, broadcast_var_3), __pack_nv_bfloat162(broadcast_var_3, broadcast_var_3));
      }
      *(uint4*)(((bfloat16_t*)a) + (((((i_4 * 2048) + ((((int)threadIdx.x) >> 3) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))) = condval_4;
    }
    if (((int)threadIdx.x) < 64) {
      float condval_5;
      if ((((c >> 5) + ((int)blockIdx.x)) < 8)) {
        condval_5 = G[((((((int64_t)((int)blockIdx.x)) * (int64_t)98304) + (((int64_t)c) * (int64_t)3072)) + (((int64_t)((int)threadIdx.x)) * (int64_t)48)) + ((int64_t)((int)blockIdx.y)))];
      } else {
        condval_5 = 0x0p+0f/*0.000000e+00*/;
      }
      ((float*)g)[((int)threadIdx.x)] = condval_5;
      float condval_6;
      if ((((c >> 5) + ((int)blockIdx.x)) < 8)) {
        condval_6 = Beta[((((((int64_t)((int)blockIdx.x)) * (int64_t)98304) + (((int64_t)c) * (int64_t)3072)) + (((int64_t)((int)threadIdx.x)) * (int64_t)48)) + ((int64_t)((int)blockIdx.y)))];
      } else {
        condval_6 = 0x0p+0f/*0.000000e+00*/;
      }
      ((float*)beta)[((int)threadIdx.x)] = condval_6;
    }
    {
      tl::GmmaDescriptor desc_a;
      tl::GmmaDescriptor desc_b;
      __syncthreads();
      tl::initialize_wgmma_descriptor<1, 0, 64>(desc_a, (&(((bfloat16_t*)a)[0])));
      tl::initialize_wgmma_descriptor<1, 512, 64>(desc_b, (&(((bfloat16_t*)ks)[0])));
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(xf + 0), 32);
      tl::warpgroup_arrive();
      tl::fence_proxy_async();
      #pragma unroll
      for (int ki = 0; ki < 4; ++ki) {
        tl::wgmma_ss<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 64, 64, 16, true, true, 1, 1>(uint64_t(desc_a + ((ki * 2048) >> 4)), uint64_t(desc_b + ((((((int)threadIdx.x) >> 7) * 8192) + (ki * 2048)) >> 4)), ((uint32_t*)(xf + 0)), ((0 < ki) ? 1 : 0));
      }
      tl::warpgroup_commit_batch();
      tl::warpgroup_wait<0>();
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(xf + 0), 32);
    }
    __syncthreads();
    #pragma unroll
    for (int i_5 = 0; i_5 < 16; ++i_5) {
      float beta_local_cast_1[2];
      bfloat16_t x_local_cast[2];
      *(float2*)(beta_local_cast_1 + 0) = make_float2(((float*)beta)[(((((((int)threadIdx.x) & 127) >> 5) * 16) + ((i_5 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2))], ((float*)beta)[(((((((int)threadIdx.x) & 127) >> 5) * 16) + ((i_5 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2))]);
      float broadcast_var_4 = -0x1p+0f/*-1.000000e+00*/;
      uint1 __1;
      float2 __2;
        float2 __3;
          float2 v_ = *(float2*)(beta_local_cast_1 + 0);
          float2 v__1 = make_float2(broadcast_var_4, broadcast_var_4);
          __3.x = (v_.x*v__1.x);
          __3.y = (v_.y*v__1.y);
        float2 v__2 = *(float2*)(xf + (i_5 * 2));
        __2.x = (__3.x*v__2.x);
        __2.y = (__3.y*v__2.y);
      (reinterpret_cast<__nv_bfloat162*>(&__1))[0] = __float22bfloat162_rn(((float2*)(&__2))[0]);
      *(uint1*)(x_local_cast + 0) = __1;
      *(uint1*)(((bfloat16_t*)x) + ((((((((((int)threadIdx.x) >> 5) * 1024) + ((i_5 & 1) * 512)) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + (((((((int)threadIdx.x) & 31) >> 4) + (i_5 >> 3)) & 1) * 32)) + (((((((int)threadIdx.x) & 15) >> 3) + ((i_5 & 7) >> 2)) & 1) * 16)) + (((((((int)threadIdx.x) & 7) >> 2) + ((i_5 & 3) >> 1)) & 1) * 8)) + ((((int)threadIdx.x) & 3) * 2))) = *(uint1*)(x_local_cast + 0);
    }
    #pragma unroll
    for (int i_6 = 0; i_6 < 16; ++i_6) {
      uint2 __4;
      float4 v__3 = *(float4*)(mt + (i_6 * 4));
      (reinterpret_cast<__nv_bfloat162*>(&__4))[0] = __float22bfloat162_rn(((float2*)(&v__3))[0]);
      (reinterpret_cast<__nv_bfloat162*>(&__4))[1] = __float22bfloat162_rn(((float2*)(&v__3))[1]);
      *(uint2*)(operand + (i_6 * 4)) = __4;
    }
    {
      tl::GmmaDescriptor desc_b_1;
      __syncthreads();
      tl::initialize_wgmma_descriptor<1, 1, 64>(desc_b_1, (&(((bfloat16_t*)ks)[0])));
      tl::warpgroup_fence_operand(reinterpret_cast<uint32_t*>(operand + 0), 32);
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(yz + 0), 64);
      tl::warpgroup_arrive();
      tl::fence_proxy_async();
      #pragma unroll
      for (int ki_1 = 0; ki_1 < 8; ++ki_1) {
        tl::wgmma_rs<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 64, 64, 16, false, false, 1, 1>(reinterpret_cast<const uint32_t*>(operand + (ki_1 * 8)), uint64_t(desc_b_1 + ((((ki_1 >> 2) * 8192) + ((ki_1 & 3) * 32)) >> 4)), reinterpret_cast<uint32_t*>(yz + 0), ((0 < ki_1) ? 1 : 0));
      }
      tl::warpgroup_commit_batch();
      tl::warpgroup_wait<0>();
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(yz + 0), 64);
      tl::warpgroup_fence_operand(reinterpret_cast<uint32_t*>(operand + 0), 32);
    }
    #pragma unroll
    for (int i_7 = 0; i_7 < 8; ++i_7) {
      uint2 __5;
      float4 v__4 = *(float4*)(yz + (i_7 * 4));
      (reinterpret_cast<__nv_bfloat162*>(&__5))[0] = __float22bfloat162_rn(((float2*)(&v__4))[0]);
      (reinterpret_cast<__nv_bfloat162*>(&__5))[1] = __float22bfloat162_rn(((float2*)(&v__4))[1]);
      *(uint2*)(yz_operand + (i_7 * 4)) = __5;
    }
    {
      tl::GmmaDescriptor desc_b_2;
      tl::initialize_wgmma_descriptor<1, 512, 64>(desc_b_2, (&(((bfloat16_t*)x)[0])));
      tl::warpgroup_fence_operand(reinterpret_cast<uint32_t*>(yz_operand + 0), 16);
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(mt + 0), 128);
      tl::warpgroup_arrive();
      #pragma unroll
      for (int ki_2 = 0; ki_2 < 4; ++ki_2) {
        tl::wgmma_rs<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 64, 128, 16, false, true, 1, 1>(reinterpret_cast<const uint32_t*>(yz_operand + (ki_2 * 8)), uint64_t(desc_b_2 + ((ki_2 * 2048) >> 4)), reinterpret_cast<uint32_t*>(mt + 0), 1);
      }
      tl::warpgroup_commit_batch();
      tl::warpgroup_wait<0>();
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(mt + 0), 128);
      tl::warpgroup_fence_operand(reinterpret_cast<uint32_t*>(yz_operand + 0), 16);
    }
    #pragma unroll
    for (int i_8 = 0; i_8 < 16; ++i_8) {
      float4 __6;
        float4 v__5 = *(float4*)(mt + (i_8 * 4));
        float4 v__6 = make_float4(expf(((float*)g)[63]), expf(((float*)g)[63]), expf(((float*)g)[63]), expf(((float*)g)[63]));
        __6.x = (v__5.x*v__6.x);
        __6.y = (v__5.y*v__6.y);
        __6.z = (v__5.z*v__6.z);
        __6.w = (v__5.w*v__6.w);
      *(float4*)(mt + (i_8 * 4)) = __6;
    }
    #pragma unroll
    for (int i_9 = 0; i_9 < 16; ++i_9) {
      uint2 __7;
      float4 v__7 = *(float4*)(nt + (i_9 * 4));
      (reinterpret_cast<__nv_bfloat162*>(&__7))[0] = __float22bfloat162_rn(((float2*)(&v__7))[0]);
      (reinterpret_cast<__nv_bfloat162*>(&__7))[1] = __float22bfloat162_rn(((float2*)(&v__7))[1]);
      *(uint2*)(operand + (i_9 * 4)) = __7;
    }
    {
      tl::GmmaDescriptor desc_b_3;
      tl::initialize_wgmma_descriptor<1, 1, 64>(desc_b_3, (&(((bfloat16_t*)ks)[0])));
      tl::warpgroup_fence_operand(reinterpret_cast<uint32_t*>(operand + 0), 32);
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(yz + 0), 64);
      tl::warpgroup_arrive();
      #pragma unroll
      for (int ki_3 = 0; ki_3 < 8; ++ki_3) {
        tl::wgmma_rs<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 64, 64, 16, false, false, 1, 1>(reinterpret_cast<const uint32_t*>(operand + (ki_3 * 8)), uint64_t(desc_b_3 + ((((ki_3 >> 2) * 8192) + ((ki_3 & 3) * 32)) >> 4)), reinterpret_cast<uint32_t*>(yz + 0), ((0 < ki_3) ? 1 : 0));
      }
      tl::warpgroup_commit_batch();
      tl::warpgroup_wait<0>();
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(yz + 0), 64);
      tl::warpgroup_fence_operand(reinterpret_cast<uint32_t*>(operand + 0), 32);
    }
    #pragma unroll
    for (int i_10 = 0; i_10 < 32; ++i_10) {
      yz_operand[i_10] = ((bfloat16_t)((expf(((float*)g)[63]) * yz[i_10]) - (((float)((bfloat16_t*)vs)[(((((((i_10 >> 2) * 1024) + ((((int)threadIdx.x) & 3) * 256)) + ((i_10 & 1) * 128)) + ((((int)threadIdx.x) >> 5) * 16)) + (((i_10 & 3) >> 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2))]) * expf((((float*)g)[63] - ((float*)g)[((((i_10 >> 2) * 8) + ((((int)threadIdx.x) & 3) * 2)) + (i_10 & 1))])))));
    }
    #pragma unroll
    for (int i_11 = 0; i_11 < 16; ++i_11) {
      float4 __8;
        float4 v__8 = *(float4*)(nt + (i_11 * 4));
        float4 v__9 = make_float4(expf(((float*)g)[63]), expf(((float*)g)[63]), expf(((float*)g)[63]), expf(((float*)g)[63]));
        __8.x = (v__8.x*v__9.x);
        __8.y = (v__8.y*v__9.y);
        __8.z = (v__8.z*v__9.z);
        __8.w = (v__8.w*v__9.w);
      *(float4*)(nt + (i_11 * 4)) = __8;
    }
    {
      tl::GmmaDescriptor desc_b_4;
      tl::initialize_wgmma_descriptor<1, 512, 64>(desc_b_4, (&(((bfloat16_t*)x)[0])));
      tl::warpgroup_fence_operand(reinterpret_cast<uint32_t*>(yz_operand + 0), 16);
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(nt + 0), 128);
      tl::warpgroup_arrive();
      #pragma unroll
      for (int ki_4 = 0; ki_4 < 4; ++ki_4) {
        tl::wgmma_rs<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 64, 128, 16, false, true, 1, 1>(reinterpret_cast<const uint32_t*>(yz_operand + (ki_4 * 8)), uint64_t(desc_b_4 + ((ki_4 * 2048) >> 4)), reinterpret_cast<uint32_t*>(nt + 0), 1);
      }
      tl::warpgroup_commit_batch();
      tl::warpgroup_wait<0>();
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(nt + 0), 128);
      tl::warpgroup_fence_operand(reinterpret_cast<uint32_t*>(yz_operand + 0), 16);
    }
  }
  #pragma unroll
  for (int i_12 = 0; i_12 < 64; ++i_12) {
    Nout[((((((((((int)blockIdx.x) * 786432) + (((int)blockIdx.y) * 16384)) + ((i_12 >> 2) * 1024)) + ((((int)threadIdx.x) & 3) * 256)) + (((i_12 & 3) >> 1) * 128)) + ((((int)threadIdx.x) >> 5) * 16)) + ((i_12 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2))] = ((bfloat16_t)nt[((((i_12 >> 2) * 4) + ((i_12 & 1) * 2)) + ((i_12 & 3) >> 1))]);
  }
  #pragma unroll
  for (int i_13 = 0; i_13 < 64; ++i_13) {
    Mout[((((((((((int)blockIdx.x) * 786432) + (((int)blockIdx.y) * 16384)) + ((i_13 >> 2) * 1024)) + ((((int)threadIdx.x) & 3) * 256)) + (((i_13 & 3) >> 1) * 128)) + ((((int)threadIdx.x) >> 5) * 16)) + ((i_13 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2))] = ((bfloat16_t)mt[((((i_13 >> 2) * 4) + ((i_13 & 1) * 2)) + ((i_13 & 3) >> 1))]);
  }
}


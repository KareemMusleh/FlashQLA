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
#include <tl_templates/cuda/copy.h>
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
  void* g = ((void*)((char*)buf_dyn_shmem + 0));
  void* beta = ((void*)((char*)buf_dyn_shmem + 512));
  void* a = ((void*)((char*)buf_dyn_shmem + 1024));
  void* ks = ((void*)((char*)buf_dyn_shmem + 17408));
  void* vs = ((void*)((char*)buf_dyn_shmem + 50176));
  void* x = ((void*)((char*)buf_dyn_shmem + 82944));
  void* decay = ((void*)((char*)buf_dyn_shmem + 99328));
  float nt[64];
  float mt[64];
  float gamma = 0x0p+0f/*0.000000e+00*/;
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
  if ((((int)blockIdx.x) == 1)) {
    condval_1 = 0;
  } else {
    condval_1 = 128;
  }
  if (0 < condval_1) {
    if (((int)threadIdx.x) < 64) {
      tl::cp_async_gs<4>((&(((float*)g)[((int)threadIdx.x)])), (&(G[(((((int)blockIdx.x) * 393216) + (((int)threadIdx.x) * 48)) + ((int)blockIdx.y))])));
    }
    tl::cp_async_commit();
    #pragma unroll
    for (int i_2 = 0; i_2 < 2; ++i_2) {
      tl::cp_async_gs<16>((&(((bfloat16_t*)a)[(((((i_2 * 2048) + ((((int)threadIdx.x) >> 3) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(A[(((((((int)blockIdx.x) * 25165824) + (i_2 * 98304)) + ((((int)threadIdx.x) >> 3) * 3072)) + (((int)blockIdx.y) * 64)) + ((((int)threadIdx.x) & 7) * 8))])));
    }
    tl::cp_async_commit();
    if (((int)threadIdx.x) < 64) {
      tl::cp_async_gs<4>((&(((float*)beta)[((int)threadIdx.x)])), (&(Beta[(((((int)blockIdx.x) * 393216) + (((int)threadIdx.x) * 48)) + ((int)blockIdx.y))])));
    }
    tl::cp_async_commit();
    #pragma unroll
    for (int i_3 = 0; i_3 < 4; ++i_3) {
      tl::cp_async_gs<16>((&(((bfloat16_t*)ks)[((((((((((int)threadIdx.x) & 15) >> 3) * 4096) + (i_3 * 1024)) + ((((int)threadIdx.x) >> 4) * 64)) + (((((((int)threadIdx.x) & 127) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(K[(((((((int)blockIdx.x) * 16777216) + (i_3 * 32768)) + ((((int)threadIdx.x) >> 4) * 2048)) + ((((int)blockIdx.y) / 3) * 128)) + ((((int)threadIdx.x) & 15) * 8))])));
    }
    tl::cp_async_commit();
    #pragma unroll
    for (int i_4 = 0; i_4 < 4; ++i_4) {
      tl::cp_async_gs<16>((&(((bfloat16_t*)vs)[((i_4 * 2048) + (((int)threadIdx.x) * 8))])), (&(V[(((((((int)blockIdx.x) * 50331648) + (i_4 * 98304)) + ((((int)threadIdx.x) >> 4) * 6144)) + (((int)blockIdx.y) * 128)) + ((((int)threadIdx.x) & 15) * 8))])));
    }
    tl::cp_async_commit();
  }
  int condval_2;
  if ((((int)blockIdx.x) == 1)) {
    condval_2 = 0;
  } else {
    condval_2 = 128;
  }
  if (1 < condval_2) {
    if (((int)threadIdx.x) < 64) {
      tl::cp_async_gs<4>((&(((float*)g)[(((int)threadIdx.x) + 64)])), (&(G[((((((int)blockIdx.x) * 393216) + (((int)threadIdx.x) * 48)) + ((int)blockIdx.y)) + 3072)])));
    }
    tl::cp_async_commit();
    #pragma unroll
    for (int i_5 = 0; i_5 < 2; ++i_5) {
      tl::cp_async_gs<16>((&(((bfloat16_t*)a)[((((((i_5 * 2048) + ((((int)threadIdx.x) >> 3) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 4096)])), (&(A[((((((((int)blockIdx.x) * 25165824) + (i_5 * 98304)) + ((((int)threadIdx.x) >> 3) * 3072)) + (((int)blockIdx.y) * 64)) + ((((int)threadIdx.x) & 7) * 8)) + 196608)])));
    }
    tl::cp_async_commit();
    if (((int)threadIdx.x) < 64) {
      tl::cp_async_gs<4>((&(((float*)beta)[(((int)threadIdx.x) + 64)])), (&(Beta[((((((int)blockIdx.x) * 393216) + (((int)threadIdx.x) * 48)) + ((int)blockIdx.y)) + 3072)])));
    }
    tl::cp_async_commit();
    #pragma unroll
    for (int i_6 = 0; i_6 < 4; ++i_6) {
      tl::cp_async_gs<16>((&(((bfloat16_t*)ks)[(((((((((((int)threadIdx.x) & 15) >> 3) * 4096) + (i_6 * 1024)) + ((((int)threadIdx.x) >> 4) * 64)) + (((((((int)threadIdx.x) & 127) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 8192)])), (&(K[((((((((int)blockIdx.x) * 16777216) + (i_6 * 32768)) + ((((int)threadIdx.x) >> 4) * 2048)) + ((((int)blockIdx.y) / 3) * 128)) + ((((int)threadIdx.x) & 15) * 8)) + 131072)])));
    }
    tl::cp_async_commit();
    #pragma unroll
    for (int i_7 = 0; i_7 < 4; ++i_7) {
      tl::cp_async_gs<16>((&(((bfloat16_t*)vs)[(((i_7 * 2048) + (((int)threadIdx.x) * 8)) + 8192)])), (&(V[((((((((int)blockIdx.x) * 50331648) + (i_7 * 98304)) + ((((int)threadIdx.x) >> 4) * 6144)) + (((int)blockIdx.y) * 128)) + ((((int)threadIdx.x) & 15) * 8)) + 393216)])));
    }
    tl::cp_async_commit();
  }
  int condval_3;
  if ((((int)blockIdx.x) == 1)) {
    condval_3 = 0;
  } else {
    condval_3 = 128;
  }
  for (int c = 0; c < (condval_3 - 2); ++c) {
    tl::cp_async_wait<9>();
    __syncthreads();
    gamma = exp2f((((float*)g)[(((c & 1) * 64) + 63)] * 0x1.71547652b82fep+0f/*1.442695e+00*/));
    if (((int)threadIdx.x) < 64) {
      ((float*)decay)[((int)threadIdx.x)] = exp2f(((((float*)g)[(((c & 1) * 64) + 63)] - ((float*)g)[(((c & 1) * 64) + ((int)threadIdx.x))]) * 0x1.71547652b82fep+0f/*1.442695e+00*/));
    }
    __syncthreads();
    if (((int)threadIdx.x) < 64) {
      tl::cp_async_gs_conditional<4>((&(((float*)g)[(((c & 1) * 64) + ((int)threadIdx.x))])), (&(G[(((((((int64_t)((int)blockIdx.x)) * (int64_t)393216) + (((int64_t)c) * (int64_t)3072)) + (((int64_t)((int)threadIdx.x)) * (int64_t)48)) + ((int64_t)((int)blockIdx.y))) + (int64_t)6144)])), ((((((int)blockIdx.x) * 64) + (c >> 1)) < 127) && (((((int)blockIdx.x) * 64) + (c >> 1)) < 127)));
    }
    tl::cp_async_commit();
    tl::cp_async_wait<7>();
    __syncthreads();
    {
      tl::GmmaDescriptor desc_a;
      tl::GmmaDescriptor desc_b;
      tl::initialize_wgmma_descriptor<1, 0, 64>(desc_a, (&(((bfloat16_t*)a)[0])));
      tl::increase_descriptor_offset<int>(desc_a, ((c & 1) * 8192));
      tl::initialize_wgmma_descriptor<1, 512, 64>(desc_b, (&(((bfloat16_t*)ks)[0])));
      tl::increase_descriptor_offset<int>(desc_b, ((c & 1) * 16384));
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
    for (int i_8 = 0; i_8 < 2; ++i_8) {
      tl::cp_async_gs_conditional<16>((&(((bfloat16_t*)a)[(((((((c & 1) * 4096) + (i_8 * 2048)) + ((((int)threadIdx.x) >> 3) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(A[(((((((((int64_t)((int)blockIdx.x)) * (int64_t)25165824) + (((int64_t)c) * (int64_t)196608)) + (((int64_t)i_8) * (int64_t)98304)) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)3) * (int64_t)3072)) + (((int64_t)((int)blockIdx.y)) * (int64_t)64)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)7) * (int64_t)8)) + (int64_t)393216)])), ((((((int)blockIdx.x) * 64) + (c >> 1)) < 127) && (((((int)blockIdx.x) * 64) + (c >> 1)) < 127)));
    }
    tl::cp_async_commit();
    tl::cp_async_wait<9>();
    __syncthreads();
    #pragma unroll
    for (int i_9 = 0; i_9 < 16; ++i_9) {
      float beta_local_cast_1[2];
      bfloat16_t x_local_cast[2];
      *(float2*)(beta_local_cast_1 + 0) = make_float2(((float*)beta)[(((((c & 1) * 64) + (((((int)threadIdx.x) & 127) >> 5) * 16)) + ((i_9 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2))], ((float*)beta)[(((((c & 1) * 64) + (((((int)threadIdx.x) & 127) >> 5) * 16)) + ((i_9 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2))]);
      float broadcast_var_1 = -0x1p+0f/*-1.000000e+00*/;
      uint1 __1;
      float2 __2;
        float2 __3;
          float2 v_ = *(float2*)(beta_local_cast_1 + 0);
          float2 v__1 = make_float2(broadcast_var_1, broadcast_var_1);
          __3.x = (v_.x*v__1.x);
          __3.y = (v_.y*v__1.y);
        float2 v__2 = *(float2*)(xf + (i_9 * 2));
        __2.x = (__3.x*v__2.x);
        __2.y = (__3.y*v__2.y);
      (reinterpret_cast<__nv_bfloat162*>(&__1))[0] = __float22bfloat162_rn(((float2*)(&__2))[0]);
      *(uint1*)(x_local_cast + 0) = __1;
      *(uint1*)(((bfloat16_t*)x) + ((((((((((int)threadIdx.x) >> 5) * 1024) + ((i_9 & 1) * 512)) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + (((((((int)threadIdx.x) & 31) >> 4) + (i_9 >> 3)) & 1) * 32)) + (((((((int)threadIdx.x) & 15) >> 3) + ((i_9 & 7) >> 2)) & 1) * 16)) + (((((((int)threadIdx.x) & 7) >> 2) + ((i_9 & 3) >> 1)) & 1) * 8)) + ((((int)threadIdx.x) & 3) * 2))) = *(uint1*)(x_local_cast + 0);
    }
    __syncthreads();
    if (((int)threadIdx.x) < 64) {
      tl::cp_async_gs_conditional<4>((&(((float*)beta)[(((c & 1) * 64) + ((int)threadIdx.x))])), (&(Beta[(((((((int64_t)((int)blockIdx.x)) * (int64_t)393216) + (((int64_t)c) * (int64_t)3072)) + (((int64_t)((int)threadIdx.x)) * (int64_t)48)) + ((int64_t)((int)blockIdx.y))) + (int64_t)6144)])), ((((((int)blockIdx.x) * 64) + (c >> 1)) < 127) && (((((int)blockIdx.x) * 64) + (c >> 1)) < 127)));
    }
    tl::cp_async_commit();
    #pragma unroll
    for (int i_10 = 0; i_10 < 16; ++i_10) {
      uint2 __4;
      float4 v__3 = *(float4*)(mt + (i_10 * 4));
      (reinterpret_cast<__nv_bfloat162*>(&__4))[0] = __float22bfloat162_rn(((float2*)(&v__3))[0]);
      (reinterpret_cast<__nv_bfloat162*>(&__4))[1] = __float22bfloat162_rn(((float2*)(&v__3))[1]);
      *(uint2*)(operand + (i_10 * 4)) = __4;
    }
    {
      tl::GmmaDescriptor desc_b_1;
      tl::initialize_wgmma_descriptor<1, 1, 64>(desc_b_1, (&(((bfloat16_t*)ks)[0])));
      tl::increase_descriptor_offset<int>(desc_b_1, ((c & 1) * 16384));
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
    for (int i_11 = 0; i_11 < 8; ++i_11) {
      uint2 __5;
      float4 v__4 = *(float4*)(yz + (i_11 * 4));
      (reinterpret_cast<__nv_bfloat162*>(&__5))[0] = __float22bfloat162_rn(((float2*)(&v__4))[0]);
      (reinterpret_cast<__nv_bfloat162*>(&__5))[1] = __float22bfloat162_rn(((float2*)(&v__4))[1]);
      *(uint2*)(yz_operand + (i_11 * 4)) = __5;
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
    for (int i_12 = 0; i_12 < 64; ++i_12) {
      mt[i_12] = (mt[i_12] * gamma);
    }
    #pragma unroll
    for (int i_13 = 0; i_13 < 16; ++i_13) {
      uint2 __6;
      float4 v__5 = *(float4*)(nt + (i_13 * 4));
      (reinterpret_cast<__nv_bfloat162*>(&__6))[0] = __float22bfloat162_rn(((float2*)(&v__5))[0]);
      (reinterpret_cast<__nv_bfloat162*>(&__6))[1] = __float22bfloat162_rn(((float2*)(&v__5))[1]);
      *(uint2*)(operand + (i_13 * 4)) = __6;
    }
    {
      tl::GmmaDescriptor desc_b_3;
      tl::initialize_wgmma_descriptor<1, 1, 64>(desc_b_3, (&(((bfloat16_t*)ks)[0])));
      tl::increase_descriptor_offset<int>(desc_b_3, ((c & 1) * 16384));
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
    __syncthreads();
    #pragma unroll
    for (int i_14 = 0; i_14 < 4; ++i_14) {
      tl::cp_async_gs_conditional<16>((&(((bfloat16_t*)ks)[((((((((c & 1) * 8192) + (((((int)threadIdx.x) & 15) >> 3) * 4096)) + (i_14 * 1024)) + ((((int)threadIdx.x) >> 4) * 64)) + (((((((int)threadIdx.x) & 127) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(K[(((((((((int64_t)((int)blockIdx.x)) * (int64_t)16777216) + (((int64_t)c) * (int64_t)131072)) + (((int64_t)i_14) * (int64_t)32768)) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)4) * (int64_t)2048)) + ((((int64_t)((int)blockIdx.y)) / (int64_t)3) * (int64_t)128)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)15) * (int64_t)8)) + (int64_t)262144)])), ((((((int)blockIdx.x) * 64) + (c >> 1)) < 127) && (((((int)blockIdx.x) * 64) + (c >> 1)) < 127)));
    }
    tl::cp_async_commit();
    tl::cp_async_wait<9>();
    __syncthreads();
    #pragma unroll
    for (int i_15 = 0; i_15 < 32; ++i_15) {
      yz_operand[i_15] = ((bfloat16_t)((gamma * yz[i_15]) - (((float)((bfloat16_t*)vs)[((((((((c & 1) * 8192) + ((i_15 >> 2) * 1024)) + ((((int)threadIdx.x) & 3) * 256)) + ((i_15 & 1) * 128)) + ((((int)threadIdx.x) >> 5) * 16)) + (((i_15 & 3) >> 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2))]) * ((float*)decay)[((((i_15 >> 2) * 8) + ((((int)threadIdx.x) & 3) * 2)) + (i_15 & 1))])));
    }
    __syncthreads();
    #pragma unroll
    for (int i_16 = 0; i_16 < 4; ++i_16) {
      tl::cp_async_gs_conditional<16>((&(((bfloat16_t*)vs)[((((c & 1) * 8192) + (i_16 * 2048)) + (((int)threadIdx.x) * 8))])), (&(V[(((((((((int64_t)((int)blockIdx.x)) * (int64_t)50331648) + (((int64_t)c) * (int64_t)393216)) + (((int64_t)i_16) * (int64_t)98304)) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)4) * (int64_t)6144)) + (((int64_t)((int)blockIdx.y)) * (int64_t)128)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)15) * (int64_t)8)) + (int64_t)786432)])), ((((((int)blockIdx.x) * 64) + (c >> 1)) < 127) && (((((int)blockIdx.x) * 64) + (c >> 1)) < 127)));
    }
    tl::cp_async_commit();
    #pragma unroll
    for (int i_17 = 0; i_17 < 64; ++i_17) {
      nt[i_17] = (nt[i_17] * gamma);
    }
    {
      tl::GmmaDescriptor desc_b_4;
      tl::initialize_wgmma_descriptor<1, 512, 64>(desc_b_4, (&(((bfloat16_t*)x)[0])));
      tl::warpgroup_fence_operand(reinterpret_cast<uint32_t*>(yz_operand + 0), 16);
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(nt + 0), 128);
      tl::warpgroup_arrive();
      tl::fence_proxy_async();
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
  tl::cp_async_wait<9>();
  __syncthreads();
  int condval_4;
  if ((((int)blockIdx.x) == 1)) {
    condval_4 = 0;
  } else {
    condval_4 = 128;
  }
  if (2 <= condval_4) {
    int condval_5;
    if ((((int)blockIdx.x) == 1)) {
      condval_5 = 0;
    } else {
      condval_5 = 128;
    }
    int condval_6;
    if ((((int)blockIdx.x) == 1)) {
      condval_6 = 0;
    } else {
      condval_6 = 128;
    }
    gamma = exp2f((((float*)g)[(((condval_6 & 1) * 64) + 63)] * 0x1.71547652b82fep+0f/*1.442695e+00*/));
    if (((int)threadIdx.x) < 64) {
      int condval_7;
      if ((((int)blockIdx.x) == 1)) {
        condval_7 = 0;
      } else {
        condval_7 = 128;
      }
      int condval_8;
      if ((((int)blockIdx.x) == 1)) {
        condval_8 = 0;
      } else {
        condval_8 = 128;
      }
      int condval_9;
      if ((((int)blockIdx.x) == 1)) {
        condval_9 = 0;
      } else {
        condval_9 = 128;
      }
      int condval_10;
      if ((((int)blockIdx.x) == 1)) {
        condval_10 = 0;
      } else {
        condval_10 = 128;
      }
      ((float*)decay)[((int)threadIdx.x)] = exp2f(((((float*)g)[(((condval_8 & 1) * 64) + 63)] - ((float*)g)[(((condval_10 & 1) * 64) + ((int)threadIdx.x))]) * 0x1.71547652b82fep+0f/*1.442695e+00*/));
    }
  }
  tl::cp_async_wait<6>();
  __syncthreads();
  int condval_11;
  if ((((int)blockIdx.x) == 1)) {
    condval_11 = 0;
  } else {
    condval_11 = 128;
  }
  if (2 <= condval_11) {
    {
      tl::GmmaDescriptor desc_a_1;
      tl::GmmaDescriptor desc_b_5;
      tl::initialize_wgmma_descriptor<1, 0, 64>(desc_a_1, (&(((bfloat16_t*)a)[0])));
      int condval_12;
      if ((((int)blockIdx.x) == 1)) {
        condval_12 = 0;
      } else {
        condval_12 = 128;
      }
      tl::increase_descriptor_offset<int>(desc_a_1, ((condval_12 & 1) * 8192));
      tl::initialize_wgmma_descriptor<1, 512, 64>(desc_b_5, (&(((bfloat16_t*)ks)[0])));
      int condval_13;
      if ((((int)blockIdx.x) == 1)) {
        condval_13 = 0;
      } else {
        condval_13 = 128;
      }
      tl::increase_descriptor_offset<int>(desc_b_5, ((condval_13 & 1) * 16384));
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(xf + 0), 32);
      tl::warpgroup_arrive();
      tl::fence_proxy_async();
      #pragma unroll
      for (int ki_5 = 0; ki_5 < 4; ++ki_5) {
        tl::wgmma_ss<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 64, 64, 16, true, true, 1, 1>(uint64_t(desc_a_1 + ((ki_5 * 2048) >> 4)), uint64_t(desc_b_5 + ((((((int)threadIdx.x) >> 7) * 8192) + (ki_5 * 2048)) >> 4)), ((uint32_t*)(xf + 0)), ((0 < ki_5) ? 1 : 0));
      }
      tl::warpgroup_commit_batch();
      tl::warpgroup_wait<0>();
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(xf + 0), 32);
    }
  }
  tl::cp_async_wait<7>();
  __syncthreads();
  int condval_14;
  if ((((int)blockIdx.x) == 1)) {
    condval_14 = 0;
  } else {
    condval_14 = 128;
  }
  if (2 <= condval_14) {
    #pragma unroll
    for (int i_18 = 0; i_18 < 16; ++i_18) {
      float beta_local_cast_3[2];
      bfloat16_t x_local_cast_2[2];
      int condval_15;
      if ((((int)blockIdx.x) == 1)) {
        condval_15 = 0;
      } else {
        condval_15 = 128;
      }
      int condval_16;
      if ((((int)blockIdx.x) == 1)) {
        condval_16 = 0;
      } else {
        condval_16 = 128;
      }
      *(float2*)(beta_local_cast_3 + 0) = make_float2(((float*)beta)[(((((condval_16 & 1) * 64) + (((((int)threadIdx.x) & 127) >> 5) * 16)) + ((i_18 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2))], ((float*)beta)[(((((condval_16 & 1) * 64) + (((((int)threadIdx.x) & 127) >> 5) * 16)) + ((i_18 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2))]);
      float broadcast_var_2 = -0x1p+0f/*-1.000000e+00*/;
      uint1 __7;
      float2 __8;
        float2 __9;
          float2 v__6 = *(float2*)(beta_local_cast_3 + 0);
          float2 v__7 = make_float2(broadcast_var_2, broadcast_var_2);
          __9.x = (v__6.x*v__7.x);
          __9.y = (v__6.y*v__7.y);
        float2 v__8 = *(float2*)(xf + (i_18 * 2));
        __8.x = (__9.x*v__8.x);
        __8.y = (__9.y*v__8.y);
      (reinterpret_cast<__nv_bfloat162*>(&__7))[0] = __float22bfloat162_rn(((float2*)(&__8))[0]);
      *(uint1*)(x_local_cast_2 + 0) = __7;
      *(uint1*)(((bfloat16_t*)x) + ((((((((((int)threadIdx.x) >> 5) * 1024) + ((i_18 & 1) * 512)) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + (((((((int)threadIdx.x) & 31) >> 4) + (i_18 >> 3)) & 1) * 32)) + (((((((int)threadIdx.x) & 15) >> 3) + ((i_18 & 7) >> 2)) & 1) * 16)) + (((((((int)threadIdx.x) & 7) >> 2) + ((i_18 & 3) >> 1)) & 1) * 8)) + ((((int)threadIdx.x) & 3) * 2))) = *(uint1*)(x_local_cast_2 + 0);
    }
    #pragma unroll
    for (int i_19 = 0; i_19 < 16; ++i_19) {
      uint2 __10;
      float4 v__9 = *(float4*)(mt + (i_19 * 4));
      (reinterpret_cast<__nv_bfloat162*>(&__10))[0] = __float22bfloat162_rn(((float2*)(&v__9))[0]);
      (reinterpret_cast<__nv_bfloat162*>(&__10))[1] = __float22bfloat162_rn(((float2*)(&v__9))[1]);
      *(uint2*)(operand + (i_19 * 4)) = __10;
    }
  }
  __syncthreads();
  int condval_17;
  if ((((int)blockIdx.x) == 1)) {
    condval_17 = 0;
  } else {
    condval_17 = 128;
  }
  if (2 <= condval_17) {
    {
      tl::GmmaDescriptor desc_b_6;
      tl::initialize_wgmma_descriptor<1, 1, 64>(desc_b_6, (&(((bfloat16_t*)ks)[0])));
      int condval_18;
      if ((((int)blockIdx.x) == 1)) {
        condval_18 = 0;
      } else {
        condval_18 = 128;
      }
      tl::increase_descriptor_offset<int>(desc_b_6, ((condval_18 & 1) * 16384));
      tl::warpgroup_fence_operand(reinterpret_cast<uint32_t*>(operand + 0), 32);
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(yz + 0), 64);
      tl::warpgroup_arrive();
      tl::fence_proxy_async();
      #pragma unroll
      for (int ki_6 = 0; ki_6 < 8; ++ki_6) {
        tl::wgmma_rs<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 64, 64, 16, false, false, 1, 1>(reinterpret_cast<const uint32_t*>(operand + (ki_6 * 8)), uint64_t(desc_b_6 + ((((ki_6 >> 2) * 8192) + ((ki_6 & 3) * 32)) >> 4)), reinterpret_cast<uint32_t*>(yz + 0), ((0 < ki_6) ? 1 : 0));
      }
      tl::warpgroup_commit_batch();
      tl::warpgroup_wait<0>();
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(yz + 0), 64);
      tl::warpgroup_fence_operand(reinterpret_cast<uint32_t*>(operand + 0), 32);
    }
    #pragma unroll
    for (int i_20 = 0; i_20 < 8; ++i_20) {
      uint2 __11;
      float4 v__10 = *(float4*)(yz + (i_20 * 4));
      (reinterpret_cast<__nv_bfloat162*>(&__11))[0] = __float22bfloat162_rn(((float2*)(&v__10))[0]);
      (reinterpret_cast<__nv_bfloat162*>(&__11))[1] = __float22bfloat162_rn(((float2*)(&v__10))[1]);
      *(uint2*)(yz_operand + (i_20 * 4)) = __11;
    }
    {
      tl::GmmaDescriptor desc_b_7;
      tl::initialize_wgmma_descriptor<1, 512, 64>(desc_b_7, (&(((bfloat16_t*)x)[0])));
      tl::warpgroup_fence_operand(reinterpret_cast<uint32_t*>(yz_operand + 0), 16);
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(mt + 0), 128);
      tl::warpgroup_arrive();
      tl::fence_proxy_async();
      #pragma unroll
      for (int ki_7 = 0; ki_7 < 4; ++ki_7) {
        tl::wgmma_rs<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 64, 128, 16, false, true, 1, 1>(reinterpret_cast<const uint32_t*>(yz_operand + (ki_7 * 8)), uint64_t(desc_b_7 + ((ki_7 * 2048) >> 4)), reinterpret_cast<uint32_t*>(mt + 0), 1);
      }
      tl::warpgroup_commit_batch();
      tl::warpgroup_wait<0>();
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(mt + 0), 128);
      tl::warpgroup_fence_operand(reinterpret_cast<uint32_t*>(yz_operand + 0), 16);
    }
    #pragma unroll
    for (int i_21 = 0; i_21 < 64; ++i_21) {
      mt[i_21] = (mt[i_21] * gamma);
    }
    #pragma unroll
    for (int i_22 = 0; i_22 < 16; ++i_22) {
      uint2 __12;
      float4 v__11 = *(float4*)(nt + (i_22 * 4));
      (reinterpret_cast<__nv_bfloat162*>(&__12))[0] = __float22bfloat162_rn(((float2*)(&v__11))[0]);
      (reinterpret_cast<__nv_bfloat162*>(&__12))[1] = __float22bfloat162_rn(((float2*)(&v__11))[1]);
      *(uint2*)(operand + (i_22 * 4)) = __12;
    }
    {
      tl::GmmaDescriptor desc_b_8;
      tl::initialize_wgmma_descriptor<1, 1, 64>(desc_b_8, (&(((bfloat16_t*)ks)[0])));
      int condval_19;
      if ((((int)blockIdx.x) == 1)) {
        condval_19 = 0;
      } else {
        condval_19 = 128;
      }
      tl::increase_descriptor_offset<int>(desc_b_8, ((condval_19 & 1) * 16384));
      tl::warpgroup_fence_operand(reinterpret_cast<uint32_t*>(operand + 0), 32);
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(yz + 0), 64);
      tl::warpgroup_arrive();
      tl::fence_proxy_async();
      #pragma unroll
      for (int ki_8 = 0; ki_8 < 8; ++ki_8) {
        tl::wgmma_rs<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 64, 64, 16, false, false, 1, 1>(reinterpret_cast<const uint32_t*>(operand + (ki_8 * 8)), uint64_t(desc_b_8 + ((((ki_8 >> 2) * 8192) + ((ki_8 & 3) * 32)) >> 4)), reinterpret_cast<uint32_t*>(yz + 0), ((0 < ki_8) ? 1 : 0));
      }
      tl::warpgroup_commit_batch();
      tl::warpgroup_wait<0>();
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(yz + 0), 64);
      tl::warpgroup_fence_operand(reinterpret_cast<uint32_t*>(operand + 0), 32);
    }
  }
  tl::cp_async_wait<5>();
  __syncthreads();
  int condval_20;
  if ((((int)blockIdx.x) == 1)) {
    condval_20 = 0;
  } else {
    condval_20 = 128;
  }
  if (2 <= condval_20) {
    #pragma unroll
    for (int i_23 = 0; i_23 < 32; ++i_23) {
      int condval_21;
      if ((((int)blockIdx.x) == 1)) {
        condval_21 = 0;
      } else {
        condval_21 = 128;
      }
      int condval_22;
      if ((((int)blockIdx.x) == 1)) {
        condval_22 = 0;
      } else {
        condval_22 = 128;
      }
      yz_operand[i_23] = ((bfloat16_t)((gamma * yz[i_23]) - (((float)((bfloat16_t*)vs)[((((((((condval_22 & 1) * 8192) + ((i_23 >> 2) * 1024)) + ((((int)threadIdx.x) & 3) * 256)) + ((i_23 & 1) * 128)) + ((((int)threadIdx.x) >> 5) * 16)) + (((i_23 & 3) >> 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2))]) * ((float*)decay)[((((i_23 >> 2) * 8) + ((((int)threadIdx.x) & 3) * 2)) + (i_23 & 1))])));
    }
    #pragma unroll
    for (int i_24 = 0; i_24 < 64; ++i_24) {
      nt[i_24] = (nt[i_24] * gamma);
    }
    {
      tl::GmmaDescriptor desc_b_9;
      tl::initialize_wgmma_descriptor<1, 512, 64>(desc_b_9, (&(((bfloat16_t*)x)[0])));
      tl::warpgroup_fence_operand(reinterpret_cast<uint32_t*>(yz_operand + 0), 16);
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(nt + 0), 128);
      tl::warpgroup_arrive();
      tl::fence_proxy_async();
      #pragma unroll
      for (int ki_9 = 0; ki_9 < 4; ++ki_9) {
        tl::wgmma_rs<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 64, 128, 16, false, true, 1, 1>(reinterpret_cast<const uint32_t*>(yz_operand + (ki_9 * 8)), uint64_t(desc_b_9 + ((ki_9 * 2048) >> 4)), reinterpret_cast<uint32_t*>(nt + 0), 1);
      }
      tl::warpgroup_commit_batch();
      tl::warpgroup_wait<0>();
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(nt + 0), 128);
      tl::warpgroup_fence_operand(reinterpret_cast<uint32_t*>(yz_operand + 0), 16);
    }
  }
  tl::cp_async_wait<4>();
  __syncthreads();
  int condval_23;
  if ((((int)blockIdx.x) == 1)) {
    condval_23 = 0;
  } else {
    condval_23 = 128;
  }
  if (1 <= condval_23) {
    int condval_24;
    if ((((int)blockIdx.x) == 1)) {
      condval_24 = 0;
    } else {
      condval_24 = 128;
    }
    int condval_25;
    if ((((int)blockIdx.x) == 1)) {
      condval_25 = 0;
    } else {
      condval_25 = 128;
    }
    gamma = exp2f((((float*)g)[((((condval_25 + 1) & 1) * 64) + 63)] * 0x1.71547652b82fep+0f/*1.442695e+00*/));
    if (((int)threadIdx.x) < 64) {
      int condval_26;
      if ((((int)blockIdx.x) == 1)) {
        condval_26 = 0;
      } else {
        condval_26 = 128;
      }
      int condval_27;
      if ((((int)blockIdx.x) == 1)) {
        condval_27 = 0;
      } else {
        condval_27 = 128;
      }
      int condval_28;
      if ((((int)blockIdx.x) == 1)) {
        condval_28 = 0;
      } else {
        condval_28 = 128;
      }
      int condval_29;
      if ((((int)blockIdx.x) == 1)) {
        condval_29 = 0;
      } else {
        condval_29 = 128;
      }
      ((float*)decay)[((int)threadIdx.x)] = exp2f(((((float*)g)[((((condval_27 + 1) & 1) * 64) + 63)] - ((float*)g)[((((condval_29 + 1) & 1) * 64) + ((int)threadIdx.x))]) * 0x1.71547652b82fep+0f/*1.442695e+00*/));
    }
  }
  tl::cp_async_wait<1>();
  __syncthreads();
  int condval_30;
  if ((((int)blockIdx.x) == 1)) {
    condval_30 = 0;
  } else {
    condval_30 = 128;
  }
  if (1 <= condval_30) {
    {
      tl::GmmaDescriptor desc_a_2;
      tl::GmmaDescriptor desc_b_10;
      tl::initialize_wgmma_descriptor<1, 0, 64>(desc_a_2, (&(((bfloat16_t*)a)[0])));
      int condval_31;
      if ((((int)blockIdx.x) == 1)) {
        condval_31 = 0;
      } else {
        condval_31 = 128;
      }
      tl::increase_descriptor_offset<int>(desc_a_2, (((condval_31 + 1) & 1) * 8192));
      tl::initialize_wgmma_descriptor<1, 512, 64>(desc_b_10, (&(((bfloat16_t*)ks)[0])));
      int condval_32;
      if ((((int)blockIdx.x) == 1)) {
        condval_32 = 0;
      } else {
        condval_32 = 128;
      }
      tl::increase_descriptor_offset<int>(desc_b_10, (((condval_32 + 1) & 1) * 16384));
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(xf + 0), 32);
      tl::warpgroup_arrive();
      tl::fence_proxy_async();
      #pragma unroll
      for (int ki_10 = 0; ki_10 < 4; ++ki_10) {
        tl::wgmma_ss<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 64, 64, 16, true, true, 1, 1>(uint64_t(desc_a_2 + ((ki_10 * 2048) >> 4)), uint64_t(desc_b_10 + ((((((int)threadIdx.x) >> 7) * 8192) + (ki_10 * 2048)) >> 4)), ((uint32_t*)(xf + 0)), ((0 < ki_10) ? 1 : 0));
      }
      tl::warpgroup_commit_batch();
      tl::warpgroup_wait<0>();
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(xf + 0), 32);
    }
  }
  tl::cp_async_wait<2>();
  __syncthreads();
  int condval_33;
  if ((((int)blockIdx.x) == 1)) {
    condval_33 = 0;
  } else {
    condval_33 = 128;
  }
  if (1 <= condval_33) {
    #pragma unroll
    for (int i_25 = 0; i_25 < 16; ++i_25) {
      float beta_local_cast_3_1[2];
      bfloat16_t x_local_cast_2_1[2];
      int condval_34;
      if ((((int)blockIdx.x) == 1)) {
        condval_34 = 0;
      } else {
        condval_34 = 128;
      }
      int condval_35;
      if ((((int)blockIdx.x) == 1)) {
        condval_35 = 0;
      } else {
        condval_35 = 128;
      }
      *(float2*)(beta_local_cast_3_1 + 0) = make_float2(((float*)beta)[((((((condval_35 + 1) & 1) * 64) + (((((int)threadIdx.x) & 127) >> 5) * 16)) + ((i_25 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2))], ((float*)beta)[((((((condval_35 + 1) & 1) * 64) + (((((int)threadIdx.x) & 127) >> 5) * 16)) + ((i_25 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2))]);
      float broadcast_var_3 = -0x1p+0f/*-1.000000e+00*/;
      uint1 __13;
      float2 __14;
        float2 __15;
          float2 v__12 = *(float2*)(beta_local_cast_3_1 + 0);
          float2 v__13 = make_float2(broadcast_var_3, broadcast_var_3);
          __15.x = (v__12.x*v__13.x);
          __15.y = (v__12.y*v__13.y);
        float2 v__14 = *(float2*)(xf + (i_25 * 2));
        __14.x = (__15.x*v__14.x);
        __14.y = (__15.y*v__14.y);
      (reinterpret_cast<__nv_bfloat162*>(&__13))[0] = __float22bfloat162_rn(((float2*)(&__14))[0]);
      *(uint1*)(x_local_cast_2_1 + 0) = __13;
      __syncthreads();
      *(uint1*)(((bfloat16_t*)x) + ((((((((((int)threadIdx.x) >> 5) * 1024) + ((i_25 & 1) * 512)) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + (((((((int)threadIdx.x) & 31) >> 4) + (i_25 >> 3)) & 1) * 32)) + (((((((int)threadIdx.x) & 15) >> 3) + ((i_25 & 7) >> 2)) & 1) * 16)) + (((((((int)threadIdx.x) & 7) >> 2) + ((i_25 & 3) >> 1)) & 1) * 8)) + ((((int)threadIdx.x) & 3) * 2))) = *(uint1*)(x_local_cast_2_1 + 0);
    }
    #pragma unroll
    for (int i_26 = 0; i_26 < 16; ++i_26) {
      uint2 __16;
      float4 v__15 = *(float4*)(mt + (i_26 * 4));
      (reinterpret_cast<__nv_bfloat162*>(&__16))[0] = __float22bfloat162_rn(((float2*)(&v__15))[0]);
      (reinterpret_cast<__nv_bfloat162*>(&__16))[1] = __float22bfloat162_rn(((float2*)(&v__15))[1]);
      *(uint2*)(operand + (i_26 * 4)) = __16;
    }
  }
  __syncthreads();
  int condval_36;
  if ((((int)blockIdx.x) == 1)) {
    condval_36 = 0;
  } else {
    condval_36 = 128;
  }
  if (1 <= condval_36) {
    {
      tl::GmmaDescriptor desc_b_11;
      tl::initialize_wgmma_descriptor<1, 1, 64>(desc_b_11, (&(((bfloat16_t*)ks)[0])));
      int condval_37;
      if ((((int)blockIdx.x) == 1)) {
        condval_37 = 0;
      } else {
        condval_37 = 128;
      }
      tl::increase_descriptor_offset<int>(desc_b_11, (((condval_37 + 1) & 1) * 16384));
      tl::warpgroup_fence_operand(reinterpret_cast<uint32_t*>(operand + 0), 32);
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(yz + 0), 64);
      tl::warpgroup_arrive();
      tl::fence_proxy_async();
      #pragma unroll
      for (int ki_11 = 0; ki_11 < 8; ++ki_11) {
        tl::wgmma_rs<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 64, 64, 16, false, false, 1, 1>(reinterpret_cast<const uint32_t*>(operand + (ki_11 * 8)), uint64_t(desc_b_11 + ((((ki_11 >> 2) * 8192) + ((ki_11 & 3) * 32)) >> 4)), reinterpret_cast<uint32_t*>(yz + 0), ((0 < ki_11) ? 1 : 0));
      }
      tl::warpgroup_commit_batch();
      tl::warpgroup_wait<0>();
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(yz + 0), 64);
      tl::warpgroup_fence_operand(reinterpret_cast<uint32_t*>(operand + 0), 32);
    }
    #pragma unroll
    for (int i_27 = 0; i_27 < 8; ++i_27) {
      uint2 __17;
      float4 v__16 = *(float4*)(yz + (i_27 * 4));
      (reinterpret_cast<__nv_bfloat162*>(&__17))[0] = __float22bfloat162_rn(((float2*)(&v__16))[0]);
      (reinterpret_cast<__nv_bfloat162*>(&__17))[1] = __float22bfloat162_rn(((float2*)(&v__16))[1]);
      *(uint2*)(yz_operand + (i_27 * 4)) = __17;
    }
    {
      tl::GmmaDescriptor desc_b_12;
      tl::initialize_wgmma_descriptor<1, 512, 64>(desc_b_12, (&(((bfloat16_t*)x)[0])));
      tl::warpgroup_fence_operand(reinterpret_cast<uint32_t*>(yz_operand + 0), 16);
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(mt + 0), 128);
      tl::warpgroup_arrive();
      tl::fence_proxy_async();
      #pragma unroll
      for (int ki_12 = 0; ki_12 < 4; ++ki_12) {
        tl::wgmma_rs<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 64, 128, 16, false, true, 1, 1>(reinterpret_cast<const uint32_t*>(yz_operand + (ki_12 * 8)), uint64_t(desc_b_12 + ((ki_12 * 2048) >> 4)), reinterpret_cast<uint32_t*>(mt + 0), 1);
      }
      tl::warpgroup_commit_batch();
      tl::warpgroup_wait<0>();
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(mt + 0), 128);
      tl::warpgroup_fence_operand(reinterpret_cast<uint32_t*>(yz_operand + 0), 16);
    }
    #pragma unroll
    for (int i_28 = 0; i_28 < 64; ++i_28) {
      mt[i_28] = (mt[i_28] * gamma);
    }
    #pragma unroll
    for (int i_29 = 0; i_29 < 16; ++i_29) {
      uint2 __18;
      float4 v__17 = *(float4*)(nt + (i_29 * 4));
      (reinterpret_cast<__nv_bfloat162*>(&__18))[0] = __float22bfloat162_rn(((float2*)(&v__17))[0]);
      (reinterpret_cast<__nv_bfloat162*>(&__18))[1] = __float22bfloat162_rn(((float2*)(&v__17))[1]);
      *(uint2*)(operand + (i_29 * 4)) = __18;
    }
    {
      tl::GmmaDescriptor desc_b_13;
      tl::initialize_wgmma_descriptor<1, 1, 64>(desc_b_13, (&(((bfloat16_t*)ks)[0])));
      int condval_38;
      if ((((int)blockIdx.x) == 1)) {
        condval_38 = 0;
      } else {
        condval_38 = 128;
      }
      tl::increase_descriptor_offset<int>(desc_b_13, (((condval_38 + 1) & 1) * 16384));
      tl::warpgroup_fence_operand(reinterpret_cast<uint32_t*>(operand + 0), 32);
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(yz + 0), 64);
      tl::warpgroup_arrive();
      tl::fence_proxy_async();
      #pragma unroll
      for (int ki_13 = 0; ki_13 < 8; ++ki_13) {
        tl::wgmma_rs<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 64, 64, 16, false, false, 1, 1>(reinterpret_cast<const uint32_t*>(operand + (ki_13 * 8)), uint64_t(desc_b_13 + ((((ki_13 >> 2) * 8192) + ((ki_13 & 3) * 32)) >> 4)), reinterpret_cast<uint32_t*>(yz + 0), ((0 < ki_13) ? 1 : 0));
      }
      tl::warpgroup_commit_batch();
      tl::warpgroup_wait<0>();
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(yz + 0), 64);
      tl::warpgroup_fence_operand(reinterpret_cast<uint32_t*>(operand + 0), 32);
    }
  }
  tl::cp_async_wait<0>();
  __syncthreads();
  int condval_39;
  if ((((int)blockIdx.x) == 1)) {
    condval_39 = 0;
  } else {
    condval_39 = 128;
  }
  if (1 <= condval_39) {
    #pragma unroll
    for (int i_30 = 0; i_30 < 32; ++i_30) {
      int condval_40;
      if ((((int)blockIdx.x) == 1)) {
        condval_40 = 0;
      } else {
        condval_40 = 128;
      }
      int condval_41;
      if ((((int)blockIdx.x) == 1)) {
        condval_41 = 0;
      } else {
        condval_41 = 128;
      }
      yz_operand[i_30] = ((bfloat16_t)((gamma * yz[i_30]) - (((float)((bfloat16_t*)vs)[(((((((((condval_41 + 1) & 1) * 8192) + ((i_30 >> 2) * 1024)) + ((((int)threadIdx.x) & 3) * 256)) + ((i_30 & 1) * 128)) + ((((int)threadIdx.x) >> 5) * 16)) + (((i_30 & 3) >> 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2))]) * ((float*)decay)[((((i_30 >> 2) * 8) + ((((int)threadIdx.x) & 3) * 2)) + (i_30 & 1))])));
    }
    #pragma unroll
    for (int i_31 = 0; i_31 < 64; ++i_31) {
      nt[i_31] = (nt[i_31] * gamma);
    }
    {
      tl::GmmaDescriptor desc_b_14;
      tl::initialize_wgmma_descriptor<1, 512, 64>(desc_b_14, (&(((bfloat16_t*)x)[0])));
      tl::warpgroup_fence_operand(reinterpret_cast<uint32_t*>(yz_operand + 0), 16);
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(nt + 0), 128);
      tl::warpgroup_arrive();
      tl::fence_proxy_async();
      #pragma unroll
      for (int ki_14 = 0; ki_14 < 4; ++ki_14) {
        tl::wgmma_rs<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 64, 128, 16, false, true, 1, 1>(reinterpret_cast<const uint32_t*>(yz_operand + (ki_14 * 8)), uint64_t(desc_b_14 + ((ki_14 * 2048) >> 4)), reinterpret_cast<uint32_t*>(nt + 0), 1);
      }
      tl::warpgroup_commit_batch();
      tl::warpgroup_wait<0>();
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(nt + 0), 128);
      tl::warpgroup_fence_operand(reinterpret_cast<uint32_t*>(yz_operand + 0), 16);
    }
  }
  #pragma unroll
  for (int i_32 = 0; i_32 < 64; ++i_32) {
    Nout[((((((((((int)blockIdx.x) * 786432) + (((int)blockIdx.y) * 16384)) + ((i_32 >> 2) * 1024)) + ((((int)threadIdx.x) & 3) * 256)) + (((i_32 & 3) >> 1) * 128)) + ((((int)threadIdx.x) >> 5) * 16)) + ((i_32 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2))] = ((bfloat16_t)nt[((((i_32 >> 2) * 4) + ((i_32 & 1) * 2)) + ((i_32 & 3) >> 1))]);
  }
  #pragma unroll
  for (int i_33 = 0; i_33 < 64; ++i_33) {
    Mout[((((((((((int)blockIdx.x) * 786432) + (((int)blockIdx.y) * 16384)) + ((i_33 >> 2) * 1024)) + ((((int)threadIdx.x) & 3) * 256)) + (((i_33 & 3) >> 1) * 128)) + ((((int)threadIdx.x) >> 5) * 16)) + ((i_33 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2))] = ((bfloat16_t)mt[((((i_33 >> 2) * 4) + ((i_33 & 1) * 2)) + ((i_33 & 3) >> 1))]);
  }
}


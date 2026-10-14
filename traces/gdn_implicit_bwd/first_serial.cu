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
#include <tl_templates/cuda/copy_sm90.h>
#include <tl_templates/cuda/reduce.h>
#include <tl_templates/cuda/scan.h>
#include <tl_templates/cuda/ldsm.h>
#include <tl_templates/cuda/threadblock_swizzle.h>
#include <tl_templates/cuda/debug.h>
#ifdef ENABLE_BF16
#include <tl_templates/cuda/cuda_bf16_fallbacks.cuh>
#endif

extern "C" __global__ void kernel_kernel(const bfloat16_t* __restrict__ A, const float* __restrict__ B, const int* __restrict__ CU, float* __restrict__ DB, float* __restrict__ DG, float* __restrict__ DH0, const float* __restrict__ DHT, bfloat16_t* __restrict__ DK, const bfloat16_t* __restrict__ DO, bfloat16_t* __restrict__ DQ, __grid_constant__ const CUtensorMap DV_desc, const float* __restrict__ G, const bfloat16_t* __restrict__ H, const bfloat16_t* __restrict__ K, const bfloat16_t* __restrict__ Q, const bfloat16_t* __restrict__ V);
extern "C" __global__ void __launch_bounds__(256, 1) kernel_kernel(const bfloat16_t* __restrict__ A, const float* __restrict__ B, const int* __restrict__ CU, float* __restrict__ DB, float* __restrict__ DG, float* __restrict__ DH0, const float* __restrict__ DHT, bfloat16_t* __restrict__ DK, const bfloat16_t* __restrict__ DO, bfloat16_t* __restrict__ DQ, __grid_constant__ const CUtensorMap DV_desc, const float* __restrict__ G, const bfloat16_t* __restrict__ H, const bfloat16_t* __restrict__ K, const bfloat16_t* __restrict__ Q, const bfloat16_t* __restrict__ V) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  void* dhs = ((void*)((char*)buf_dyn_shmem + 0));
  void* hs = ((void*)((char*)buf_dyn_shmem + 32768));
  void* ds = ((void*)((char*)buf_dyn_shmem + 65536));
  void* ks = ((void*)((char*)buf_dyn_shmem + 81920));
  void* qs = ((void*)((char*)buf_dyn_shmem + 98304));
  void* us = ((void*)((char*)buf_dyn_shmem + 114688));
  void* vs = ((void*)((char*)buf_dyn_shmem + 131072));
  void* ws = ((void*)((char*)buf_dyn_shmem + 147456));
  void* zs = ((void*)((char*)buf_dyn_shmem + 163840));
  void* a = ((void*)((char*)buf_dyn_shmem + 180224));
  void* dps = ((void*)((char*)buf_dyn_shmem + 188416));
  void* ps = ((void*)((char*)buf_dyn_shmem + 196608));
  void* workspace = ((void*)((char*)buf_dyn_shmem + 204800));
  void* workspace_1 = ((void*)((char*)buf_dyn_shmem + 205824));
  void* workspace_2 = ((void*)((char*)buf_dyn_shmem + 206848));
  void* workspace_3 = ((void*)((char*)buf_dyn_shmem + 207872));
  void* workspace_4 = ((void*)((char*)buf_dyn_shmem + 208896));
  void* workspace_5 = ((void*)((char*)buf_dyn_shmem + 209920));
  void* workspace_6 = ((void*)((char*)buf_dyn_shmem + 210944));
  void* beta = ((void*)((char*)buf_dyn_shmem + 211968));
  void* ex = ((void*)((char*)buf_dyn_shmem + 212224));
  void* invex = ((void*)((char*)buf_dyn_shmem + 212480));
  float dh[64];
  float dg[2];
  float partial[1];
  float last[1];
  float acc[32];
  float tile[16];
  float prod[32];
  float db[2];
  float dg_clear[2];
  float dg_clear_1[2];
  float dg_clear_2[2];
  float dk[32];
  float db_clear[2];
  if (tl::tl_shuffle_elect<0>()) {
    tl::prefetch_tma_descriptor(DV_desc);
  }
  #pragma unroll
  for (int i = 0; i < 64; ++i) {
    dh[i] = DHT[(((((((((((int)blockIdx.x) * 16384) + ((((int)threadIdx.x) >> 7) * 8192)) + (((i & 31) >> 2) * 1024)) + ((((int)threadIdx.x) & 3) * 256)) + ((i & 1) * 128)) + ((i >> 5) * 64)) + (((((int)threadIdx.x) & 127) >> 5) * 16)) + (((i & 3) >> 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2))];
  }
  for (int c = 0; c < ((CU[((((int)blockIdx.x) / 48) + 1)] - CU[(((int)blockIdx.x) / 48)]) >> 6); ++c) {
    int start = ((CU[((((int)blockIdx.x) / 48) + 1)] - (c * 64)) - 64);
    __syncthreads();
    #pragma unroll
    for (int i_1 = 0; i_1 < 4; ++i_1) {
      bfloat16_t broadcast_var = bfloat16_t(0x0p+0f/*0.000000e+00*/);
      uint4 condval;
      if (((((((((int)threadIdx.x) >> 4) + start) >> 4) + i_1) < 1024) && (0 <= (((i_1 * 16) + (((int)threadIdx.x) >> 4)) + start)))) {
        condval = *(uint4*)(Q + (((((((int64_t)i_1) * (int64_t)32768) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)4) * (int64_t)2048)) + (((int64_t)start) * (int64_t)2048)) + (((((int64_t)((int)blockIdx.x)) % (int64_t)48) / (int64_t)3) * (int64_t)128)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)15) * (int64_t)8)));
      } else {
        condval = make_uint4(__pack_nv_bfloat162(broadcast_var, broadcast_var), __pack_nv_bfloat162(broadcast_var, broadcast_var), __pack_nv_bfloat162(broadcast_var, broadcast_var), __pack_nv_bfloat162(broadcast_var, broadcast_var));
      }
      *(uint4*)(((bfloat16_t*)qs) + ((((((((((int)threadIdx.x) & 15) >> 3) * 4096) + (i_1 * 1024)) + ((((int)threadIdx.x) >> 4) * 64)) + (((((((int)threadIdx.x) & 127) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))) = condval;
    }
    #pragma unroll
    for (int i_2 = 0; i_2 < 4; ++i_2) {
      bfloat16_t broadcast_var_1 = bfloat16_t(0x0p+0f/*0.000000e+00*/);
      uint4 condval_1;
      if (((((((((int)threadIdx.x) >> 4) + start) >> 4) + i_2) < 1024) && (0 <= (((i_2 * 16) + (((int)threadIdx.x) >> 4)) + start)))) {
        condval_1 = *(uint4*)(K + (((((((int64_t)i_2) * (int64_t)32768) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)4) * (int64_t)2048)) + (((int64_t)start) * (int64_t)2048)) + (((((int64_t)((int)blockIdx.x)) % (int64_t)48) / (int64_t)3) * (int64_t)128)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)15) * (int64_t)8)));
      } else {
        condval_1 = make_uint4(__pack_nv_bfloat162(broadcast_var_1, broadcast_var_1), __pack_nv_bfloat162(broadcast_var_1, broadcast_var_1), __pack_nv_bfloat162(broadcast_var_1, broadcast_var_1), __pack_nv_bfloat162(broadcast_var_1, broadcast_var_1));
      }
      *(uint4*)(((bfloat16_t*)ks) + ((((((((((int)threadIdx.x) & 15) >> 3) * 4096) + (i_2 * 1024)) + ((((int)threadIdx.x) >> 4) * 64)) + (((((((int)threadIdx.x) & 127) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))) = condval_1;
    }
    #pragma unroll
    for (int i_3 = 0; i_3 < 4; ++i_3) {
      bfloat16_t broadcast_var_2 = bfloat16_t(0x0p+0f/*0.000000e+00*/);
      uint4 condval_2;
      if (((((((((int)threadIdx.x) >> 4) + start) >> 4) + i_3) < 1024) && (0 <= (((i_3 * 16) + (((int)threadIdx.x) >> 4)) + start)))) {
        condval_2 = *(uint4*)(V + (((((((int64_t)i_3) * (int64_t)98304) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)4) * (int64_t)6144)) + (((int64_t)start) * (int64_t)6144)) + ((((int64_t)((int)blockIdx.x)) % (int64_t)48) * (int64_t)128)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)15) * (int64_t)8)));
      } else {
        condval_2 = make_uint4(__pack_nv_bfloat162(broadcast_var_2, broadcast_var_2), __pack_nv_bfloat162(broadcast_var_2, broadcast_var_2), __pack_nv_bfloat162(broadcast_var_2, broadcast_var_2), __pack_nv_bfloat162(broadcast_var_2, broadcast_var_2));
      }
      *(uint4*)(((bfloat16_t*)vs) + ((((((((((int)threadIdx.x) & 15) >> 3) * 4096) + (i_3 * 1024)) + ((((int)threadIdx.x) >> 4) * 64)) + (((((((int)threadIdx.x) & 127) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))) = condval_2;
    }
    #pragma unroll
    for (int i_4 = 0; i_4 < 4; ++i_4) {
      bfloat16_t broadcast_var_3 = bfloat16_t(0x0p+0f/*0.000000e+00*/);
      uint4 condval_3;
      if (((((((((int)threadIdx.x) >> 4) + start) >> 4) + i_4) < 1024) && (0 <= (((i_4 * 16) + (((int)threadIdx.x) >> 4)) + start)))) {
        condval_3 = *(uint4*)(DO + (((((((int64_t)i_4) * (int64_t)98304) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)4) * (int64_t)6144)) + (((int64_t)start) * (int64_t)6144)) + ((((int64_t)((int)blockIdx.x)) % (int64_t)48) * (int64_t)128)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)15) * (int64_t)8)));
      } else {
        condval_3 = make_uint4(__pack_nv_bfloat162(broadcast_var_3, broadcast_var_3), __pack_nv_bfloat162(broadcast_var_3, broadcast_var_3), __pack_nv_bfloat162(broadcast_var_3, broadcast_var_3), __pack_nv_bfloat162(broadcast_var_3, broadcast_var_3));
      }
      *(uint4*)(((bfloat16_t*)ds) + ((((((((((int)threadIdx.x) & 15) >> 3) * 4096) + (i_4 * 1024)) + ((((int)threadIdx.x) >> 4) * 64)) + (((((((int)threadIdx.x) & 127) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))) = condval_3;
    }
    #pragma unroll
    for (int i_5 = 0; i_5 < 2; ++i_5) {
      bfloat16_t broadcast_var_4 = bfloat16_t(0x0p+0f/*0.000000e+00*/);
      uint4 condval_4;
      if (((((((((int)threadIdx.x) >> 3) + start) >> 5) + i_5) < 512) && (0 <= (((i_5 * 32) + (((int)threadIdx.x) >> 3)) + start)))) {
        condval_4 = *(uint4*)(A + (((((((int64_t)i_5) * (int64_t)98304) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)3) * (int64_t)3072)) + (((int64_t)start) * (int64_t)3072)) + ((((int64_t)((int)blockIdx.x)) % (int64_t)48) * (int64_t)64)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)7) * (int64_t)8)));
      } else {
        condval_4 = make_uint4(__pack_nv_bfloat162(broadcast_var_4, broadcast_var_4), __pack_nv_bfloat162(broadcast_var_4, broadcast_var_4), __pack_nv_bfloat162(broadcast_var_4, broadcast_var_4), __pack_nv_bfloat162(broadcast_var_4, broadcast_var_4));
      }
      *(uint4*)(((bfloat16_t*)a) + (((((i_5 * 2048) + ((((int)threadIdx.x) >> 3) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))) = condval_4;
    }
    if (((int)threadIdx.x) < 64) {
      float condval_5;
      if ((((start + ((int)threadIdx.x)) < 16384) && (0 <= (start + ((int)threadIdx.x))))) {
        condval_5 = B[(((((int64_t)start) * (int64_t)48) + (((int64_t)((int)threadIdx.x)) * (int64_t)48)) + (((int64_t)((int)blockIdx.x)) % (int64_t)48))];
      } else {
        condval_5 = 0x0p+0f/*0.000000e+00*/;
      }
      ((float*)beta)[((int)threadIdx.x)] = condval_5;
      float condval_6;
      if (((0 <= (start + ((int)threadIdx.x))) && ((start + ((int)threadIdx.x)) < 16384))) {
        condval_6 = G[(((((int64_t)start) * (int64_t)48) + (((int64_t)((int)threadIdx.x)) * (int64_t)48)) + (((int64_t)((int)blockIdx.x)) % (int64_t)48))];
      } else {
        condval_6 = 0x0p+0f/*0.000000e+00*/;
      }
      ((float*)ex)[((int)threadIdx.x)] = exp2f((condval_6 * 0x1.71547652b82fep+0f/*1.442695e+00*/));
      float condval_7;
      if (((0 <= (start + ((int)threadIdx.x))) && ((start + ((int)threadIdx.x)) < 16384))) {
        condval_7 = G[(((((int64_t)start) * (int64_t)48) + (((int64_t)((int)threadIdx.x)) * (int64_t)48)) + (((int64_t)((int)blockIdx.x)) % (int64_t)48))];
      } else {
        condval_7 = 0x0p+0f/*0.000000e+00*/;
      }
      ((float*)invex)[((int)threadIdx.x)] = exp2f(((condval_7 * -0x1p+0f/*-1.000000e+00*/) * 0x1.71547652b82fep+0f/*1.442695e+00*/));
    }
    #pragma unroll
    for (int i_6 = 0; i_6 < 64; ++i_6) {
      bfloat16_t condval_8;
      if (((0 <= start) && (start < 16384))) {
        condval_8 = H[((((((((int64_t)start) >> (int64_t)6) * (int64_t)786432) + ((((int64_t)((int)blockIdx.x)) % (int64_t)48) * (int64_t)16384)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)127) * (int64_t)128)) + (((int64_t)i_6) * (int64_t)2)) + (((int64_t)((int)threadIdx.x)) >> (int64_t)7))];
      } else {
        condval_8 = bfloat16_t(0x0p+0f/*0.000000e+00*/);
      }
      ((bfloat16_t*)hs)[(((((((((((int)threadIdx.x) & 127) >> 6) * 8192) + (i_6 * 128)) + ((((int)threadIdx.x) >> 7) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((i_6 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + (i_6 & 1)) & 1) * 16)) + ((((((int)threadIdx.x) >> 7) + ((((int)threadIdx.x) & 15) >> 3)) & 1) * 8)) + (((int)threadIdx.x) & 7))] = condval_8;
    }
    __syncthreads();
    #pragma unroll
    for (int i_7 = 0; i_7 < 8; ++i_7) {
      tl::ptx_stmatrix_m8n8_x4((&(((bfloat16_t*)dhs)[((((((((int)threadIdx.x) >> 7) * 8192) + ((i_7 >> 2) * 4096)) + (((((int)threadIdx.x) & 127) >> 5) * 1024)) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + ((i_7 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_7 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511))])), __pack_half2(((bfloat16_t)dh[(i_7 * 8)]), ((bfloat16_t)dh[((i_7 * 8) + 1)])), __pack_half2(((bfloat16_t)dh[((i_7 * 8) + 2)]), ((bfloat16_t)dh[((i_7 * 8) + 3)])), __pack_half2(((bfloat16_t)dh[((i_7 * 8) + 4)]), ((bfloat16_t)dh[((i_7 * 8) + 5)])), __pack_half2(((bfloat16_t)dh[((i_7 * 8) + 6)]), ((bfloat16_t)dh[((i_7 * 8) + 7)])));
    }
    float broadcast_var_5 = 0x0p+0f/*0.000000e+00*/;
    *(float2*)(dg + 0) = make_float2(broadcast_var_5, broadcast_var_5);
    partial[0] = 0x0p+0f/*0.000000e+00*/;
    __syncthreads();
    #pragma unroll
    for (int i_8 = 0; i_8 < 32; ++i_8) {
      bfloat16_t hs_local_cast[2];
      *(uint1*)(hs_local_cast + 0) = *(uint1*)(((bfloat16_t*)hs) + ((((((((((((int)threadIdx.x) >> 7) * 8192) + ((i_8 >> 4) * 4096)) + (((((int)threadIdx.x) & 127) >> 5) * 1024)) + (((i_8 & 15) >> 3) * 512)) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + (((((((int)threadIdx.x) & 31) >> 4) + ((i_8 & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 15) >> 3) + ((i_8 & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 7) >> 2) + (i_8 & 1)) & 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)));
      for (int vec = 0; vec < 2; ++vec) {
        partial[0] = (partial[0] + (dh[(((((i_8 >> 4) * 32) + ((i_8 & 7) * 4)) + (((i_8 & 15) >> 3) * 2)) + vec)] * ((float)hs_local_cast[vec])));
      }
    }
    last[0] = 0x0p+0f/*0.000000e+00*/;
    last[0] = (last[0] + partial[0]);
    __syncthreads();
    last[0] = tl::AllReduce<tl::SumOp, 256, 1, 0, tl::NamedBarrier<256>>::run(last[0], (&(((float*)workspace_1)[0])));
    __syncthreads();
    #pragma unroll
    for (int i_9 = 0; i_9 < 2; ++i_9) {
      if ((((((((int)threadIdx.x) & 127) >> 5) * 16) + (i_9 * 8)) + ((((int)threadIdx.x) & 31) >> 2)) == 63) {
        dg[i_9] = (((float*)ex)[63] * last[0]);
      }
    }
    {
      tl::GmmaDescriptor desc_a;
      tl::GmmaDescriptor desc_b;
      tl::initialize_wgmma_descriptor<1, 1, 64>(desc_a, (&(((bfloat16_t*)ks)[0])));
      tl::initialize_wgmma_descriptor<1, 1, 64>(desc_b, (&(((bfloat16_t*)hs)[0])));
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc + 0), 32);
      tl::warpgroup_arrive();
      tl::fence_proxy_async();
      #pragma unroll
      for (int ki = 0; ki < 8; ++ki) {
        tl::wgmma_ss<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 64, 64, 16, false, false, 1, 1>(uint64_t(desc_a + ((((ki >> 2) * 8192) + ((ki & 3) * 32)) >> 4)), uint64_t(desc_b + (((((ki >> 2) * 16384) + ((((int)threadIdx.x) >> 7) * 8192)) + ((ki & 3) * 32)) >> 4)), ((uint32_t*)(acc + 0)), ((0 < ki) ? 1 : 0));
      }
      tl::warpgroup_commit_batch();
      tl::warpgroup_wait<0>();
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc + 0), 32);
    }
    #pragma unroll
    for (int i_10 = 0; i_10 < 16; ++i_10) {
      __syncthreads();
      uint1 __1;
      float2 __2;
        float2 __3;
          float2 __4;
          uint1 v_ = *(uint1*)(((bfloat16_t*)vs) + ((((((((((int)threadIdx.x) >> 5) * 1024) + ((i_10 & 1) * 512)) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + (((((((int)threadIdx.x) & 31) >> 4) + (i_10 >> 3)) & 1) * 32)) + (((((((int)threadIdx.x) & 15) >> 3) + ((i_10 & 7) >> 2)) & 1) * 16)) + (((((((int)threadIdx.x) & 7) >> 2) + ((i_10 & 3) >> 1)) & 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)));
          ((float2*)(&__4))[0] = __bfloat1622float2((reinterpret_cast<__nv_bfloat162*>(&v_))[0]);
          float2 v__1 = make_float2(((float*)invex)[(((((((int)threadIdx.x) & 127) >> 5) * 16) + ((i_10 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2))], ((float*)invex)[(((((((int)threadIdx.x) & 127) >> 5) * 16) + ((i_10 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2))]);
          __3.x = (__4.x*v__1.x);
          __3.y = (__4.y*v__1.y);
        float2 v__2 = *(float2*)(acc + (i_10 * 2));
        __2.x = (__3.x-v__2.x);
        __2.y = (__3.y-v__2.y);
      (reinterpret_cast<__nv_bfloat162*>(&__1))[0] = __float22bfloat162_rn(((float2*)(&__2))[0]);
      *(uint1*)(((bfloat16_t*)ws) + ((((((((((int)threadIdx.x) >> 5) * 1024) + ((i_10 & 1) * 512)) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + (((((((int)threadIdx.x) & 31) >> 4) + (i_10 >> 3)) & 1) * 32)) + (((((((int)threadIdx.x) & 15) >> 3) + ((i_10 & 7) >> 2)) & 1) * 16)) + (((((((int)threadIdx.x) & 7) >> 2) + ((i_10 & 3) >> 1)) & 1) * 8)) + ((((int)threadIdx.x) & 3) * 2))) = __1;
      uint1 __5;
      float2 __6;
        float2 v__3 = make_float2(((float*)beta)[(((((((int)threadIdx.x) & 127) >> 5) * 16) + ((i_10 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2))], ((float*)beta)[(((((((int)threadIdx.x) & 127) >> 5) * 16) + ((i_10 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2))]);
        float2 __7;
        uint1 v__4 = *(uint1*)(((bfloat16_t*)ws) + ((((((((((int)threadIdx.x) >> 5) * 1024) + ((i_10 & 1) * 512)) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + (((((((int)threadIdx.x) & 31) >> 4) + (i_10 >> 3)) & 1) * 32)) + (((((((int)threadIdx.x) & 15) >> 3) + ((i_10 & 7) >> 2)) & 1) * 16)) + (((((((int)threadIdx.x) & 7) >> 2) + ((i_10 & 3) >> 1)) & 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)));
        ((float2*)(&__7))[0] = __bfloat1622float2((reinterpret_cast<__nv_bfloat162*>(&v__4))[0]);
        __6.x = (v__3.x*__7.x);
        __6.y = (v__3.y*__7.y);
      (reinterpret_cast<__nv_bfloat162*>(&__5))[0] = __float22bfloat162_rn(((float2*)(&__6))[0]);
      *(uint1*)(((bfloat16_t*)zs) + ((((((((((int)threadIdx.x) >> 5) * 1024) + ((i_10 & 1) * 512)) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + (((((((int)threadIdx.x) & 31) >> 4) + (i_10 >> 3)) & 1) * 32)) + (((((((int)threadIdx.x) & 15) >> 3) + ((i_10 & 7) >> 2)) & 1) * 16)) + (((((((int)threadIdx.x) & 7) >> 2) + ((i_10 & 3) >> 1)) & 1) * 8)) + ((((int)threadIdx.x) & 3) * 2))) = __5;
      uint1 __8;
      float2 __9;
        float2 __10;
        uint1 v__5 = *(uint1*)(((bfloat16_t*)ds) + ((((((((((int)threadIdx.x) >> 5) * 1024) + ((i_10 & 1) * 512)) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + (((((((int)threadIdx.x) & 31) >> 4) + (i_10 >> 3)) & 1) * 32)) + (((((((int)threadIdx.x) & 15) >> 3) + ((i_10 & 7) >> 2)) & 1) * 16)) + (((((((int)threadIdx.x) & 7) >> 2) + ((i_10 & 3) >> 1)) & 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)));
        ((float2*)(&__10))[0] = __bfloat1622float2((reinterpret_cast<__nv_bfloat162*>(&v__5))[0]);
        float2 v__6 = make_float2((0x1.6a09e667f3bcdp-4f/*8.838835e-02*/ * ((float*)ex)[(((((((int)threadIdx.x) & 127) >> 5) * 16) + ((i_10 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2))]), (0x1.6a09e667f3bcdp-4f/*8.838835e-02*/ * ((float*)ex)[(((((((int)threadIdx.x) & 127) >> 5) * 16) + ((i_10 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2))]));
        __9.x = (__10.x*v__6.x);
        __9.y = (__10.y*v__6.y);
      (reinterpret_cast<__nv_bfloat162*>(&__8))[0] = __float22bfloat162_rn(((float2*)(&__9))[0]);
      *(uint1*)(((bfloat16_t*)ds) + ((((((((((int)threadIdx.x) >> 5) * 1024) + ((i_10 & 1) * 512)) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + (((((((int)threadIdx.x) & 31) >> 4) + (i_10 >> 3)) & 1) * 32)) + (((((((int)threadIdx.x) & 15) >> 3) + ((i_10 & 7) >> 2)) & 1) * 16)) + (((((((int)threadIdx.x) & 7) >> 2) + ((i_10 & 3) >> 1)) & 1) * 8)) + ((((int)threadIdx.x) & 3) * 2))) = __8;
    }
    {
      tl::GmmaDescriptor desc_a_1;
      tl::GmmaDescriptor desc_b_1;
      __syncthreads();
      tl::initialize_wgmma_descriptor<1, 1, 64>(desc_a_1, (&(((bfloat16_t*)a)[0])));
      tl::initialize_wgmma_descriptor<1, 512, 64>(desc_b_1, (&(((bfloat16_t*)zs)[0])));
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc + 0), 32);
      tl::warpgroup_arrive();
      tl::fence_proxy_async();
      #pragma unroll
      for (int ki_1 = 0; ki_1 < 4; ++ki_1) {
        tl::wgmma_ss<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 64, 64, 16, false, true, 1, 1>(uint64_t(desc_a_1 + ((ki_1 * 32) >> 4)), uint64_t(desc_b_1 + ((((((int)threadIdx.x) >> 7) * 8192) + (ki_1 * 2048)) >> 4)), ((uint32_t*)(acc + 0)), ((0 < ki_1) ? 1 : 0));
      }
      tl::warpgroup_commit_batch();
      tl::warpgroup_wait<0>();
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc + 0), 32);
    }
    __syncthreads();
    #pragma unroll
    for (int i_11 = 0; i_11 < 4; ++i_11) {
      tl::ptx_stmatrix_m8n8_x4((&(((bfloat16_t*)us)[((((((int)threadIdx.x) >> 5) * 1024) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + (i_11 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_11 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511))])), __pack_half2(((bfloat16_t)acc[(i_11 * 8)]), ((bfloat16_t)acc[((i_11 * 8) + 1)])), __pack_half2(((bfloat16_t)acc[((i_11 * 8) + 2)]), ((bfloat16_t)acc[((i_11 * 8) + 3)])), __pack_half2(((bfloat16_t)acc[((i_11 * 8) + 4)]), ((bfloat16_t)acc[((i_11 * 8) + 5)])), __pack_half2(((bfloat16_t)acc[((i_11 * 8) + 6)]), ((bfloat16_t)acc[((i_11 * 8) + 7)])));
    }
    {
      tl::GmmaDescriptor desc_a_2;
      tl::GmmaDescriptor desc_b_2;
      tl::initialize_wgmma_descriptor<1, 1, 64>(desc_a_2, (&(((bfloat16_t*)qs)[0])));
      tl::initialize_wgmma_descriptor<1, 1, 64>(desc_b_2, (&(((bfloat16_t*)ks)[0])));
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(tile + 0), 16);
      tl::warpgroup_arrive();
      tl::fence_proxy_async();
      #pragma unroll
      for (int ki_2 = 0; ki_2 < 8; ++ki_2) {
        tl::wgmma_ss<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 64, 32, 16, false, false, 1, 1>(uint64_t(desc_a_2 + ((((ki_2 >> 2) * 8192) + ((ki_2 & 3) * 32)) >> 4)), uint64_t(desc_b_2 + (((((ki_2 >> 2) * 8192) + ((((int)threadIdx.x) >> 7) * 4096)) + ((ki_2 & 3) * 32)) >> 4)), ((uint32_t*)(tile + 0)), ((0 < ki_2) ? 1 : 0));
      }
      tl::warpgroup_commit_batch();
      tl::warpgroup_wait<0>();
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(tile + 0), 16);
    }
    __syncthreads();
    #pragma unroll
    for (int i_12 = 0; i_12 < 16; ++i_12) {
      float condval_9;
      if (((((((((int)threadIdx.x) >> 7) * 32) + ((i_12 >> 2) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + (i_12 & 1)) <= (((((((int)threadIdx.x) & 127) >> 5) * 16) + (((i_12 & 3) >> 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2)))) {
        condval_9 = tile[i_12];
      } else {
        condval_9 = 0x0p+0f/*0.000000e+00*/;
      }
      ((bfloat16_t*)ps)[((((((((((((int)threadIdx.x) & 127) >> 5) * 1024) + (((i_12 & 3) >> 1) * 512)) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + ((((((int)threadIdx.x) >> 7) + ((((int)threadIdx.x) & 31) >> 4)) & 1) * 32)) + ((((i_12 >> 3) + ((((int)threadIdx.x) & 15) >> 3)) & 1) * 16)) + (((((i_12 & 7) >> 2) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + (i_12 & 1))] = ((bfloat16_t)condval_9);
    }
    {
      tl::GmmaDescriptor desc_a_3;
      tl::GmmaDescriptor desc_b_3;
      __syncthreads();
      tl::initialize_wgmma_descriptor<1, 1, 64>(desc_a_3, (&(((bfloat16_t*)ks)[0])));
      tl::initialize_wgmma_descriptor<1, 1, 64>(desc_b_3, (&(((bfloat16_t*)dhs)[0])));
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc + 0), 32);
      tl::warpgroup_arrive();
      tl::fence_proxy_async();
      #pragma unroll
      for (int ki_3 = 0; ki_3 < 8; ++ki_3) {
        tl::wgmma_ss<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 64, 64, 16, false, false, 1, 1>(uint64_t(desc_a_3 + ((((ki_3 >> 2) * 8192) + ((ki_3 & 3) * 32)) >> 4)), uint64_t(desc_b_3 + (((((ki_3 >> 2) * 16384) + ((((int)threadIdx.x) >> 7) * 8192)) + ((ki_3 & 3) * 32)) >> 4)), ((uint32_t*)(acc + 0)), ((0 < ki_3) ? 1 : 0));
      }
      tl::warpgroup_commit_batch();
      tl::warpgroup_wait<0>();
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc + 0), 32);
    }
    partial[0] = 0x0p+0f/*0.000000e+00*/;
    #pragma unroll
    for (int i_13 = 0; i_13 < 32; ++i_13) {
      acc[(((((i_13 & 15) >> 1) * 4) + ((i_13 >> 4) * 2)) + (i_13 & 1))] = (acc[(((((i_13 & 15) >> 1) * 4) + ((i_13 >> 4) * 2)) + (i_13 & 1))] * ((float*)ex)[63]);
      partial[0] = (partial[0] + (acc[(((((i_13 & 15) >> 1) * 4) + ((i_13 >> 4) * 2)) + (i_13 & 1))] * ((float)((bfloat16_t*)us)[(((((((((((int)threadIdx.x) >> 5) * 1024) + ((i_13 >> 4) * 512)) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + (((((((int)threadIdx.x) & 31) >> 4) + ((i_13 & 15) >> 3)) & 1) * 32)) + (((((((int)threadIdx.x) & 15) >> 3) + ((i_13 & 7) >> 2)) & 1) * 16)) + (((((((int)threadIdx.x) & 7) >> 2) + ((i_13 & 3) >> 1)) & 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + (i_13 & 1))])));
    }
    last[0] = 0x0p+0f/*0.000000e+00*/;
    last[0] = (last[0] + partial[0]);
    __syncthreads();
    last[0] = tl::AllReduce<tl::SumOp, 256, 1, 0, tl::NamedBarrier<256>>::run(last[0], (&(((float*)workspace_5)[0])));
    #pragma unroll
    for (int i_14 = 0; i_14 < 2; ++i_14) {
      if ((((((((int)threadIdx.x) & 127) >> 5) * 16) + (i_14 * 8)) + ((((int)threadIdx.x) & 31) >> 2)) == 63) {
        dg[i_14] = (dg[i_14] + last[0]);
      }
    }
    {
      tl::GmmaDescriptor desc_a_4;
      tl::GmmaDescriptor desc_b_4;
      tl::initialize_wgmma_descriptor<1, 0, 64>(desc_a_4, (&(((bfloat16_t*)ps)[0])));
      tl::initialize_wgmma_descriptor<1, 512, 64>(desc_b_4, (&(((bfloat16_t*)ds)[0])));
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc + 0), 32);
      tl::warpgroup_arrive();
      #pragma unroll
      for (int ki_4 = 0; ki_4 < 4; ++ki_4) {
        tl::wgmma_ss<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 64, 64, 16, true, true, 1, 1>(uint64_t(desc_a_4 + ((ki_4 * 2048) >> 4)), uint64_t(desc_b_4 + ((((((int)threadIdx.x) >> 7) * 8192) + (ki_4 * 2048)) >> 4)), ((uint32_t*)(acc + 0)), 1);
      }
      tl::warpgroup_commit_batch();
      tl::warpgroup_wait<0>();
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc + 0), 32);
    }
    __syncthreads();
    #pragma unroll
    for (int i_15 = 0; i_15 < 4; ++i_15) {
      tl::ptx_stmatrix_m8n8_x4((&(((bfloat16_t*)zs)[((((((int)threadIdx.x) >> 5) * 1024) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + (i_15 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_15 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511))])), __pack_half2(((bfloat16_t)acc[(i_15 * 8)]), ((bfloat16_t)acc[((i_15 * 8) + 1)])), __pack_half2(((bfloat16_t)acc[((i_15 * 8) + 2)]), ((bfloat16_t)acc[((i_15 * 8) + 3)])), __pack_half2(((bfloat16_t)acc[((i_15 * 8) + 4)]), ((bfloat16_t)acc[((i_15 * 8) + 5)])), __pack_half2(((bfloat16_t)acc[((i_15 * 8) + 6)]), ((bfloat16_t)acc[((i_15 * 8) + 7)])));
    }
    {
      tl::GmmaDescriptor desc_a_5;
      tl::GmmaDescriptor desc_b_5;
      tl::initialize_wgmma_descriptor<1, 0, 64>(desc_a_5, (&(((bfloat16_t*)a)[0])));
      __syncthreads();
      tl::initialize_wgmma_descriptor<1, 512, 64>(desc_b_5, (&(((bfloat16_t*)zs)[0])));
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc + 0), 32);
      tl::warpgroup_arrive();
      tl::fence_proxy_async();
      #pragma unroll
      for (int ki_5 = 0; ki_5 < 4; ++ki_5) {
        tl::wgmma_ss<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 64, 64, 16, true, true, 1, 1>(uint64_t(desc_a_5 + ((ki_5 * 2048) >> 4)), uint64_t(desc_b_5 + ((((((int)threadIdx.x) >> 7) * 8192) + (ki_5 * 2048)) >> 4)), ((uint32_t*)(acc + 0)), ((0 < ki_5) ? 1 : 0));
      }
      tl::warpgroup_commit_batch();
      tl::warpgroup_wait<0>();
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc + 0), 32);
    }
    __syncthreads();
    #pragma unroll
    for (int i_16 = 0; i_16 < 4; ++i_16) {
      tl::ptx_stmatrix_m8n8_x4((&(((bfloat16_t*)zs)[((((((int)threadIdx.x) >> 5) * 1024) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + (i_16 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_16 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511))])), __pack_half2(((bfloat16_t)acc[(i_16 * 8)]), ((bfloat16_t)acc[((i_16 * 8) + 1)])), __pack_half2(((bfloat16_t)acc[((i_16 * 8) + 2)]), ((bfloat16_t)acc[((i_16 * 8) + 3)])), __pack_half2(((bfloat16_t)acc[((i_16 * 8) + 4)]), ((bfloat16_t)acc[((i_16 * 8) + 5)])), __pack_half2(((bfloat16_t)acc[((i_16 * 8) + 6)]), ((bfloat16_t)acc[((i_16 * 8) + 7)])));
    }
    __syncthreads();
    #pragma unroll
    for (int i_17 = 0; i_17 < 16; ++i_17) {
      bfloat16_t ws_local_cast_1[2];
      *(uint1*)(ws_local_cast_1 + 0) = *(uint1*)(((bfloat16_t*)ws) + ((((((((((int)threadIdx.x) >> 5) * 1024) + ((i_17 & 1) * 512)) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + (((((((int)threadIdx.x) & 31) >> 4) + (i_17 >> 3)) & 1) * 32)) + (((((((int)threadIdx.x) & 15) >> 3) + ((i_17 & 7) >> 2)) & 1) * 16)) + (((((((int)threadIdx.x) & 7) >> 2) + ((i_17 & 3) >> 1)) & 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)));
      float2 __11;
        float2 v__7 = *(float2*)(acc + (i_17 * 2));
        float2 __12;
        uint1 v__8 = *(uint1*)(ws_local_cast_1 + 0);
        ((float2*)(&__12))[0] = __bfloat1622float2((reinterpret_cast<__nv_bfloat162*>(&v__8))[0]);
        __11.x = (v__7.x*__12.x);
        __11.y = (v__7.y*__12.y);
      *(float2*)(prod + (i_17 * 2)) = __11;
    }
    #pragma unroll
    for (int i_18 = 0; i_18 < 2; ++i_18) {
      db[i_18] = 0x0p+0f/*0.000000e+00*/;
      #pragma unroll
      for (int rv = 0; rv < 16; ++rv) {
        db[i_18] = (db[i_18] + prod[((((rv & 7) * 4) + (i_18 * 2)) + (rv >> 3))]);
      }
      __syncthreads();
      db[i_18] = tl::AllReduce<tl::SumOp, 256, 128, 0, tl::NamedBarrier<256>>::run(db[i_18], (&(((float*)workspace_3)[0])));
      db[i_18] = tl::AllReduce<tl::SumOp, 4, 1, 0, tl::NamedBarrier<256>>::run(db[i_18]);
    }
    #pragma unroll
    for (int i_19 = 0; i_19 < 16; ++i_19) {
      __syncthreads();
      uint1 __13;
      float2 __14;
        float2 v__9 = make_float2(((float*)beta)[(((((((int)threadIdx.x) & 127) >> 5) * 16) + ((i_19 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2))], ((float*)beta)[(((((((int)threadIdx.x) & 127) >> 5) * 16) + ((i_19 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2))]);
        float2 v__10 = *(float2*)(acc + (i_19 * 2));
        __14.x = (v__9.x*v__10.x);
        __14.y = (v__9.y*v__10.y);
      (reinterpret_cast<__nv_bfloat162*>(&__13))[0] = __float22bfloat162_rn(((float2*)(&__14))[0]);
      *(uint1*)(((bfloat16_t*)ws) + ((((((((((int)threadIdx.x) >> 5) * 1024) + ((i_19 & 1) * 512)) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + (((((((int)threadIdx.x) & 31) >> 4) + (i_19 >> 3)) & 1) * 32)) + (((((((int)threadIdx.x) & 15) >> 3) + ((i_19 & 7) >> 2)) & 1) * 16)) + (((((((int)threadIdx.x) & 7) >> 2) + ((i_19 & 3) >> 1)) & 1) * 8)) + ((((int)threadIdx.x) & 3) * 2))) = __13;
      bfloat16_t broadcast_var_6 = bfloat16_t(-0x1p+0f/*-1.000000e+00*/);
      float2 __15;
        float2 __16;
          float2 __17;
          uint1 __18;
            uint1 v__11 = *(uint1*)(((bfloat16_t*)vs) + ((((((((((int)threadIdx.x) >> 5) * 1024) + ((i_19 & 1) * 512)) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + (((((((int)threadIdx.x) & 31) >> 4) + (i_19 >> 3)) & 1) * 32)) + (((((((int)threadIdx.x) & 15) >> 3) + ((i_19 & 7) >> 2)) & 1) * 16)) + (((((((int)threadIdx.x) & 7) >> 2) + ((i_19 & 3) >> 1)) & 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)));
            uint1 v__12 = make_uint1(__pack_nv_bfloat162(broadcast_var_6, broadcast_var_6));
            *(uint1*)(&(__18.x)) = tl::to_uint1(tl::mul2(tl::from_uint1<__nv_bfloat162>(*(uint1*)(&(v__11.x))), tl::from_uint1<__nv_bfloat162>(*(uint1*)(&(v__12.x)))));
          ((float2*)(&__17))[0] = __bfloat1622float2((reinterpret_cast<__nv_bfloat162*>(&__18))[0]);
          float2 v__13 = make_float2(((float*)invex)[(((((((int)threadIdx.x) & 127) >> 5) * 16) + ((i_19 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2))], ((float*)invex)[(((((((int)threadIdx.x) & 127) >> 5) * 16) + ((i_19 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2))]);
          __16.x = (__17.x*v__13.x);
          __16.y = (__17.y*v__13.y);
        float2 __19;
        uint1 v__14 = *(uint1*)(((bfloat16_t*)ws) + ((((((((((int)threadIdx.x) >> 5) * 1024) + ((i_19 & 1) * 512)) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + (((((((int)threadIdx.x) & 31) >> 4) + (i_19 >> 3)) & 1) * 32)) + (((((((int)threadIdx.x) & 15) >> 3) + ((i_19 & 7) >> 2)) & 1) * 16)) + (((((((int)threadIdx.x) & 7) >> 2) + ((i_19 & 3) >> 1)) & 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)));
        ((float2*)(&__19))[0] = __bfloat1622float2((reinterpret_cast<__nv_bfloat162*>(&v__14))[0]);
        __15.x = (__16.x*__19.x);
        __15.y = (__16.y*__19.y);
      *(float2*)(prod + (i_19 * 2)) = __15;
      uint1 __20;
      float2 __21;
        float2 v__15 = make_float2(((float*)invex)[(((((((int)threadIdx.x) & 127) >> 5) * 16) + ((i_19 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2))], ((float*)invex)[(((((((int)threadIdx.x) & 127) >> 5) * 16) + ((i_19 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2))]);
        float2 __22;
        uint1 v__16 = *(uint1*)(((bfloat16_t*)ws) + ((((((((((int)threadIdx.x) >> 5) * 1024) + ((i_19 & 1) * 512)) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + (((((((int)threadIdx.x) & 31) >> 4) + (i_19 >> 3)) & 1) * 32)) + (((((((int)threadIdx.x) & 15) >> 3) + ((i_19 & 7) >> 2)) & 1) * 16)) + (((((((int)threadIdx.x) & 7) >> 2) + ((i_19 & 3) >> 1)) & 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)));
        ((float2*)(&__22))[0] = __bfloat1622float2((reinterpret_cast<__nv_bfloat162*>(&v__16))[0]);
        __21.x = (v__15.x*__22.x);
        __21.y = (v__15.y*__22.y);
      (reinterpret_cast<__nv_bfloat162*>(&__20))[0] = __float22bfloat162_rn(((float2*)(&__21))[0]);
      *(uint1*)(((bfloat16_t*)vs) + ((((((((((int)threadIdx.x) >> 5) * 1024) + ((i_19 & 1) * 512)) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + (((((((int)threadIdx.x) & 31) >> 4) + (i_19 >> 3)) & 1) * 32)) + (((((((int)threadIdx.x) & 15) >> 3) + ((i_19 & 7) >> 2)) & 1) * 16)) + (((((((int)threadIdx.x) & 7) >> 2) + ((i_19 & 3) >> 1)) & 1) * 8)) + ((((int)threadIdx.x) & 3) * 2))) = __20;
    }
    #pragma unroll
    for (int i_20 = 0; i_20 < 2; ++i_20) {
      dg_clear[i_20] = 0x0p+0f/*0.000000e+00*/;
      #pragma unroll
      for (int rv_1 = 0; rv_1 < 16; ++rv_1) {
        dg_clear[i_20] = (dg_clear[i_20] + prod[((((rv_1 & 7) * 4) + (i_20 * 2)) + (rv_1 >> 3))]);
      }
      __syncthreads();
      dg_clear[i_20] = tl::AllReduce<tl::SumOp, 256, 128, 0, tl::NamedBarrier<256>>::run(dg_clear[i_20], (&(((float*)workspace_2)[0])));
      dg_clear[i_20] = tl::AllReduce<tl::SumOp, 4, 1, 0, tl::NamedBarrier<256>>::run(dg_clear[i_20]);
      dg[i_20] = (dg[i_20] + dg_clear[i_20]);
    }
    if (tl::tl_shuffle_elect<256>()) {
      tl::fence_proxy_async();
      tl::tma_store(DV_desc, (&(((bfloat16_t*)vs)[0])), 0, start, (((int)blockIdx.x) % 48), 0);
      tl::tma_store(DV_desc, (&(((bfloat16_t*)vs)[4096])), 64, start, (((int)blockIdx.x) % 48), 0);
      tl::tma_store_arrive();
      tl::tma_store_wait<0, true>();
    }
    {
      tl::GmmaDescriptor desc_a_6;
      tl::GmmaDescriptor desc_b_6;
      tl::initialize_wgmma_descriptor<1, 1, 64>(desc_a_6, (&(((bfloat16_t*)ds)[0])));
      tl::initialize_wgmma_descriptor<1, 1024, 64>(desc_b_6, (&(((bfloat16_t*)hs)[0])));
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc + 0), 32);
      tl::warpgroup_arrive();
      tl::fence_proxy_async();
      #pragma unroll
      for (int ki_6 = 0; ki_6 < 8; ++ki_6) {
        tl::wgmma_ss<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 64, 64, 16, false, true, 1, 1>(uint64_t(desc_a_6 + ((((ki_6 >> 2) * 8192) + ((ki_6 & 3) * 32)) >> 4)), uint64_t(desc_b_6 + ((((((int)threadIdx.x) >> 7) * 16384) + (ki_6 * 2048)) >> 4)), ((uint32_t*)(acc + 0)), ((0 < ki_6) ? 1 : 0));
      }
      tl::warpgroup_commit_batch();
      tl::warpgroup_wait<0>();
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc + 0), 32);
    }
    __syncthreads();
    #pragma unroll
    for (int i_21 = 0; i_21 < 16; ++i_21) {
      bfloat16_t qs_local_cast_2[2];
      *(uint1*)(qs_local_cast_2 + 0) = *(uint1*)(((bfloat16_t*)qs) + ((((((((((int)threadIdx.x) >> 5) * 1024) + ((i_21 & 1) * 512)) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + (((((((int)threadIdx.x) & 31) >> 4) + (i_21 >> 3)) & 1) * 32)) + (((((((int)threadIdx.x) & 15) >> 3) + ((i_21 & 7) >> 2)) & 1) * 16)) + (((((((int)threadIdx.x) & 7) >> 2) + ((i_21 & 3) >> 1)) & 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)));
      float2 __23;
        float2 v__17 = *(float2*)(acc + (i_21 * 2));
        float2 __24;
        uint1 v__18 = *(uint1*)(qs_local_cast_2 + 0);
        ((float2*)(&__24))[0] = __bfloat1622float2((reinterpret_cast<__nv_bfloat162*>(&v__18))[0]);
        __23.x = (v__17.x*__24.x);
        __23.y = (v__17.y*__24.y);
      *(float2*)(prod + (i_21 * 2)) = __23;
    }
    #pragma unroll
    for (int i_22 = 0; i_22 < 2; ++i_22) {
      dg_clear_1[i_22] = 0x0p+0f/*0.000000e+00*/;
      #pragma unroll
      for (int rv_2 = 0; rv_2 < 16; ++rv_2) {
        dg_clear_1[i_22] = (dg_clear_1[i_22] + prod[((((rv_2 & 7) * 4) + (i_22 * 2)) + (rv_2 >> 3))]);
      }
      __syncthreads();
      dg_clear_1[i_22] = tl::AllReduce<tl::SumOp, 256, 128, 0, tl::NamedBarrier<256>>::run(dg_clear_1[i_22], (&(((float*)workspace_4)[0])));
      dg_clear_1[i_22] = tl::AllReduce<tl::SumOp, 4, 1, 0, tl::NamedBarrier<256>>::run(dg_clear_1[i_22]);
      dg[i_22] = (dg[i_22] + dg_clear_1[i_22]);
    }
    __syncthreads();
    #pragma unroll
    for (int i_23 = 0; i_23 < 4; ++i_23) {
      tl::ptx_stmatrix_m8n8_x4((&(((bfloat16_t*)vs)[((((((int)threadIdx.x) >> 5) * 1024) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + (i_23 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_23 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511))])), __pack_half2(((bfloat16_t)acc[(i_23 * 8)]), ((bfloat16_t)acc[((i_23 * 8) + 1)])), __pack_half2(((bfloat16_t)acc[((i_23 * 8) + 2)]), ((bfloat16_t)acc[((i_23 * 8) + 3)])), __pack_half2(((bfloat16_t)acc[((i_23 * 8) + 4)]), ((bfloat16_t)acc[((i_23 * 8) + 5)])), __pack_half2(((bfloat16_t)acc[((i_23 * 8) + 6)]), ((bfloat16_t)acc[((i_23 * 8) + 7)])));
    }
    {
      tl::GmmaDescriptor desc_a_7;
      tl::GmmaDescriptor desc_b_7;
      tl::initialize_wgmma_descriptor<1, 1, 64>(desc_a_7, (&(((bfloat16_t*)ds)[0])));
      tl::initialize_wgmma_descriptor<1, 1, 64>(desc_b_7, (&(((bfloat16_t*)us)[0])));
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(tile + 0), 16);
      tl::warpgroup_arrive();
      tl::fence_proxy_async();
      #pragma unroll
      for (int ki_7 = 0; ki_7 < 8; ++ki_7) {
        tl::wgmma_ss<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 64, 32, 16, false, false, 1, 1>(uint64_t(desc_a_7 + ((((ki_7 >> 2) * 8192) + ((ki_7 & 3) * 32)) >> 4)), uint64_t(desc_b_7 + (((((ki_7 >> 2) * 8192) + ((((int)threadIdx.x) >> 7) * 4096)) + ((ki_7 & 3) * 32)) >> 4)), ((uint32_t*)(tile + 0)), ((0 < ki_7) ? 1 : 0));
      }
      tl::warpgroup_commit_batch();
      tl::warpgroup_wait<0>();
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(tile + 0), 16);
    }
    __syncthreads();
    #pragma unroll
    for (int i_24 = 0; i_24 < 16; ++i_24) {
      float condval_10;
      if (((((((((int)threadIdx.x) >> 7) * 32) + ((i_24 >> 2) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + (i_24 & 1)) <= (((((((int)threadIdx.x) & 127) >> 5) * 16) + (((i_24 & 3) >> 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2)))) {
        condval_10 = tile[i_24];
      } else {
        condval_10 = 0x0p+0f/*0.000000e+00*/;
      }
      tile[i_24] = condval_10;
      ((bfloat16_t*)dps)[((((((((((((int)threadIdx.x) & 127) >> 5) * 1024) + (((i_24 & 3) >> 1) * 512)) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + ((((((int)threadIdx.x) >> 7) + ((((int)threadIdx.x) & 31) >> 4)) & 1) * 32)) + ((((i_24 >> 3) + ((((int)threadIdx.x) & 15) >> 3)) & 1) * 16)) + (((((i_24 & 7) >> 2) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + (i_24 & 1))] = ((bfloat16_t)tile[i_24]);
      tile[i_24] = (tile[i_24] * ((float)((bfloat16_t*)ps)[((((((((((((int)threadIdx.x) & 127) >> 5) * 1024) + (((i_24 & 3) >> 1) * 512)) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + ((((((int)threadIdx.x) >> 7) + ((((int)threadIdx.x) & 31) >> 4)) & 1) * 32)) + ((((i_24 >> 3) + ((((int)threadIdx.x) & 15) >> 3)) & 1) * 16)) + (((((i_24 & 7) >> 2) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + (i_24 & 1))]));
    }
    #pragma unroll
    for (int i_25 = 0; i_25 < 2; ++i_25) {
      dg_clear_2[i_25] = 0x0p+0f/*0.000000e+00*/;
      #pragma unroll
      for (int rv_3 = 0; rv_3 < 8; ++rv_3) {
        dg_clear_2[i_25] = (dg_clear_2[i_25] + tile[((((rv_3 & 3) * 4) + (i_25 * 2)) + (rv_3 >> 2))]);
      }
      __syncthreads();
      dg_clear_2[i_25] = tl::AllReduce<tl::SumOp, 256, 128, 0, tl::NamedBarrier<256>>::run(dg_clear_2[i_25], (&(((float*)workspace)[0])));
      dg_clear_2[i_25] = tl::AllReduce<tl::SumOp, 4, 1, 0, tl::NamedBarrier<256>>::run(dg_clear_2[i_25]);
      dg[i_25] = (dg[i_25] + dg_clear_2[i_25]);
    }
    __syncthreads();
    #pragma unroll
    for (int i_26 = 0; i_26 < 16; ++i_26) {
      bfloat16_t vs_local_cast_3[2];
      *(uint1*)(vs_local_cast_3 + 0) = *(uint1*)(((bfloat16_t*)vs) + ((((((((((int)threadIdx.x) >> 5) * 1024) + ((i_26 & 1) * 512)) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + (((((((int)threadIdx.x) & 31) >> 4) + (i_26 >> 3)) & 1) * 32)) + (((((((int)threadIdx.x) & 15) >> 3) + ((i_26 & 7) >> 2)) & 1) * 16)) + (((((((int)threadIdx.x) & 7) >> 2) + ((i_26 & 3) >> 1)) & 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)));
      float2 __25;
      uint1 v__19 = *(uint1*)(vs_local_cast_3 + 0);
      ((float2*)(&__25))[0] = __bfloat1622float2((reinterpret_cast<__nv_bfloat162*>(&v__19))[0]);
      *(float2*)(acc + (i_26 * 2)) = __25;
    }
    {
      tl::GmmaDescriptor desc_a_8;
      tl::GmmaDescriptor desc_b_8;
      tl::initialize_wgmma_descriptor<1, 1, 64>(desc_a_8, (&(((bfloat16_t*)dps)[0])));
      tl::initialize_wgmma_descriptor<1, 512, 64>(desc_b_8, (&(((bfloat16_t*)ks)[0])));
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc + 0), 32);
      tl::warpgroup_arrive();
      tl::fence_proxy_async();
      #pragma unroll
      for (int ki_8 = 0; ki_8 < 4; ++ki_8) {
        tl::wgmma_ss<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 64, 64, 16, false, true, 1, 1>(uint64_t(desc_a_8 + ((ki_8 * 32) >> 4)), uint64_t(desc_b_8 + ((((((int)threadIdx.x) >> 7) * 8192) + (ki_8 * 2048)) >> 4)), ((uint32_t*)(acc + 0)), 1);
      }
      tl::warpgroup_commit_batch();
      tl::warpgroup_wait<0>();
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc + 0), 32);
    }
    #pragma unroll
    for (int i_27 = 0; i_27 < 16; ++i_27) {
      bfloat16_t DQ_local_cast_4[2];
      if ((((((((((int)threadIdx.x) & 127) >> 5) * 16) + ((i_27 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2)) + start) < 16384) && (0 <= ((((((((int)threadIdx.x) & 127) >> 5) * 16) + ((i_27 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2)) + start))) {
        uint1 __26;
        float2 v__20 = *(float2*)(acc + (i_27 * 2));
        (reinterpret_cast<__nv_bfloat162*>(&__26))[0] = __float22bfloat162_rn(((float2*)(&v__20))[0]);
        *(uint1*)(DQ_local_cast_4 + 0) = __26;
        *(uint1*)(DQ + ((((((((((((int64_t)((int)threadIdx.x)) & (int64_t)127) >> (int64_t)5) * (int64_t)98304) + ((((int64_t)i_27) & (int64_t)1) * (int64_t)49152)) + (((((int64_t)((int)threadIdx.x)) & (int64_t)31) >> (int64_t)2) * (int64_t)6144)) + (((int64_t)start) * (int64_t)6144)) + ((((int64_t)((int)blockIdx.x)) % (int64_t)48) * (int64_t)128)) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)7) * (int64_t)64)) + ((((int64_t)i_27) >> (int64_t)1) * (int64_t)8)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)3) * (int64_t)2))) = *(uint1*)(DQ_local_cast_4 + 0);
      }
    }
    {
      tl::GmmaDescriptor desc_a_9;
      tl::GmmaDescriptor desc_b_9;
      tl::initialize_wgmma_descriptor<1, 1, 64>(desc_a_9, (&(((bfloat16_t*)us)[0])));
      tl::initialize_wgmma_descriptor<1, 1024, 64>(desc_b_9, (&(((bfloat16_t*)dhs)[0])));
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(dk + 0), 32);
      tl::warpgroup_arrive();
      #pragma unroll
      for (int ki_9 = 0; ki_9 < 8; ++ki_9) {
        tl::wgmma_ss<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 64, 64, 16, false, true, 1, 1>(uint64_t(desc_a_9 + ((((ki_9 >> 2) * 8192) + ((ki_9 & 3) * 32)) >> 4)), uint64_t(desc_b_9 + ((((((int)threadIdx.x) >> 7) * 16384) + (ki_9 * 2048)) >> 4)), ((uint32_t*)(dk + 0)), ((0 < ki_9) ? 1 : 0));
      }
      tl::warpgroup_commit_batch();
      tl::warpgroup_wait<0>();
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(dk + 0), 32);
    }
    #pragma unroll
    for (int i_28 = 0; i_28 < 8; ++i_28) {
      float4 __27;
        float4 v__21 = *(float4*)(dk + (i_28 * 4));
        float4 v__22 = make_float4(((float*)ex)[63], ((float*)ex)[63], ((float*)ex)[63], ((float*)ex)[63]);
        __27.x = (v__21.x*v__22.x);
        __27.y = (v__21.y*v__22.y);
        __27.z = (v__21.z*v__22.z);
        __27.w = (v__21.w*v__22.w);
      *(float4*)(dk + (i_28 * 4)) = __27;
    }
    {
      tl::GmmaDescriptor desc_a_10;
      tl::GmmaDescriptor desc_b_10;
      tl::initialize_wgmma_descriptor<1, 0, 64>(desc_a_10, (&(((bfloat16_t*)dps)[0])));
      tl::initialize_wgmma_descriptor<1, 512, 64>(desc_b_10, (&(((bfloat16_t*)qs)[0])));
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(dk + 0), 32);
      tl::warpgroup_arrive();
      #pragma unroll
      for (int ki_10 = 0; ki_10 < 4; ++ki_10) {
        tl::wgmma_ss<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 64, 64, 16, true, true, 1, 1>(uint64_t(desc_a_10 + ((ki_10 * 2048) >> 4)), uint64_t(desc_b_10 + ((((((int)threadIdx.x) >> 7) * 8192) + (ki_10 * 2048)) >> 4)), ((uint32_t*)(dk + 0)), 1);
      }
      tl::warpgroup_commit_batch();
      tl::warpgroup_wait<0>();
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(dk + 0), 32);
    }
    {
      tl::GmmaDescriptor desc_a_11;
      tl::GmmaDescriptor desc_b_11;
      tl::initialize_wgmma_descriptor<1, 1, 64>(desc_a_11, (&(((bfloat16_t*)ws)[0])));
      tl::initialize_wgmma_descriptor<1, 1024, 64>(desc_b_11, (&(((bfloat16_t*)hs)[0])));
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc + 0), 32);
      tl::warpgroup_arrive();
      #pragma unroll
      for (int ki_11 = 0; ki_11 < 8; ++ki_11) {
        tl::wgmma_ss<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 64, 64, 16, false, true, 1, 1>(uint64_t(desc_a_11 + ((((ki_11 >> 2) * 8192) + ((ki_11 & 3) * 32)) >> 4)), uint64_t(desc_b_11 + ((((((int)threadIdx.x) >> 7) * 16384) + (ki_11 * 2048)) >> 4)), ((uint32_t*)(acc + 0)), ((0 < ki_11) ? 1 : 0));
      }
      tl::warpgroup_commit_batch();
      tl::warpgroup_wait<0>();
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc + 0), 32);
    }
    #pragma unroll
    for (int i_29 = 0; i_29 < 32; ++i_29) {
      dk[i_29] = (dk[i_29] - acc[i_29]);
    }
    {
      tl::GmmaDescriptor desc_a_12;
      tl::GmmaDescriptor desc_b_12;
      tl::initialize_wgmma_descriptor<1, 1, 64>(desc_a_12, (&(((bfloat16_t*)zs)[0])));
      tl::initialize_wgmma_descriptor<1, 1, 64>(desc_b_12, (&(((bfloat16_t*)us)[0])));
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(tile + 0), 16);
      tl::warpgroup_arrive();
      #pragma unroll
      for (int ki_12 = 0; ki_12 < 8; ++ki_12) {
        tl::wgmma_ss<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 64, 32, 16, false, false, 1, 1>(uint64_t(desc_a_12 + ((((ki_12 >> 2) * 8192) + ((ki_12 & 3) * 32)) >> 4)), uint64_t(desc_b_12 + (((((ki_12 >> 2) * 8192) + ((((int)threadIdx.x) >> 7) * 4096)) + ((ki_12 & 3) * 32)) >> 4)), ((uint32_t*)(tile + 0)), ((0 < ki_12) ? 1 : 0));
      }
      tl::warpgroup_commit_batch();
      tl::warpgroup_wait<0>();
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(tile + 0), 16);
    }
    __syncthreads();
    #pragma unroll
    for (int i_30 = 0; i_30 < 16; ++i_30) {
      float condval_11;
      if (((((((((int)threadIdx.x) >> 7) * 32) + ((i_30 >> 2) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + (i_30 & 1)) < (((((((int)threadIdx.x) & 127) >> 5) * 16) + (((i_30 & 3) >> 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2)))) {
        condval_11 = (tile[i_30] * -0x1p+0f/*-1.000000e+00*/);
      } else {
        condval_11 = 0x0p+0f/*0.000000e+00*/;
      }
      ((bfloat16_t*)a)[((((((((((((int)threadIdx.x) & 127) >> 5) * 1024) + (((i_30 & 3) >> 1) * 512)) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + ((((((int)threadIdx.x) >> 7) + ((((int)threadIdx.x) & 31) >> 4)) & 1) * 32)) + ((((i_30 >> 3) + ((((int)threadIdx.x) & 15) >> 3)) & 1) * 16)) + (((((i_30 & 7) >> 2) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + (i_30 & 1))] = ((bfloat16_t)condval_11);
    }
    {
      tl::GmmaDescriptor desc_a_13;
      tl::GmmaDescriptor desc_b_13;
      __syncthreads();
      tl::initialize_wgmma_descriptor<1, 1, 64>(desc_a_13, (&(((bfloat16_t*)a)[0])));
      tl::initialize_wgmma_descriptor<1, 512, 64>(desc_b_13, (&(((bfloat16_t*)ks)[0])));
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc + 0), 32);
      tl::warpgroup_arrive();
      tl::fence_proxy_async();
      #pragma unroll
      for (int ki_13 = 0; ki_13 < 4; ++ki_13) {
        tl::wgmma_ss<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 64, 64, 16, false, true, 1, 1>(uint64_t(desc_a_13 + ((ki_13 * 32) >> 4)), uint64_t(desc_b_13 + ((((((int)threadIdx.x) >> 7) * 8192) + (ki_13 * 2048)) >> 4)), ((uint32_t*)(acc + 0)), ((0 < ki_13) ? 1 : 0));
      }
      tl::warpgroup_commit_batch();
      tl::warpgroup_wait<0>();
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc + 0), 32);
    }
    __syncthreads();
    #pragma unroll
    for (int i_31 = 0; i_31 < 16; ++i_31) {
      float2 __28;
        float2 v__23 = *(float2*)(acc + (i_31 * 2));
        float2 __29;
        uint1 v__24 = *(uint1*)(((bfloat16_t*)ks) + ((((((((((int)threadIdx.x) >> 5) * 1024) + ((i_31 & 1) * 512)) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + (((((((int)threadIdx.x) & 31) >> 4) + (i_31 >> 3)) & 1) * 32)) + (((((((int)threadIdx.x) & 15) >> 3) + ((i_31 & 7) >> 2)) & 1) * 16)) + (((((((int)threadIdx.x) & 7) >> 2) + ((i_31 & 3) >> 1)) & 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)));
        ((float2*)(&__29))[0] = __bfloat1622float2((reinterpret_cast<__nv_bfloat162*>(&v__24))[0]);
        __28.x = (v__23.x*__29.x);
        __28.y = (v__23.y*__29.y);
      *(float2*)(prod + (i_31 * 2)) = __28;
      float2 __30;
        float2 v__25 = *(float2*)(dk + (i_31 * 2));
        float2 __31;
          float2 v__26 = make_float2(((float*)beta)[(((((((int)threadIdx.x) & 127) >> 5) * 16) + ((i_31 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2))], ((float*)beta)[(((((((int)threadIdx.x) & 127) >> 5) * 16) + ((i_31 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2))]);
          float2 v__27 = *(float2*)(acc + (i_31 * 2));
          __31.x = (v__26.x*v__27.x);
          __31.y = (v__26.y*v__27.y);
        __30.x = (v__25.x+__31.x);
        __30.y = (v__25.y+__31.y);
      *(float2*)(dk + (i_31 * 2)) = __30;
      uint1 __32;
      float2 __33;
        float2 v__28 = make_float2(((float*)beta)[(((((((int)threadIdx.x) & 127) >> 5) * 16) + ((i_31 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2))], ((float*)beta)[(((((((int)threadIdx.x) & 127) >> 5) * 16) + ((i_31 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2))]);
        float2 __34;
        uint1 v__29 = *(uint1*)(((bfloat16_t*)ks) + ((((((((((int)threadIdx.x) >> 5) * 1024) + ((i_31 & 1) * 512)) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + (((((((int)threadIdx.x) & 31) >> 4) + (i_31 >> 3)) & 1) * 32)) + (((((((int)threadIdx.x) & 15) >> 3) + ((i_31 & 7) >> 2)) & 1) * 16)) + (((((((int)threadIdx.x) & 7) >> 2) + ((i_31 & 3) >> 1)) & 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)));
        ((float2*)(&__34))[0] = __bfloat1622float2((reinterpret_cast<__nv_bfloat162*>(&v__29))[0]);
        __33.x = (v__28.x*__34.x);
        __33.y = (v__28.y*__34.y);
      (reinterpret_cast<__nv_bfloat162*>(&__32))[0] = __float22bfloat162_rn(((float2*)(&__33))[0]);
      *(uint1*)(((bfloat16_t*)vs) + ((((((((((int)threadIdx.x) >> 5) * 1024) + ((i_31 & 1) * 512)) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + (((((((int)threadIdx.x) & 31) >> 4) + (i_31 >> 3)) & 1) * 32)) + (((((((int)threadIdx.x) & 15) >> 3) + ((i_31 & 7) >> 2)) & 1) * 16)) + (((((((int)threadIdx.x) & 7) >> 2) + ((i_31 & 3) >> 1)) & 1) * 8)) + ((((int)threadIdx.x) & 3) * 2))) = __32;
    }
    #pragma unroll
    for (int i_32 = 0; i_32 < 2; ++i_32) {
      db_clear[i_32] = 0x0p+0f/*0.000000e+00*/;
      #pragma unroll
      for (int rv_4 = 0; rv_4 < 16; ++rv_4) {
        db_clear[i_32] = (db_clear[i_32] + prod[((((rv_4 & 7) * 4) + (i_32 * 2)) + (rv_4 >> 3))]);
      }
      __syncthreads();
      db_clear[i_32] = tl::AllReduce<tl::SumOp, 256, 128, 0, tl::NamedBarrier<256>>::run(db_clear[i_32], (&(((float*)workspace_6)[0])));
      db_clear[i_32] = tl::AllReduce<tl::SumOp, 4, 1, 0, tl::NamedBarrier<256>>::run(db_clear[i_32]);
      db[i_32] = (db[i_32] + db_clear[i_32]);
    }
    {
      tl::GmmaDescriptor desc_a_14;
      tl::GmmaDescriptor desc_b_14;
      tl::initialize_wgmma_descriptor<1, 0, 64>(desc_a_14, (&(((bfloat16_t*)a)[0])));
      tl::initialize_wgmma_descriptor<1, 512, 64>(desc_b_14, (&(((bfloat16_t*)vs)[0])));
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(dk + 0), 32);
      tl::warpgroup_arrive();
      tl::fence_proxy_async();
      #pragma unroll
      for (int ki_14 = 0; ki_14 < 4; ++ki_14) {
        tl::wgmma_ss<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 64, 64, 16, true, true, 1, 1>(uint64_t(desc_a_14 + ((ki_14 * 2048) >> 4)), uint64_t(desc_b_14 + ((((((int)threadIdx.x) >> 7) * 8192) + (ki_14 * 2048)) >> 4)), ((uint32_t*)(dk + 0)), 1);
      }
      tl::warpgroup_commit_batch();
      tl::warpgroup_wait<0>();
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(dk + 0), 32);
    }
    #pragma unroll
    for (int i_33 = 0; i_33 < 16; ++i_33) {
      bfloat16_t DK_local_cast_5[2];
      if ((((((((((int)threadIdx.x) & 127) >> 5) * 16) + ((i_33 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2)) + start) < 16384) && (0 <= ((((((((int)threadIdx.x) & 127) >> 5) * 16) + ((i_33 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2)) + start))) {
        uint1 __35;
        float2 v__30 = *(float2*)(dk + (i_33 * 2));
        (reinterpret_cast<__nv_bfloat162*>(&__35))[0] = __float22bfloat162_rn(((float2*)(&v__30))[0]);
        *(uint1*)(DK_local_cast_5 + 0) = __35;
        *(uint1*)(DK + ((((((((((((int64_t)((int)threadIdx.x)) & (int64_t)127) >> (int64_t)5) * (int64_t)98304) + ((((int64_t)i_33) & (int64_t)1) * (int64_t)49152)) + (((((int64_t)((int)threadIdx.x)) & (int64_t)31) >> (int64_t)2) * (int64_t)6144)) + (((int64_t)start) * (int64_t)6144)) + ((((int64_t)((int)blockIdx.x)) % (int64_t)48) * (int64_t)128)) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)7) * (int64_t)64)) + ((((int64_t)i_33) >> (int64_t)1) * (int64_t)8)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)3) * (int64_t)2))) = *(uint1*)(DK_local_cast_5 + 0);
      }
    }
    if ((((((int)threadIdx.x) & 3) * 2) + (((int)threadIdx.x) >> 7)) == 0) {
      #pragma unroll
      for (int i_34 = 0; i_34 < 2; ++i_34) {
        if (0 <= ((((((((int)threadIdx.x) & 127) >> 5) * 16) + (i_34 * 8)) + ((((int)threadIdx.x) & 31) >> 2)) + start)) {
          if (((((((((int)threadIdx.x) & 127) >> 5) * 16) + (i_34 * 8)) + ((((int)threadIdx.x) & 31) >> 2)) + start) < 16384) {
            DG[(((((((((int64_t)((int)threadIdx.x)) & (int64_t)127) >> (int64_t)5) * (int64_t)768) + (((int64_t)i_34) * (int64_t)384)) + (((((int64_t)((int)threadIdx.x)) & (int64_t)31) >> (int64_t)2) * (int64_t)48)) + (((int64_t)start) * (int64_t)48)) + (((int64_t)((int)blockIdx.x)) % (int64_t)48))] = dg[i_34];
            DB[(((((((((int64_t)((int)threadIdx.x)) & (int64_t)127) >> (int64_t)5) * (int64_t)768) + (((int64_t)i_34) * (int64_t)384)) + (((((int64_t)((int)threadIdx.x)) & (int64_t)31) >> (int64_t)2) * (int64_t)48)) + (((int64_t)start) * (int64_t)48)) + (((int64_t)((int)blockIdx.x)) % (int64_t)48))] = db[i_34];
          }
        }
      }
    }
    __syncthreads();
    #pragma unroll
    for (int i_35 = 0; i_35 < 16; ++i_35) {
      float4 __36;
        float4 v__31 = *(float4*)(dh + (i_35 * 4));
        float4 v__32 = make_float4(((float*)ex)[63], ((float*)ex)[63], ((float*)ex)[63], ((float*)ex)[63]);
        __36.x = (v__31.x*v__32.x);
        __36.y = (v__31.y*v__32.y);
        __36.z = (v__31.z*v__32.z);
        __36.w = (v__31.w*v__32.w);
      *(float4*)(dh + (i_35 * 4)) = __36;
    }
    {
      tl::GmmaDescriptor desc_a_15;
      tl::GmmaDescriptor desc_b_15;
      tl::initialize_wgmma_descriptor<1, 512, 64>(desc_a_15, (&(((bfloat16_t*)ds)[0])));
      tl::initialize_wgmma_descriptor<1, 512, 64>(desc_b_15, (&(((bfloat16_t*)qs)[0])));
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(dh + 0), 64);
      tl::warpgroup_arrive();
      #pragma unroll
      for (int i_36 = 0; i_36 < 2; ++i_36) {
        #pragma unroll
        for (int ki_15 = 0; ki_15 < 4; ++ki_15) {
          tl::wgmma_ss<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 64, 64, 16, true, true, 1, 1>(uint64_t(desc_a_15 + (((i_36 * 8192) + (ki_15 * 2048)) >> 4)), uint64_t(desc_b_15 + ((((((int)threadIdx.x) >> 7) * 8192) + (ki_15 * 2048)) >> 4)), ((uint32_t*)(dh + (i_36 * 32))), 1);
        }
      }
      tl::warpgroup_commit_batch();
      tl::warpgroup_wait<0>();
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(dh + 0), 64);
    }
    #pragma unroll
    for (int i_37 = 0; i_37 < 4; ++i_37) {
      __syncthreads();
      bfloat16_t broadcast_var_7 = bfloat16_t(-0x1p+0f/*-1.000000e+00*/);
      uint4 __37;
        uint4 v__33 = *(uint4*)(((bfloat16_t*)ws) + ((((((((((int)threadIdx.x) & 15) >> 3) * 4096) + (i_37 * 1024)) + ((((int)threadIdx.x) >> 4) * 64)) + (((((((int)threadIdx.x) & 127) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)));
        uint4 v__34 = make_uint4(__pack_nv_bfloat162(broadcast_var_7, broadcast_var_7), __pack_nv_bfloat162(broadcast_var_7, broadcast_var_7), __pack_nv_bfloat162(broadcast_var_7, broadcast_var_7), __pack_nv_bfloat162(broadcast_var_7, broadcast_var_7));
        *(uint1*)(&(__37.x)) = tl::to_uint1(tl::mul2(tl::from_uint1<__nv_bfloat162>(*(uint1*)(&(v__33.x))), tl::from_uint1<__nv_bfloat162>(*(uint1*)(&(v__34.x)))));
        *(uint1*)(&(__37.y)) = tl::to_uint1(tl::mul2(tl::from_uint1<__nv_bfloat162>(*(uint1*)(&(v__33.y))), tl::from_uint1<__nv_bfloat162>(*(uint1*)(&(v__34.y)))));
        *(uint1*)(&(__37.z)) = tl::to_uint1(tl::mul2(tl::from_uint1<__nv_bfloat162>(*(uint1*)(&(v__33.z))), tl::from_uint1<__nv_bfloat162>(*(uint1*)(&(v__34.z)))));
        *(uint1*)(&(__37.w)) = tl::to_uint1(tl::mul2(tl::from_uint1<__nv_bfloat162>(*(uint1*)(&(v__33.w))), tl::from_uint1<__nv_bfloat162>(*(uint1*)(&(v__34.w)))));
      *(uint4*)(((bfloat16_t*)ws) + ((((((((((int)threadIdx.x) & 15) >> 3) * 4096) + (i_37 * 1024)) + ((((int)threadIdx.x) >> 4) * 64)) + (((((((int)threadIdx.x) & 127) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))) = __37;
    }
    {
      tl::GmmaDescriptor desc_a_16;
      tl::GmmaDescriptor desc_b_16;
      __syncthreads();
      tl::initialize_wgmma_descriptor<1, 512, 64>(desc_a_16, (&(((bfloat16_t*)ws)[0])));
      tl::initialize_wgmma_descriptor<1, 512, 64>(desc_b_16, (&(((bfloat16_t*)ks)[0])));
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(dh + 0), 64);
      tl::warpgroup_arrive();
      tl::fence_proxy_async();
      #pragma unroll
      for (int i_38 = 0; i_38 < 2; ++i_38) {
        #pragma unroll
        for (int ki_16 = 0; ki_16 < 4; ++ki_16) {
          tl::wgmma_ss<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 64, 64, 16, true, true, 1, 1>(uint64_t(desc_a_16 + (((i_38 * 8192) + (ki_16 * 2048)) >> 4)), uint64_t(desc_b_16 + ((((((int)threadIdx.x) >> 7) * 8192) + (ki_16 * 2048)) >> 4)), ((uint32_t*)(dh + (i_38 * 32))), 1);
        }
      }
      tl::warpgroup_commit_batch();
      tl::warpgroup_wait<0>();
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(dh + 0), 64);
    }
  }
  #pragma unroll
  for (int i_39 = 0; i_39 < 64; ++i_39) {
    DH0[(((((((((((int)blockIdx.x) * 16384) + ((((int)threadIdx.x) >> 7) * 8192)) + (((i_39 & 31) >> 2) * 1024)) + ((((int)threadIdx.x) & 3) * 256)) + ((i_39 & 1) * 128)) + ((i_39 >> 5) * 64)) + (((((int)threadIdx.x) & 127) >> 5) * 16)) + (((i_39 & 3) >> 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2))] = dh[i_39];
  }
}


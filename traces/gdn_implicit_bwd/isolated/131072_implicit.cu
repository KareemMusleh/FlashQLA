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

extern "C" __global__ void kernel_kernel(__grid_constant__ const CUtensorMap A_desc, const float* __restrict__ B, const int* __restrict__ CU, float* __restrict__ DB, float* __restrict__ DG, float* __restrict__ DH0, const float* __restrict__ DHT, __grid_constant__ const CUtensorMap DK_desc, __grid_constant__ const CUtensorMap DO_desc, __grid_constant__ const CUtensorMap DQ_desc, __grid_constant__ const CUtensorMap DV_desc, const float* __restrict__ G, __grid_constant__ const CUtensorMap H_desc, __grid_constant__ const CUtensorMap K_desc, __grid_constant__ const CUtensorMap Q_desc, __grid_constant__ const CUtensorMap V_desc);
extern "C" __global__ void __launch_bounds__(256, 1) kernel_kernel(__grid_constant__ const CUtensorMap A_desc, const float* __restrict__ B, const int* __restrict__ CU, float* __restrict__ DB, float* __restrict__ DG, float* __restrict__ DH0, const float* __restrict__ DHT, __grid_constant__ const CUtensorMap DK_desc, __grid_constant__ const CUtensorMap DO_desc, __grid_constant__ const CUtensorMap DQ_desc, __grid_constant__ const CUtensorMap DV_desc, const float* __restrict__ G, __grid_constant__ const CUtensorMap H_desc, __grid_constant__ const CUtensorMap K_desc, __grid_constant__ const CUtensorMap Q_desc, __grid_constant__ const CUtensorMap V_desc) {
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
  __shared__ __align__(16) uint64_t input_ready_mem[1];
  auto input_ready = reinterpret_cast<Barrier*>(input_ready_mem);
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
    tl::prefetch_tma_descriptor(Q_desc);
    tl::prefetch_tma_descriptor(K_desc);
    tl::prefetch_tma_descriptor(V_desc);
    tl::prefetch_tma_descriptor(DO_desc);
    tl::prefetch_tma_descriptor(A_desc);
    tl::prefetch_tma_descriptor(H_desc);
    tl::prefetch_tma_descriptor(DV_desc);
    tl::prefetch_tma_descriptor(DQ_desc);
    tl::prefetch_tma_descriptor(DK_desc);
  }
  if (tl::tl_shuffle_elect<0>()) {
    input_ready[0].init(32);
  }
  tl::fence_barrier_init();
  __syncthreads();
  #pragma unroll
  for (int i = 0; i < 64; ++i) {
    dh[i] = DHT[(((((((((((int)blockIdx.x) * 16384) + ((((int)threadIdx.x) >> 7) * 8192)) + (((i & 31) >> 2) * 1024)) + ((((int)threadIdx.x) & 3) * 256)) + ((i & 1) * 128)) + ((i >> 5) * 64)) + (((((int)threadIdx.x) & 127) >> 5) * 16)) + (((i & 3) >> 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2))];
  }
  for (int c = 0; c < ((CU[((((int)blockIdx.x) / 48) + 1)] - CU[(((int)blockIdx.x) / 48)]) >> 6); ++c) {
    int start = ((CU[((((int)blockIdx.x) / 48) + 1)] - (c * 64)) - 64);
    __syncthreads();
    if (((int)threadIdx.x) < 32) {
      if (tl::tl_shuffle_elect<32>()) {
        input_ready[0].expect_transaction(16384);
        tl::tma_load(Q_desc, input_ready[0], (&(((bfloat16_t*)qs)[0])), 0, start, ((((int)blockIdx.x) % 48) / 3), 0);
        tl::tma_load(Q_desc, input_ready[0], (&(((bfloat16_t*)qs)[4096])), 64, start, ((((int)blockIdx.x) % 48) / 3), 0);
        input_ready[0].expect_transaction(16384);
        tl::tma_load(K_desc, input_ready[0], (&(((bfloat16_t*)ks)[0])), 0, start, ((((int)blockIdx.x) % 48) / 3), 0);
        tl::tma_load(K_desc, input_ready[0], (&(((bfloat16_t*)ks)[4096])), 64, start, ((((int)blockIdx.x) % 48) / 3), 0);
        input_ready[0].expect_transaction(16384);
        tl::tma_load(V_desc, input_ready[0], (&(((bfloat16_t*)vs)[0])), 0, start, (((int)blockIdx.x) % 48), 0);
        tl::tma_load(V_desc, input_ready[0], (&(((bfloat16_t*)vs)[4096])), 64, start, (((int)blockIdx.x) % 48), 0);
        input_ready[0].expect_transaction(16384);
        tl::tma_load(DO_desc, input_ready[0], (&(((bfloat16_t*)ds)[0])), 0, start, (((int)blockIdx.x) % 48), 0);
        tl::tma_load(DO_desc, input_ready[0], (&(((bfloat16_t*)ds)[4096])), 64, start, (((int)blockIdx.x) % 48), 0);
        input_ready[0].expect_transaction(8192);
        tl::tma_load(A_desc, input_ready[0], (&(((bfloat16_t*)a)[0])), 0, start, (((int)blockIdx.x) % 48), 0);
        input_ready[0].expect_transaction(32768);
        tl::tma_load(H_desc, input_ready[0], (&(((bfloat16_t*)hs)[0])), 0, 0, (((int)blockIdx.x) % 48), (start >> 6), 0);
        tl::tma_load(H_desc, input_ready[0], (&(((bfloat16_t*)hs)[8192])), 64, 0, (((int)blockIdx.x) % 48), (start >> 6), 0);
      }
      input_ready[0].arrive();
    }
    __syncthreads();
    if (((int)threadIdx.x) < 64) {
      float condval;
      if ((((start + ((int)threadIdx.x)) < 131072) && (0 <= (start + ((int)threadIdx.x))))) {
        condval = B[(((((int64_t)start) * (int64_t)48) + (((int64_t)((int)threadIdx.x)) * (int64_t)48)) + (((int64_t)((int)blockIdx.x)) % (int64_t)48))];
      } else {
        condval = 0x0p+0f/*0.000000e+00*/;
      }
      ((float*)beta)[((int)threadIdx.x)] = condval;
      float condval_1;
      if (((0 <= (start + ((int)threadIdx.x))) && ((start + ((int)threadIdx.x)) < 131072))) {
        condval_1 = G[(((((int64_t)start) * (int64_t)48) + (((int64_t)((int)threadIdx.x)) * (int64_t)48)) + (((int64_t)((int)blockIdx.x)) % (int64_t)48))];
      } else {
        condval_1 = 0x0p+0f/*0.000000e+00*/;
      }
      ((float*)ex)[((int)threadIdx.x)] = exp2f((condval_1 * 0x1.71547652b82fep+0f/*1.442695e+00*/));
      float condval_2;
      if (((0 <= (start + ((int)threadIdx.x))) && ((start + ((int)threadIdx.x)) < 131072))) {
        condval_2 = G[(((((int64_t)start) * (int64_t)48) + (((int64_t)((int)threadIdx.x)) * (int64_t)48)) + (((int64_t)((int)blockIdx.x)) % (int64_t)48))];
      } else {
        condval_2 = 0x0p+0f/*0.000000e+00*/;
      }
      ((float*)invex)[((int)threadIdx.x)] = exp2f(((condval_2 * -0x1p+0f/*-1.000000e+00*/) * 0x1.71547652b82fep+0f/*1.442695e+00*/));
    }
    input_ready[0].wait((c & 1));
    __syncthreads();
    #pragma unroll
    for (int i_1 = 0; i_1 < 8; ++i_1) {
      tl::ptx_stmatrix_m8n8_x4((&(((bfloat16_t*)dhs)[((((((((int)threadIdx.x) >> 7) * 8192) + ((i_1 >> 2) * 4096)) + (((((int)threadIdx.x) & 127) >> 5) * 1024)) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + ((i_1 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_1 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511))])), __pack_half2(((bfloat16_t)dh[(i_1 * 8)]), ((bfloat16_t)dh[((i_1 * 8) + 1)])), __pack_half2(((bfloat16_t)dh[((i_1 * 8) + 2)]), ((bfloat16_t)dh[((i_1 * 8) + 3)])), __pack_half2(((bfloat16_t)dh[((i_1 * 8) + 4)]), ((bfloat16_t)dh[((i_1 * 8) + 5)])), __pack_half2(((bfloat16_t)dh[((i_1 * 8) + 6)]), ((bfloat16_t)dh[((i_1 * 8) + 7)])));
    }
    float broadcast_var = 0x0p+0f/*0.000000e+00*/;
    *(float2*)(dg + 0) = make_float2(broadcast_var, broadcast_var);
    partial[0] = 0x0p+0f/*0.000000e+00*/;
    __syncthreads();
    #pragma unroll
    for (int i_2 = 0; i_2 < 64; ++i_2) {
      partial[0] = (partial[0] + (dh[(((((i_2 >> 5) * 32) + (((i_2 & 15) >> 1) * 4)) + (((i_2 & 31) >> 4) * 2)) + (i_2 & 1))] * ((float)((bfloat16_t*)hs)[((((((((((i_2 >> 5) * 8192) + ((((int)threadIdx.x) >> 7) * 4096)) + (((i_2 & 15) >> 1) * 512)) + ((((int)threadIdx.x) & 3) * 128)) + ((i_2 & 1) * 64)) + (((((((int)threadIdx.x) & 127) >> 6) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + (((int)threadIdx.x) & 1)) & 1) * 16)) + (((((i_2 & 31) >> 4) + (i_2 & 1)) & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2))])));
    }
    last[0] = 0x0p+0f/*0.000000e+00*/;
    last[0] = (last[0] + partial[0]);
    __syncthreads();
    last[0] = tl::AllReduce<tl::SumOp, 256, 1, 0, tl::NamedBarrier<256>>::run(last[0], (&(((float*)workspace_3)[0])));
    __syncthreads();
    #pragma unroll
    for (int i_3 = 0; i_3 < 2; ++i_3) {
      if ((((((((int)threadIdx.x) & 127) >> 5) * 16) + (i_3 * 8)) + ((((int)threadIdx.x) & 31) >> 2)) == 63) {
        dg[i_3] = (((float*)ex)[63] * last[0]);
      }
    }
    {
      tl::GmmaDescriptor desc_a;
      tl::GmmaDescriptor desc_b;
      tl::initialize_wgmma_descriptor<1, 1, 64>(desc_a, (&(((bfloat16_t*)ks)[0])));
      tl::initialize_wgmma_descriptor<1, 1024, 64>(desc_b, (&(((bfloat16_t*)hs)[0])));
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc + 0), 32);
      tl::warpgroup_arrive();
      tl::fence_proxy_async();
      #pragma unroll
      for (int ki = 0; ki < 8; ++ki) {
        tl::wgmma_ss<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 64, 64, 16, false, true, 1, 1>(uint64_t(desc_a + ((((ki >> 2) * 8192) + ((ki & 3) * 32)) >> 4)), uint64_t(desc_b + ((((((int)threadIdx.x) >> 7) * 16384) + (ki * 2048)) >> 4)), ((uint32_t*)(acc + 0)), ((0 < ki) ? 1 : 0));
      }
      tl::warpgroup_commit_batch();
      tl::warpgroup_wait<0>();
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc + 0), 32);
    }
    #pragma unroll
    for (int i_4 = 0; i_4 < 16; ++i_4) {
      __syncthreads();
      uint1 __1;
      float2 __2;
        float2 __3;
          float2 __4;
          uint1 v_ = *(uint1*)(((bfloat16_t*)vs) + ((((((((((int)threadIdx.x) >> 5) * 1024) + ((i_4 & 1) * 512)) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + (((((((int)threadIdx.x) & 31) >> 4) + (i_4 >> 3)) & 1) * 32)) + (((((((int)threadIdx.x) & 15) >> 3) + ((i_4 & 7) >> 2)) & 1) * 16)) + (((((((int)threadIdx.x) & 7) >> 2) + ((i_4 & 3) >> 1)) & 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)));
          ((float2*)(&__4))[0] = __bfloat1622float2((reinterpret_cast<__nv_bfloat162*>(&v_))[0]);
          float2 v__1 = make_float2(((float*)invex)[(((((((int)threadIdx.x) & 127) >> 5) * 16) + ((i_4 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2))], ((float*)invex)[(((((((int)threadIdx.x) & 127) >> 5) * 16) + ((i_4 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2))]);
          __3.x = (__4.x*v__1.x);
          __3.y = (__4.y*v__1.y);
        float2 v__2 = *(float2*)(acc + (i_4 * 2));
        __2.x = (__3.x-v__2.x);
        __2.y = (__3.y-v__2.y);
      (reinterpret_cast<__nv_bfloat162*>(&__1))[0] = __float22bfloat162_rn(((float2*)(&__2))[0]);
      *(uint1*)(((bfloat16_t*)ws) + ((((((((((int)threadIdx.x) >> 5) * 1024) + ((i_4 & 1) * 512)) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + (((((((int)threadIdx.x) & 31) >> 4) + (i_4 >> 3)) & 1) * 32)) + (((((((int)threadIdx.x) & 15) >> 3) + ((i_4 & 7) >> 2)) & 1) * 16)) + (((((((int)threadIdx.x) & 7) >> 2) + ((i_4 & 3) >> 1)) & 1) * 8)) + ((((int)threadIdx.x) & 3) * 2))) = __1;
      uint1 __5;
      float2 __6;
        float2 v__3 = make_float2(((float*)beta)[(((((((int)threadIdx.x) & 127) >> 5) * 16) + ((i_4 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2))], ((float*)beta)[(((((((int)threadIdx.x) & 127) >> 5) * 16) + ((i_4 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2))]);
        float2 __7;
        uint1 v__4 = *(uint1*)(((bfloat16_t*)ws) + ((((((((((int)threadIdx.x) >> 5) * 1024) + ((i_4 & 1) * 512)) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + (((((((int)threadIdx.x) & 31) >> 4) + (i_4 >> 3)) & 1) * 32)) + (((((((int)threadIdx.x) & 15) >> 3) + ((i_4 & 7) >> 2)) & 1) * 16)) + (((((((int)threadIdx.x) & 7) >> 2) + ((i_4 & 3) >> 1)) & 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)));
        ((float2*)(&__7))[0] = __bfloat1622float2((reinterpret_cast<__nv_bfloat162*>(&v__4))[0]);
        __6.x = (v__3.x*__7.x);
        __6.y = (v__3.y*__7.y);
      (reinterpret_cast<__nv_bfloat162*>(&__5))[0] = __float22bfloat162_rn(((float2*)(&__6))[0]);
      *(uint1*)(((bfloat16_t*)zs) + ((((((((((int)threadIdx.x) >> 5) * 1024) + ((i_4 & 1) * 512)) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + (((((((int)threadIdx.x) & 31) >> 4) + (i_4 >> 3)) & 1) * 32)) + (((((((int)threadIdx.x) & 15) >> 3) + ((i_4 & 7) >> 2)) & 1) * 16)) + (((((((int)threadIdx.x) & 7) >> 2) + ((i_4 & 3) >> 1)) & 1) * 8)) + ((((int)threadIdx.x) & 3) * 2))) = __5;
      uint1 __8;
      float2 __9;
        float2 __10;
        uint1 v__5 = *(uint1*)(((bfloat16_t*)ds) + ((((((((((int)threadIdx.x) >> 5) * 1024) + ((i_4 & 1) * 512)) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + (((((((int)threadIdx.x) & 31) >> 4) + (i_4 >> 3)) & 1) * 32)) + (((((((int)threadIdx.x) & 15) >> 3) + ((i_4 & 7) >> 2)) & 1) * 16)) + (((((((int)threadIdx.x) & 7) >> 2) + ((i_4 & 3) >> 1)) & 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)));
        ((float2*)(&__10))[0] = __bfloat1622float2((reinterpret_cast<__nv_bfloat162*>(&v__5))[0]);
        float2 v__6 = make_float2((0x1.6a09e667f3bcdp-4f/*8.838835e-02*/ * ((float*)ex)[(((((((int)threadIdx.x) & 127) >> 5) * 16) + ((i_4 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2))]), (0x1.6a09e667f3bcdp-4f/*8.838835e-02*/ * ((float*)ex)[(((((((int)threadIdx.x) & 127) >> 5) * 16) + ((i_4 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2))]));
        __9.x = (__10.x*v__6.x);
        __9.y = (__10.y*v__6.y);
      (reinterpret_cast<__nv_bfloat162*>(&__8))[0] = __float22bfloat162_rn(((float2*)(&__9))[0]);
      *(uint1*)(((bfloat16_t*)ds) + ((((((((((int)threadIdx.x) >> 5) * 1024) + ((i_4 & 1) * 512)) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + (((((((int)threadIdx.x) & 31) >> 4) + (i_4 >> 3)) & 1) * 32)) + (((((((int)threadIdx.x) & 15) >> 3) + ((i_4 & 7) >> 2)) & 1) * 16)) + (((((((int)threadIdx.x) & 7) >> 2) + ((i_4 & 3) >> 1)) & 1) * 8)) + ((((int)threadIdx.x) & 3) * 2))) = __8;
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
    for (int i_5 = 0; i_5 < 4; ++i_5) {
      tl::ptx_stmatrix_m8n8_x4((&(((bfloat16_t*)us)[((((((int)threadIdx.x) >> 5) * 1024) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + (i_5 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_5 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511))])), __pack_half2(((bfloat16_t)acc[(i_5 * 8)]), ((bfloat16_t)acc[((i_5 * 8) + 1)])), __pack_half2(((bfloat16_t)acc[((i_5 * 8) + 2)]), ((bfloat16_t)acc[((i_5 * 8) + 3)])), __pack_half2(((bfloat16_t)acc[((i_5 * 8) + 4)]), ((bfloat16_t)acc[((i_5 * 8) + 5)])), __pack_half2(((bfloat16_t)acc[((i_5 * 8) + 6)]), ((bfloat16_t)acc[((i_5 * 8) + 7)])));
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
    for (int i_6 = 0; i_6 < 16; ++i_6) {
      float condval_3;
      if (((((((((int)threadIdx.x) >> 7) * 32) + ((i_6 >> 2) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + (i_6 & 1)) <= (((((((int)threadIdx.x) & 127) >> 5) * 16) + (((i_6 & 3) >> 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2)))) {
        condval_3 = tile[i_6];
      } else {
        condval_3 = 0x0p+0f/*0.000000e+00*/;
      }
      ((bfloat16_t*)ps)[((((((((((((int)threadIdx.x) & 127) >> 5) * 1024) + (((i_6 & 3) >> 1) * 512)) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + ((((((int)threadIdx.x) >> 7) + ((((int)threadIdx.x) & 31) >> 4)) & 1) * 32)) + ((((i_6 >> 3) + ((((int)threadIdx.x) & 15) >> 3)) & 1) * 16)) + (((((i_6 & 7) >> 2) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + (i_6 & 1))] = ((bfloat16_t)condval_3);
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
    for (int i_7 = 0; i_7 < 32; ++i_7) {
      acc[(((((i_7 & 15) >> 1) * 4) + ((i_7 >> 4) * 2)) + (i_7 & 1))] = (acc[(((((i_7 & 15) >> 1) * 4) + ((i_7 >> 4) * 2)) + (i_7 & 1))] * ((float*)ex)[63]);
      partial[0] = (partial[0] + (acc[(((((i_7 & 15) >> 1) * 4) + ((i_7 >> 4) * 2)) + (i_7 & 1))] * ((float)((bfloat16_t*)us)[(((((((((((int)threadIdx.x) >> 5) * 1024) + ((i_7 >> 4) * 512)) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + (((((((int)threadIdx.x) & 31) >> 4) + ((i_7 & 15) >> 3)) & 1) * 32)) + (((((((int)threadIdx.x) & 15) >> 3) + ((i_7 & 7) >> 2)) & 1) * 16)) + (((((((int)threadIdx.x) & 7) >> 2) + ((i_7 & 3) >> 1)) & 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + (i_7 & 1))])));
    }
    last[0] = 0x0p+0f/*0.000000e+00*/;
    last[0] = (last[0] + partial[0]);
    __syncthreads();
    last[0] = tl::AllReduce<tl::SumOp, 256, 1, 0, tl::NamedBarrier<256>>::run(last[0], (&(((float*)workspace_5)[0])));
    #pragma unroll
    for (int i_8 = 0; i_8 < 2; ++i_8) {
      if ((((((((int)threadIdx.x) & 127) >> 5) * 16) + (i_8 * 8)) + ((((int)threadIdx.x) & 31) >> 2)) == 63) {
        dg[i_8] = (dg[i_8] + last[0]);
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
    for (int i_9 = 0; i_9 < 4; ++i_9) {
      tl::ptx_stmatrix_m8n8_x4((&(((bfloat16_t*)zs)[((((((int)threadIdx.x) >> 5) * 1024) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + (i_9 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_9 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511))])), __pack_half2(((bfloat16_t)acc[(i_9 * 8)]), ((bfloat16_t)acc[((i_9 * 8) + 1)])), __pack_half2(((bfloat16_t)acc[((i_9 * 8) + 2)]), ((bfloat16_t)acc[((i_9 * 8) + 3)])), __pack_half2(((bfloat16_t)acc[((i_9 * 8) + 4)]), ((bfloat16_t)acc[((i_9 * 8) + 5)])), __pack_half2(((bfloat16_t)acc[((i_9 * 8) + 6)]), ((bfloat16_t)acc[((i_9 * 8) + 7)])));
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
    for (int i_10 = 0; i_10 < 4; ++i_10) {
      tl::ptx_stmatrix_m8n8_x4((&(((bfloat16_t*)zs)[((((((int)threadIdx.x) >> 5) * 1024) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + (i_10 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_10 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511))])), __pack_half2(((bfloat16_t)acc[(i_10 * 8)]), ((bfloat16_t)acc[((i_10 * 8) + 1)])), __pack_half2(((bfloat16_t)acc[((i_10 * 8) + 2)]), ((bfloat16_t)acc[((i_10 * 8) + 3)])), __pack_half2(((bfloat16_t)acc[((i_10 * 8) + 4)]), ((bfloat16_t)acc[((i_10 * 8) + 5)])), __pack_half2(((bfloat16_t)acc[((i_10 * 8) + 6)]), ((bfloat16_t)acc[((i_10 * 8) + 7)])));
    }
    __syncthreads();
    #pragma unroll
    for (int i_11 = 0; i_11 < 16; ++i_11) {
      bfloat16_t ws_local_cast[2];
      *(uint1*)(ws_local_cast + 0) = *(uint1*)(((bfloat16_t*)ws) + ((((((((((int)threadIdx.x) >> 5) * 1024) + ((i_11 & 1) * 512)) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + (((((((int)threadIdx.x) & 31) >> 4) + (i_11 >> 3)) & 1) * 32)) + (((((((int)threadIdx.x) & 15) >> 3) + ((i_11 & 7) >> 2)) & 1) * 16)) + (((((((int)threadIdx.x) & 7) >> 2) + ((i_11 & 3) >> 1)) & 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)));
      float2 __11;
        float2 v__7 = *(float2*)(acc + (i_11 * 2));
        float2 __12;
        uint1 v__8 = *(uint1*)(ws_local_cast + 0);
        ((float2*)(&__12))[0] = __bfloat1622float2((reinterpret_cast<__nv_bfloat162*>(&v__8))[0]);
        __11.x = (v__7.x*__12.x);
        __11.y = (v__7.y*__12.y);
      *(float2*)(prod + (i_11 * 2)) = __11;
    }
    #pragma unroll
    for (int i_12 = 0; i_12 < 2; ++i_12) {
      db[i_12] = 0x0p+0f/*0.000000e+00*/;
      #pragma unroll
      for (int rv = 0; rv < 16; ++rv) {
        db[i_12] = (db[i_12] + prod[((((rv & 7) * 4) + (i_12 * 2)) + (rv >> 3))]);
      }
      __syncthreads();
      db[i_12] = tl::AllReduce<tl::SumOp, 256, 128, 0, tl::NamedBarrier<256>>::run(db[i_12], (&(((float*)workspace_6)[0])));
      db[i_12] = tl::AllReduce<tl::SumOp, 4, 1, 0, tl::NamedBarrier<256>>::run(db[i_12]);
    }
    #pragma unroll
    for (int i_13 = 0; i_13 < 16; ++i_13) {
      __syncthreads();
      uint1 __13;
      float2 __14;
        float2 v__9 = make_float2(((float*)beta)[(((((((int)threadIdx.x) & 127) >> 5) * 16) + ((i_13 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2))], ((float*)beta)[(((((((int)threadIdx.x) & 127) >> 5) * 16) + ((i_13 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2))]);
        float2 v__10 = *(float2*)(acc + (i_13 * 2));
        __14.x = (v__9.x*v__10.x);
        __14.y = (v__9.y*v__10.y);
      (reinterpret_cast<__nv_bfloat162*>(&__13))[0] = __float22bfloat162_rn(((float2*)(&__14))[0]);
      *(uint1*)(((bfloat16_t*)ws) + ((((((((((int)threadIdx.x) >> 5) * 1024) + ((i_13 & 1) * 512)) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + (((((((int)threadIdx.x) & 31) >> 4) + (i_13 >> 3)) & 1) * 32)) + (((((((int)threadIdx.x) & 15) >> 3) + ((i_13 & 7) >> 2)) & 1) * 16)) + (((((((int)threadIdx.x) & 7) >> 2) + ((i_13 & 3) >> 1)) & 1) * 8)) + ((((int)threadIdx.x) & 3) * 2))) = __13;
      bfloat16_t broadcast_var_1 = bfloat16_t(-0x1p+0f/*-1.000000e+00*/);
      float2 __15;
        float2 __16;
          float2 __17;
          uint1 __18;
            uint1 v__11 = *(uint1*)(((bfloat16_t*)vs) + ((((((((((int)threadIdx.x) >> 5) * 1024) + ((i_13 & 1) * 512)) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + (((((((int)threadIdx.x) & 31) >> 4) + (i_13 >> 3)) & 1) * 32)) + (((((((int)threadIdx.x) & 15) >> 3) + ((i_13 & 7) >> 2)) & 1) * 16)) + (((((((int)threadIdx.x) & 7) >> 2) + ((i_13 & 3) >> 1)) & 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)));
            uint1 v__12 = make_uint1(__pack_nv_bfloat162(broadcast_var_1, broadcast_var_1));
            *(uint1*)(&(__18.x)) = tl::to_uint1(tl::mul2(tl::from_uint1<__nv_bfloat162>(*(uint1*)(&(v__11.x))), tl::from_uint1<__nv_bfloat162>(*(uint1*)(&(v__12.x)))));
          ((float2*)(&__17))[0] = __bfloat1622float2((reinterpret_cast<__nv_bfloat162*>(&__18))[0]);
          float2 v__13 = make_float2(((float*)invex)[(((((((int)threadIdx.x) & 127) >> 5) * 16) + ((i_13 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2))], ((float*)invex)[(((((((int)threadIdx.x) & 127) >> 5) * 16) + ((i_13 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2))]);
          __16.x = (__17.x*v__13.x);
          __16.y = (__17.y*v__13.y);
        float2 __19;
        uint1 v__14 = *(uint1*)(((bfloat16_t*)ws) + ((((((((((int)threadIdx.x) >> 5) * 1024) + ((i_13 & 1) * 512)) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + (((((((int)threadIdx.x) & 31) >> 4) + (i_13 >> 3)) & 1) * 32)) + (((((((int)threadIdx.x) & 15) >> 3) + ((i_13 & 7) >> 2)) & 1) * 16)) + (((((((int)threadIdx.x) & 7) >> 2) + ((i_13 & 3) >> 1)) & 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)));
        ((float2*)(&__19))[0] = __bfloat1622float2((reinterpret_cast<__nv_bfloat162*>(&v__14))[0]);
        __15.x = (__16.x*__19.x);
        __15.y = (__16.y*__19.y);
      *(float2*)(prod + (i_13 * 2)) = __15;
      uint1 __20;
      float2 __21;
        float2 v__15 = make_float2(((float*)invex)[(((((((int)threadIdx.x) & 127) >> 5) * 16) + ((i_13 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2))], ((float*)invex)[(((((((int)threadIdx.x) & 127) >> 5) * 16) + ((i_13 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2))]);
        float2 __22;
        uint1 v__16 = *(uint1*)(((bfloat16_t*)ws) + ((((((((((int)threadIdx.x) >> 5) * 1024) + ((i_13 & 1) * 512)) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + (((((((int)threadIdx.x) & 31) >> 4) + (i_13 >> 3)) & 1) * 32)) + (((((((int)threadIdx.x) & 15) >> 3) + ((i_13 & 7) >> 2)) & 1) * 16)) + (((((((int)threadIdx.x) & 7) >> 2) + ((i_13 & 3) >> 1)) & 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)));
        ((float2*)(&__22))[0] = __bfloat1622float2((reinterpret_cast<__nv_bfloat162*>(&v__16))[0]);
        __21.x = (v__15.x*__22.x);
        __21.y = (v__15.y*__22.y);
      (reinterpret_cast<__nv_bfloat162*>(&__20))[0] = __float22bfloat162_rn(((float2*)(&__21))[0]);
      *(uint1*)(((bfloat16_t*)vs) + ((((((((((int)threadIdx.x) >> 5) * 1024) + ((i_13 & 1) * 512)) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + (((((((int)threadIdx.x) & 31) >> 4) + (i_13 >> 3)) & 1) * 32)) + (((((((int)threadIdx.x) & 15) >> 3) + ((i_13 & 7) >> 2)) & 1) * 16)) + (((((((int)threadIdx.x) & 7) >> 2) + ((i_13 & 3) >> 1)) & 1) * 8)) + ((((int)threadIdx.x) & 3) * 2))) = __20;
    }
    #pragma unroll
    for (int i_14 = 0; i_14 < 2; ++i_14) {
      dg_clear[i_14] = 0x0p+0f/*0.000000e+00*/;
      #pragma unroll
      for (int rv_1 = 0; rv_1 < 16; ++rv_1) {
        dg_clear[i_14] = (dg_clear[i_14] + prod[((((rv_1 & 7) * 4) + (i_14 * 2)) + (rv_1 >> 3))]);
      }
      __syncthreads();
      dg_clear[i_14] = tl::AllReduce<tl::SumOp, 256, 128, 0, tl::NamedBarrier<256>>::run(dg_clear[i_14], (&(((float*)workspace_2)[0])));
      dg_clear[i_14] = tl::AllReduce<tl::SumOp, 4, 1, 0, tl::NamedBarrier<256>>::run(dg_clear[i_14]);
      dg[i_14] = (dg[i_14] + dg_clear[i_14]);
    }
    tl::fence_proxy_async();
    __syncthreads();
    if (((int)threadIdx.x) < 32) {
      if (tl::tl_shuffle_elect<32>()) {
        tl::tma_store(DV_desc, (&(((bfloat16_t*)vs)[0])), 0, start, (((int)blockIdx.x) % 48), 0);
        tl::tma_store(DV_desc, (&(((bfloat16_t*)vs)[4096])), 64, start, (((int)blockIdx.x) % 48), 0);
        tl::tma_store_arrive();
      }
      tl::tma_store_wait<0, true>();
    }
    __syncthreads();
    {
      tl::GmmaDescriptor desc_a_6;
      tl::GmmaDescriptor desc_b_6;
      tl::initialize_wgmma_descriptor<1, 1, 64>(desc_a_6, (&(((bfloat16_t*)ds)[0])));
      tl::initialize_wgmma_descriptor<1, 1, 64>(desc_b_6, (&(((bfloat16_t*)hs)[0])));
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc + 0), 32);
      tl::warpgroup_arrive();
      #pragma unroll
      for (int ki_6 = 0; ki_6 < 8; ++ki_6) {
        tl::wgmma_ss<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 64, 64, 16, false, false, 1, 1>(uint64_t(desc_a_6 + ((((ki_6 >> 2) * 8192) + ((ki_6 & 3) * 32)) >> 4)), uint64_t(desc_b_6 + (((((ki_6 >> 2) * 16384) + ((((int)threadIdx.x) >> 7) * 8192)) + ((ki_6 & 3) * 32)) >> 4)), ((uint32_t*)(acc + 0)), ((0 < ki_6) ? 1 : 0));
      }
      tl::warpgroup_commit_batch();
      tl::warpgroup_wait<0>();
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc + 0), 32);
    }
    #pragma unroll
    for (int i_15 = 0; i_15 < 16; ++i_15) {
      bfloat16_t qs_local_cast_1[2];
      *(uint1*)(qs_local_cast_1 + 0) = *(uint1*)(((bfloat16_t*)qs) + ((((((((((int)threadIdx.x) >> 5) * 1024) + ((i_15 & 1) * 512)) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + (((((((int)threadIdx.x) & 31) >> 4) + (i_15 >> 3)) & 1) * 32)) + (((((((int)threadIdx.x) & 15) >> 3) + ((i_15 & 7) >> 2)) & 1) * 16)) + (((((((int)threadIdx.x) & 7) >> 2) + ((i_15 & 3) >> 1)) & 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)));
      float2 __23;
        float2 v__17 = *(float2*)(acc + (i_15 * 2));
        float2 __24;
        uint1 v__18 = *(uint1*)(qs_local_cast_1 + 0);
        ((float2*)(&__24))[0] = __bfloat1622float2((reinterpret_cast<__nv_bfloat162*>(&v__18))[0]);
        __23.x = (v__17.x*__24.x);
        __23.y = (v__17.y*__24.y);
      *(float2*)(prod + (i_15 * 2)) = __23;
    }
    #pragma unroll
    for (int i_16 = 0; i_16 < 2; ++i_16) {
      dg_clear_1[i_16] = 0x0p+0f/*0.000000e+00*/;
      #pragma unroll
      for (int rv_2 = 0; rv_2 < 16; ++rv_2) {
        dg_clear_1[i_16] = (dg_clear_1[i_16] + prod[((((rv_2 & 7) * 4) + (i_16 * 2)) + (rv_2 >> 3))]);
      }
      __syncthreads();
      dg_clear_1[i_16] = tl::AllReduce<tl::SumOp, 256, 128, 0, tl::NamedBarrier<256>>::run(dg_clear_1[i_16], (&(((float*)workspace_1)[0])));
      dg_clear_1[i_16] = tl::AllReduce<tl::SumOp, 4, 1, 0, tl::NamedBarrier<256>>::run(dg_clear_1[i_16]);
      dg[i_16] = (dg[i_16] + dg_clear_1[i_16]);
    }
    __syncthreads();
    #pragma unroll
    for (int i_17 = 0; i_17 < 4; ++i_17) {
      tl::ptx_stmatrix_m8n8_x4((&(((bfloat16_t*)vs)[((((((int)threadIdx.x) >> 5) * 1024) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + (i_17 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_17 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511))])), __pack_half2(((bfloat16_t)acc[(i_17 * 8)]), ((bfloat16_t)acc[((i_17 * 8) + 1)])), __pack_half2(((bfloat16_t)acc[((i_17 * 8) + 2)]), ((bfloat16_t)acc[((i_17 * 8) + 3)])), __pack_half2(((bfloat16_t)acc[((i_17 * 8) + 4)]), ((bfloat16_t)acc[((i_17 * 8) + 5)])), __pack_half2(((bfloat16_t)acc[((i_17 * 8) + 6)]), ((bfloat16_t)acc[((i_17 * 8) + 7)])));
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
    for (int i_18 = 0; i_18 < 16; ++i_18) {
      float condval_4;
      if (((((((((int)threadIdx.x) >> 7) * 32) + ((i_18 >> 2) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + (i_18 & 1)) <= (((((((int)threadIdx.x) & 127) >> 5) * 16) + (((i_18 & 3) >> 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2)))) {
        condval_4 = tile[i_18];
      } else {
        condval_4 = 0x0p+0f/*0.000000e+00*/;
      }
      tile[i_18] = condval_4;
      ((bfloat16_t*)dps)[((((((((((((int)threadIdx.x) & 127) >> 5) * 1024) + (((i_18 & 3) >> 1) * 512)) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + ((((((int)threadIdx.x) >> 7) + ((((int)threadIdx.x) & 31) >> 4)) & 1) * 32)) + ((((i_18 >> 3) + ((((int)threadIdx.x) & 15) >> 3)) & 1) * 16)) + (((((i_18 & 7) >> 2) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + (i_18 & 1))] = ((bfloat16_t)tile[i_18]);
      tile[i_18] = (tile[i_18] * ((float)((bfloat16_t*)ps)[((((((((((((int)threadIdx.x) & 127) >> 5) * 1024) + (((i_18 & 3) >> 1) * 512)) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + ((((((int)threadIdx.x) >> 7) + ((((int)threadIdx.x) & 31) >> 4)) & 1) * 32)) + ((((i_18 >> 3) + ((((int)threadIdx.x) & 15) >> 3)) & 1) * 16)) + (((((i_18 & 7) >> 2) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + (i_18 & 1))]));
    }
    #pragma unroll
    for (int i_19 = 0; i_19 < 2; ++i_19) {
      dg_clear_2[i_19] = 0x0p+0f/*0.000000e+00*/;
      #pragma unroll
      for (int rv_3 = 0; rv_3 < 8; ++rv_3) {
        dg_clear_2[i_19] = (dg_clear_2[i_19] + tile[((((rv_3 & 3) * 4) + (i_19 * 2)) + (rv_3 >> 2))]);
      }
      __syncthreads();
      dg_clear_2[i_19] = tl::AllReduce<tl::SumOp, 256, 128, 0, tl::NamedBarrier<256>>::run(dg_clear_2[i_19], (&(((float*)workspace_4)[0])));
      dg_clear_2[i_19] = tl::AllReduce<tl::SumOp, 4, 1, 0, tl::NamedBarrier<256>>::run(dg_clear_2[i_19]);
      dg[i_19] = (dg[i_19] + dg_clear_2[i_19]);
    }
    __syncthreads();
    #pragma unroll
    for (int i_20 = 0; i_20 < 16; ++i_20) {
      bfloat16_t vs_local_cast_2[2];
      *(uint1*)(vs_local_cast_2 + 0) = *(uint1*)(((bfloat16_t*)vs) + ((((((((((int)threadIdx.x) >> 5) * 1024) + ((i_20 & 1) * 512)) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + (((((((int)threadIdx.x) & 31) >> 4) + (i_20 >> 3)) & 1) * 32)) + (((((((int)threadIdx.x) & 15) >> 3) + ((i_20 & 7) >> 2)) & 1) * 16)) + (((((((int)threadIdx.x) & 7) >> 2) + ((i_20 & 3) >> 1)) & 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)));
      float2 __25;
      uint1 v__19 = *(uint1*)(vs_local_cast_2 + 0);
      ((float2*)(&__25))[0] = __bfloat1622float2((reinterpret_cast<__nv_bfloat162*>(&v__19))[0]);
      *(float2*)(acc + (i_20 * 2)) = __25;
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
    __syncthreads();
    #pragma unroll
    for (int i_21 = 0; i_21 < 4; ++i_21) {
      tl::ptx_stmatrix_m8n8_x4((&(((bfloat16_t*)vs)[((((((int)threadIdx.x) >> 5) * 1024) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + (i_21 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_21 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511))])), __pack_half2(((bfloat16_t)acc[(i_21 * 8)]), ((bfloat16_t)acc[((i_21 * 8) + 1)])), __pack_half2(((bfloat16_t)acc[((i_21 * 8) + 2)]), ((bfloat16_t)acc[((i_21 * 8) + 3)])), __pack_half2(((bfloat16_t)acc[((i_21 * 8) + 4)]), ((bfloat16_t)acc[((i_21 * 8) + 5)])), __pack_half2(((bfloat16_t)acc[((i_21 * 8) + 6)]), ((bfloat16_t)acc[((i_21 * 8) + 7)])));
    }
    tl::fence_proxy_async();
    __syncthreads();
    if (((int)threadIdx.x) < 32) {
      if (tl::tl_shuffle_elect<32>()) {
        tl::tma_store(DQ_desc, (&(((bfloat16_t*)vs)[0])), 0, start, (((int)blockIdx.x) % 48), 0);
        tl::tma_store(DQ_desc, (&(((bfloat16_t*)vs)[4096])), 64, start, (((int)blockIdx.x) % 48), 0);
        tl::tma_store_arrive();
      }
      tl::tma_store_wait<0, true>();
    }
    __syncthreads();
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
    for (int i_22 = 0; i_22 < 8; ++i_22) {
      float4 __26;
        float4 v__20 = *(float4*)(dk + (i_22 * 4));
        float4 v__21 = make_float4(((float*)ex)[63], ((float*)ex)[63], ((float*)ex)[63], ((float*)ex)[63]);
        __26.x = (v__20.x*v__21.x);
        __26.y = (v__20.y*v__21.y);
        __26.z = (v__20.z*v__21.z);
        __26.w = (v__20.w*v__21.w);
      *(float4*)(dk + (i_22 * 4)) = __26;
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
      tl::initialize_wgmma_descriptor<1, 1, 64>(desc_b_11, (&(((bfloat16_t*)hs)[0])));
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc + 0), 32);
      tl::warpgroup_arrive();
      #pragma unroll
      for (int ki_11 = 0; ki_11 < 8; ++ki_11) {
        tl::wgmma_ss<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 64, 64, 16, false, false, 1, 1>(uint64_t(desc_a_11 + ((((ki_11 >> 2) * 8192) + ((ki_11 & 3) * 32)) >> 4)), uint64_t(desc_b_11 + (((((ki_11 >> 2) * 16384) + ((((int)threadIdx.x) >> 7) * 8192)) + ((ki_11 & 3) * 32)) >> 4)), ((uint32_t*)(acc + 0)), ((0 < ki_11) ? 1 : 0));
      }
      tl::warpgroup_commit_batch();
      tl::warpgroup_wait<0>();
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(acc + 0), 32);
    }
    #pragma unroll
    for (int i_23 = 0; i_23 < 32; ++i_23) {
      dk[i_23] = (dk[i_23] - acc[i_23]);
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
    for (int i_24 = 0; i_24 < 16; ++i_24) {
      float condval_5;
      if (((((((((int)threadIdx.x) >> 7) * 32) + ((i_24 >> 2) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + (i_24 & 1)) < (((((((int)threadIdx.x) & 127) >> 5) * 16) + (((i_24 & 3) >> 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2)))) {
        condval_5 = (tile[i_24] * -0x1p+0f/*-1.000000e+00*/);
      } else {
        condval_5 = 0x0p+0f/*0.000000e+00*/;
      }
      ((bfloat16_t*)a)[((((((((((((int)threadIdx.x) & 127) >> 5) * 1024) + (((i_24 & 3) >> 1) * 512)) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + ((((((int)threadIdx.x) >> 7) + ((((int)threadIdx.x) & 31) >> 4)) & 1) * 32)) + ((((i_24 >> 3) + ((((int)threadIdx.x) & 15) >> 3)) & 1) * 16)) + (((((i_24 & 7) >> 2) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + (i_24 & 1))] = ((bfloat16_t)condval_5);
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
    for (int i_25 = 0; i_25 < 16; ++i_25) {
      float2 __27;
        float2 v__22 = *(float2*)(acc + (i_25 * 2));
        float2 __28;
        uint1 v__23 = *(uint1*)(((bfloat16_t*)ks) + ((((((((((int)threadIdx.x) >> 5) * 1024) + ((i_25 & 1) * 512)) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + (((((((int)threadIdx.x) & 31) >> 4) + (i_25 >> 3)) & 1) * 32)) + (((((((int)threadIdx.x) & 15) >> 3) + ((i_25 & 7) >> 2)) & 1) * 16)) + (((((((int)threadIdx.x) & 7) >> 2) + ((i_25 & 3) >> 1)) & 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)));
        ((float2*)(&__28))[0] = __bfloat1622float2((reinterpret_cast<__nv_bfloat162*>(&v__23))[0]);
        __27.x = (v__22.x*__28.x);
        __27.y = (v__22.y*__28.y);
      *(float2*)(prod + (i_25 * 2)) = __27;
      float2 __29;
        float2 v__24 = *(float2*)(dk + (i_25 * 2));
        float2 __30;
          float2 v__25 = make_float2(((float*)beta)[(((((((int)threadIdx.x) & 127) >> 5) * 16) + ((i_25 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2))], ((float*)beta)[(((((((int)threadIdx.x) & 127) >> 5) * 16) + ((i_25 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2))]);
          float2 v__26 = *(float2*)(acc + (i_25 * 2));
          __30.x = (v__25.x*v__26.x);
          __30.y = (v__25.y*v__26.y);
        __29.x = (v__24.x+__30.x);
        __29.y = (v__24.y+__30.y);
      *(float2*)(dk + (i_25 * 2)) = __29;
      uint1 __31;
      float2 __32;
        float2 v__27 = make_float2(((float*)beta)[(((((((int)threadIdx.x) & 127) >> 5) * 16) + ((i_25 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2))], ((float*)beta)[(((((((int)threadIdx.x) & 127) >> 5) * 16) + ((i_25 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2))]);
        float2 __33;
        uint1 v__28 = *(uint1*)(((bfloat16_t*)ks) + ((((((((((int)threadIdx.x) >> 5) * 1024) + ((i_25 & 1) * 512)) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + (((((((int)threadIdx.x) & 31) >> 4) + (i_25 >> 3)) & 1) * 32)) + (((((((int)threadIdx.x) & 15) >> 3) + ((i_25 & 7) >> 2)) & 1) * 16)) + (((((((int)threadIdx.x) & 7) >> 2) + ((i_25 & 3) >> 1)) & 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)));
        ((float2*)(&__33))[0] = __bfloat1622float2((reinterpret_cast<__nv_bfloat162*>(&v__28))[0]);
        __32.x = (v__27.x*__33.x);
        __32.y = (v__27.y*__33.y);
      (reinterpret_cast<__nv_bfloat162*>(&__31))[0] = __float22bfloat162_rn(((float2*)(&__32))[0]);
      *(uint1*)(((bfloat16_t*)vs) + ((((((((((int)threadIdx.x) >> 5) * 1024) + ((i_25 & 1) * 512)) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + (((((((int)threadIdx.x) & 31) >> 4) + (i_25 >> 3)) & 1) * 32)) + (((((((int)threadIdx.x) & 15) >> 3) + ((i_25 & 7) >> 2)) & 1) * 16)) + (((((((int)threadIdx.x) & 7) >> 2) + ((i_25 & 3) >> 1)) & 1) * 8)) + ((((int)threadIdx.x) & 3) * 2))) = __31;
    }
    #pragma unroll
    for (int i_26 = 0; i_26 < 2; ++i_26) {
      db_clear[i_26] = 0x0p+0f/*0.000000e+00*/;
      #pragma unroll
      for (int rv_4 = 0; rv_4 < 16; ++rv_4) {
        db_clear[i_26] = (db_clear[i_26] + prod[((((rv_4 & 7) * 4) + (i_26 * 2)) + (rv_4 >> 3))]);
      }
      __syncthreads();
      db_clear[i_26] = tl::AllReduce<tl::SumOp, 256, 128, 0, tl::NamedBarrier<256>>::run(db_clear[i_26], (&(((float*)workspace)[0])));
      db_clear[i_26] = tl::AllReduce<tl::SumOp, 4, 1, 0, tl::NamedBarrier<256>>::run(db_clear[i_26]);
      db[i_26] = (db[i_26] + db_clear[i_26]);
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
    __syncthreads();
    #pragma unroll
    for (int i_27 = 0; i_27 < 4; ++i_27) {
      tl::ptx_stmatrix_m8n8_x4((&(((bfloat16_t*)vs)[((((((int)threadIdx.x) >> 5) * 1024) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + (i_27 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_27 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511))])), __pack_half2(((bfloat16_t)dk[(i_27 * 8)]), ((bfloat16_t)dk[((i_27 * 8) + 1)])), __pack_half2(((bfloat16_t)dk[((i_27 * 8) + 2)]), ((bfloat16_t)dk[((i_27 * 8) + 3)])), __pack_half2(((bfloat16_t)dk[((i_27 * 8) + 4)]), ((bfloat16_t)dk[((i_27 * 8) + 5)])), __pack_half2(((bfloat16_t)dk[((i_27 * 8) + 6)]), ((bfloat16_t)dk[((i_27 * 8) + 7)])));
    }
    tl::fence_proxy_async();
    __syncthreads();
    if (((int)threadIdx.x) < 32) {
      if (tl::tl_shuffle_elect<32>()) {
        tl::tma_store(DK_desc, (&(((bfloat16_t*)vs)[0])), 0, start, (((int)blockIdx.x) % 48), 0);
        tl::tma_store(DK_desc, (&(((bfloat16_t*)vs)[4096])), 64, start, (((int)blockIdx.x) % 48), 0);
        tl::tma_store_arrive();
      }
      tl::tma_store_wait<0, true>();
    }
    __syncthreads();
    if ((((((int)threadIdx.x) & 3) * 2) + (((int)threadIdx.x) >> 7)) == 0) {
      #pragma unroll
      for (int i_28 = 0; i_28 < 2; ++i_28) {
        if (0 <= ((((((((int)threadIdx.x) & 127) >> 5) * 16) + (i_28 * 8)) + ((((int)threadIdx.x) & 31) >> 2)) + start)) {
          if (((((((((int)threadIdx.x) & 127) >> 5) * 16) + (i_28 * 8)) + ((((int)threadIdx.x) & 31) >> 2)) + start) < 131072) {
            DG[(((((((((int64_t)((int)threadIdx.x)) & (int64_t)127) >> (int64_t)5) * (int64_t)768) + (((int64_t)i_28) * (int64_t)384)) + (((((int64_t)((int)threadIdx.x)) & (int64_t)31) >> (int64_t)2) * (int64_t)48)) + (((int64_t)start) * (int64_t)48)) + (((int64_t)((int)blockIdx.x)) % (int64_t)48))] = dg[i_28];
            DB[(((((((((int64_t)((int)threadIdx.x)) & (int64_t)127) >> (int64_t)5) * (int64_t)768) + (((int64_t)i_28) * (int64_t)384)) + (((((int64_t)((int)threadIdx.x)) & (int64_t)31) >> (int64_t)2) * (int64_t)48)) + (((int64_t)start) * (int64_t)48)) + (((int64_t)((int)blockIdx.x)) % (int64_t)48))] = db[i_28];
          }
        }
      }
    }
    #pragma unroll
    for (int i_29 = 0; i_29 < 16; ++i_29) {
      float4 __34;
        float4 v__29 = *(float4*)(dh + (i_29 * 4));
        float4 v__30 = make_float4(((float*)ex)[63], ((float*)ex)[63], ((float*)ex)[63], ((float*)ex)[63]);
        __34.x = (v__29.x*v__30.x);
        __34.y = (v__29.y*v__30.y);
        __34.z = (v__29.z*v__30.z);
        __34.w = (v__29.w*v__30.w);
      *(float4*)(dh + (i_29 * 4)) = __34;
    }
    {
      tl::GmmaDescriptor desc_a_15;
      tl::GmmaDescriptor desc_b_15;
      tl::initialize_wgmma_descriptor<1, 512, 64>(desc_a_15, (&(((bfloat16_t*)ds)[0])));
      tl::initialize_wgmma_descriptor<1, 512, 64>(desc_b_15, (&(((bfloat16_t*)qs)[0])));
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(dh + 0), 64);
      tl::warpgroup_arrive();
      #pragma unroll
      for (int i_30 = 0; i_30 < 2; ++i_30) {
        #pragma unroll
        for (int ki_15 = 0; ki_15 < 4; ++ki_15) {
          tl::wgmma_ss<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 64, 64, 16, true, true, 1, 1>(uint64_t(desc_a_15 + (((i_30 * 8192) + (ki_15 * 2048)) >> 4)), uint64_t(desc_b_15 + ((((((int)threadIdx.x) >> 7) * 8192) + (ki_15 * 2048)) >> 4)), ((uint32_t*)(dh + (i_30 * 32))), 1);
        }
      }
      tl::warpgroup_commit_batch();
      tl::warpgroup_wait<0>();
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(dh + 0), 64);
    }
    #pragma unroll
    for (int i_31 = 0; i_31 < 4; ++i_31) {
      __syncthreads();
      bfloat16_t broadcast_var_2 = bfloat16_t(-0x1p+0f/*-1.000000e+00*/);
      uint4 __35;
        uint4 v__31 = *(uint4*)(((bfloat16_t*)ws) + ((((((((((int)threadIdx.x) & 15) >> 3) * 4096) + (i_31 * 1024)) + ((((int)threadIdx.x) >> 4) * 64)) + (((((((int)threadIdx.x) & 127) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)));
        uint4 v__32 = make_uint4(__pack_nv_bfloat162(broadcast_var_2, broadcast_var_2), __pack_nv_bfloat162(broadcast_var_2, broadcast_var_2), __pack_nv_bfloat162(broadcast_var_2, broadcast_var_2), __pack_nv_bfloat162(broadcast_var_2, broadcast_var_2));
        *(uint1*)(&(__35.x)) = tl::to_uint1(tl::mul2(tl::from_uint1<__nv_bfloat162>(*(uint1*)(&(v__31.x))), tl::from_uint1<__nv_bfloat162>(*(uint1*)(&(v__32.x)))));
        *(uint1*)(&(__35.y)) = tl::to_uint1(tl::mul2(tl::from_uint1<__nv_bfloat162>(*(uint1*)(&(v__31.y))), tl::from_uint1<__nv_bfloat162>(*(uint1*)(&(v__32.y)))));
        *(uint1*)(&(__35.z)) = tl::to_uint1(tl::mul2(tl::from_uint1<__nv_bfloat162>(*(uint1*)(&(v__31.z))), tl::from_uint1<__nv_bfloat162>(*(uint1*)(&(v__32.z)))));
        *(uint1*)(&(__35.w)) = tl::to_uint1(tl::mul2(tl::from_uint1<__nv_bfloat162>(*(uint1*)(&(v__31.w))), tl::from_uint1<__nv_bfloat162>(*(uint1*)(&(v__32.w)))));
      *(uint4*)(((bfloat16_t*)ws) + ((((((((((int)threadIdx.x) & 15) >> 3) * 4096) + (i_31 * 1024)) + ((((int)threadIdx.x) >> 4) * 64)) + (((((((int)threadIdx.x) & 127) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))) = __35;
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
      for (int i_32 = 0; i_32 < 2; ++i_32) {
        #pragma unroll
        for (int ki_16 = 0; ki_16 < 4; ++ki_16) {
          tl::wgmma_ss<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 64, 64, 16, true, true, 1, 1>(uint64_t(desc_a_16 + (((i_32 * 8192) + (ki_16 * 2048)) >> 4)), uint64_t(desc_b_16 + ((((((int)threadIdx.x) >> 7) * 8192) + (ki_16 * 2048)) >> 4)), ((uint32_t*)(dh + (i_32 * 32))), 1);
        }
      }
      tl::warpgroup_commit_batch();
      tl::warpgroup_wait<0>();
      tl::warpgroup_fence_operand(reinterpret_cast<float*>(dh + 0), 64);
    }
  }
  if (((int)threadIdx.x) < 32) {
    tl::tma_store_wait<0, false>();
  }
  #pragma unroll
  for (int i_33 = 0; i_33 < 64; ++i_33) {
    DH0[(((((((((((int)blockIdx.x) * 16384) + ((((int)threadIdx.x) >> 7) * 8192)) + (((i_33 & 31) >> 2) * 1024)) + ((((int)threadIdx.x) & 3) * 256)) + ((i_33 & 1) * 128)) + ((i_33 >> 5) * 64)) + (((((int)threadIdx.x) & 127) >> 5) * 16)) + (((i_33 & 3) >> 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2))] = dh[i_33];
  }
}


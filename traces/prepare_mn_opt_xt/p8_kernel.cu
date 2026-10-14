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

extern "C" __global__ void kernel_kernel(__grid_constant__ const CUtensorMap A_desc, const float* __restrict__ Beta, const float* __restrict__ G, __grid_constant__ const CUtensorMap K_desc, float* __restrict__ Mout, float* __restrict__ Nout, __grid_constant__ const CUtensorMap V_desc);
extern "C" __global__ void __launch_bounds__(384, 1) kernel_kernel(__grid_constant__ const CUtensorMap A_desc, const float* __restrict__ Beta, const float* __restrict__ G, __grid_constant__ const CUtensorMap K_desc, float* __restrict__ Mout, float* __restrict__ Nout, __grid_constant__ const CUtensorMap V_desc) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  void* ks = ((void*)((char*)buf_dyn_shmem + 0));
  void* vs = ((void*)((char*)buf_dyn_shmem + 32768));
  void* a = ((void*)((char*)buf_dyn_shmem + 65536));
  void* x = ((void*)((char*)buf_dyn_shmem + 81920));
  void* beta = ((void*)((char*)buf_dyn_shmem + 98304));
  void* decay = ((void*)((char*)buf_dyn_shmem + 98816));
  void* g = ((void*)((char*)buf_dyn_shmem + 99328));
  void* gammas = ((void*)((char*)buf_dyn_shmem + 99840));
  __shared__ __align__(16) uint64_t ready_mem[2];
  auto ready = reinterpret_cast<Barrier*>(ready_mem);
  __shared__ __align__(16) uint64_t free_mem[2];
  auto free = reinterpret_cast<Barrier*>(free_mem);
  int iters = 0;
  float nt[64];
  float mt[64];
  float gamma = 0x0p+0f/*0.000000e+00*/;
  float xf[32];
  bfloat16_t operand[64];
  float yz[32];
  bfloat16_t yz_operand[32];
  if (tl::tl_shuffle_elect<0>()) {
    tl::prefetch_tma_descriptor(K_desc);
    tl::prefetch_tma_descriptor(V_desc);
    tl::prefetch_tma_descriptor(A_desc);
  }
  if (tl::tl_shuffle_elect<0>()) {
    ready[0].init(128);
    ready[1].init(128);
    free[0].init(256);
    free[1].init(256);
  }
  tl::fence_barrier_init();
  __syncthreads();
  iters = 32;
  if (((int)threadIdx.x) < 256) {
    tl::warpgroup_reg_alloc<240>();
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
    for (int c = 0; c < iters; ++c) {
      ready[(c & 1)].wait(((c & 3) >> 1));
      gamma = ((float*)gammas)[(c & 1)];
      {
        tl::GmmaDescriptor desc_a;
        tl::GmmaDescriptor desc_b;
        tl::initialize_wgmma_descriptor<1, 512, 64>(desc_a, (&(((bfloat16_t*)ks)[0])));
        tl::increase_descriptor_offset<int>(desc_a, ((c & 1) * 16384));
        tl::initialize_wgmma_descriptor<1, 0, 64>(desc_b, (&(((bfloat16_t*)a)[0])));
        tl::increase_descriptor_offset<int>(desc_b, ((c & 1) * 8192));
        tl::warpgroup_fence_operand(reinterpret_cast<float*>(xf + 0), 64);
        tl::warpgroup_arrive();
        #pragma unroll
        for (int ki = 0; ki < 4; ++ki) {
          tl::wgmma_ss<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 64, 64, 16, true, true, 1, 1>(uint64_t(desc_a + ((((((int)threadIdx.x) >> 7) * 8192) + (ki * 2048)) >> 4)), uint64_t(desc_b + ((ki * 2048) >> 4)), ((uint32_t*)(xf + 0)), ((0 < ki) ? 1 : 0));
        }
        tl::warpgroup_commit_batch();
        tl::warpgroup_wait<0>();
        tl::warpgroup_fence_operand(reinterpret_cast<float*>(xf + 0), 64);
      }
      tl::__sync_thread_partial(3, 256);
      #pragma unroll
      for (int i_2 = 0; i_2 < 32; ++i_2) {
        ((bfloat16_t*)x)[(((((((((((int)threadIdx.x) >> 7) * 4096) + ((i_2 >> 2) * 512)) + ((((int)threadIdx.x) & 3) * 128)) + (((i_2 & 3) >> 1) * 64)) + (((((((int)threadIdx.x) & 127) >> 6) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + (((int)threadIdx.x) & 1)) & 1) * 16)) + (((((i_2 & 3) >> 1) + (i_2 & 1)) & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2))] = ((bfloat16_t)((((float*)beta)[(((((c & 1) * 64) + ((i_2 >> 2) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + ((i_2 & 3) >> 1))] * -0x1p+0f/*-1.000000e+00*/) * xf[((((i_2 >> 2) * 4) + ((i_2 & 1) * 2)) + ((i_2 & 3) >> 1))]));
      }
      #pragma unroll
      for (int i_3 = 0; i_3 < 16; ++i_3) {
        uint2 __1;
        float4 v_ = *(float4*)(mt + (i_3 * 4));
        (reinterpret_cast<__nv_bfloat162*>(&__1))[0] = __float22bfloat162_rn(((float2*)(&v_))[0]);
        (reinterpret_cast<__nv_bfloat162*>(&__1))[1] = __float22bfloat162_rn(((float2*)(&v_))[1]);
        *(uint2*)(operand + (i_3 * 4)) = __1;
      }
      {
        tl::GmmaDescriptor desc_b_1;
        tl::__sync_thread_partial(3, 256);
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
      for (int i_4 = 0; i_4 < 8; ++i_4) {
        uint2 __2;
        float4 v__1 = *(float4*)(yz + (i_4 * 4));
        (reinterpret_cast<__nv_bfloat162*>(&__2))[0] = __float22bfloat162_rn(((float2*)(&v__1))[0]);
        (reinterpret_cast<__nv_bfloat162*>(&__2))[1] = __float22bfloat162_rn(((float2*)(&v__1))[1]);
        *(uint2*)(yz_operand + (i_4 * 4)) = __2;
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
      for (int i_5 = 0; i_5 < 64; ++i_5) {
        mt[i_5] = (mt[i_5] * gamma);
      }
      #pragma unroll
      for (int i_6 = 0; i_6 < 16; ++i_6) {
        uint2 __3;
        float4 v__2 = *(float4*)(nt + (i_6 * 4));
        (reinterpret_cast<__nv_bfloat162*>(&__3))[0] = __float22bfloat162_rn(((float2*)(&v__2))[0]);
        (reinterpret_cast<__nv_bfloat162*>(&__3))[1] = __float22bfloat162_rn(((float2*)(&v__2))[1]);
        *(uint2*)(operand + (i_6 * 4)) = __3;
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
      #pragma unroll
      for (int i_7 = 0; i_7 < 32; ++i_7) {
        yz_operand[i_7] = ((bfloat16_t)((gamma * yz[i_7]) - (((float)((bfloat16_t*)vs)[((((((((c & 1) * 8192) + ((i_7 >> 2) * 1024)) + ((((int)threadIdx.x) & 3) * 256)) + ((i_7 & 1) * 128)) + ((((int)threadIdx.x) >> 5) * 16)) + (((i_7 & 3) >> 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2))]) * ((float*)decay)[(((((c & 1) * 64) + ((i_7 >> 2) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + (i_7 & 1))])));
      }
      #pragma unroll
      for (int i_8 = 0; i_8 < 64; ++i_8) {
        nt[i_8] = (nt[i_8] * gamma);
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
      free[(c & 1)].arrive();
    }
    #pragma unroll
    for (int i_9 = 0; i_9 < 64; ++i_9) {
      Nout[((((((((((int)blockIdx.x) * 786432) + (((int)blockIdx.y) * 16384)) + ((i_9 >> 2) * 1024)) + ((((int)threadIdx.x) & 3) * 256)) + (((i_9 & 3) >> 1) * 128)) + ((((int)threadIdx.x) >> 5) * 16)) + ((i_9 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2))] = nt[((((i_9 >> 2) * 4) + ((i_9 & 1) * 2)) + ((i_9 & 3) >> 1))];
    }
    #pragma unroll
    for (int i_10 = 0; i_10 < 64; ++i_10) {
      Mout[((((((((((int)blockIdx.x) * 786432) + (((int)blockIdx.y) * 16384)) + ((i_10 >> 2) * 1024)) + ((((int)threadIdx.x) & 3) * 256)) + (((i_10 & 3) >> 1) * 128)) + ((((int)threadIdx.x) >> 5) * 16)) + ((i_10 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2))] = mt[((((i_10 >> 2) * 4) + ((i_10 & 1) * 2)) + ((i_10 & 3) >> 1))];
    }
  } else {
    tl::warpgroup_reg_dealloc<24>();
    for (int c_1 = 0; c_1 < iters; ++c_1) {
      free[(c_1 & 1)].wait((((c_1 >> 1) + 1) & 1));
      tl::__sync_thread_partial(4, 128);
      if (((int)threadIdx.x) < 288) {
        if (tl::tl_shuffle_elect<32>()) {
          ready[(c_1 & 1)].expect_transaction(16384);
          tl::tma_load(K_desc, ready[(c_1 & 1)], (&(((bfloat16_t*)ks)[((c_1 & 1) * 8192)])), 0, ((((int)blockIdx.x) * 2048) + (c_1 * 64)), (((int)blockIdx.y) / 3), 0);
          tl::tma_load(K_desc, ready[(c_1 & 1)], (&(((bfloat16_t*)ks)[(((c_1 & 1) * 8192) + 4096)])), 64, ((((int)blockIdx.x) * 2048) + (c_1 * 64)), (((int)blockIdx.y) / 3), 0);
        }
      } else {
        if (((int)threadIdx.x) < 320) {
          if (tl::tl_shuffle_elect<32>()) {
            ready[(c_1 & 1)].expect_transaction(16384);
            tl::tma_load(V_desc, ready[(c_1 & 1)], (&(((bfloat16_t*)vs)[((c_1 & 1) * 8192)])), 0, ((((int)blockIdx.x) * 2048) + (c_1 * 64)), ((int)blockIdx.y), 0);
            ready[(c_1 & 1)].expect_transaction(8192);
            tl::tma_load(A_desc, ready[(c_1 & 1)], (&(((bfloat16_t*)a)[((c_1 & 1) * 4096)])), 0, ((((int)blockIdx.x) * 2048) + (c_1 * 64)), ((int)blockIdx.y), 0);
          }
        } else {
          float condval_1;
          if ((((((int)blockIdx.x) * 32) + c_1) < 256)) {
            condval_1 = G[(((((((int64_t)((int)blockIdx.x)) * (int64_t)98304) + (((int64_t)c_1) * (int64_t)3072)) + (((int64_t)((int)threadIdx.x)) * (int64_t)48)) + ((int64_t)((int)blockIdx.y))) - (int64_t)15360)];
          } else {
            condval_1 = 0x0p+0f/*0.000000e+00*/;
          }
          ((float*)g)[((((c_1 & 1) * 64) + ((int)threadIdx.x)) - 320)] = condval_1;
          float condval_2;
          if ((((((int)blockIdx.x) * 32) + c_1) < 256)) {
            condval_2 = Beta[(((((((int64_t)((int)blockIdx.x)) * (int64_t)98304) + (((int64_t)c_1) * (int64_t)3072)) + (((int64_t)((int)threadIdx.x)) * (int64_t)48)) + ((int64_t)((int)blockIdx.y))) - (int64_t)15360)];
          } else {
            condval_2 = 0x0p+0f/*0.000000e+00*/;
          }
          ((float*)beta)[((((c_1 & 1) * 64) + ((int)threadIdx.x)) - 320)] = condval_2;
          tl::__sync_thread_partial(7, 64);
          ((float*)decay)[((((c_1 & 1) * 64) + ((int)threadIdx.x)) - 320)] = exp2f(((((float*)g)[(((c_1 & 1) * 64) + 63)] - ((float*)g)[((((c_1 & 1) * 64) + ((int)threadIdx.x)) - 320)]) * 0x1.71547652b82fep+0f/*1.442695e+00*/));
          if (((int)threadIdx.x) == 320) {
            ((float*)gammas)[(c_1 & 1)] = exp2f((((float*)g)[(((c_1 & 1) * 64) + 63)] * 0x1.71547652b82fep+0f/*1.442695e+00*/));
          }
          tl::fence_proxy_async();
        }
      }
      ready[(c_1 & 1)].arrive();
    }
  }
}


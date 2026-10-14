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

extern "C" __global__ void kernel_kernel(__grid_constant__ const CUtensorMap A_desc, const float* __restrict__ Beta, const float* __restrict__ G, __grid_constant__ const CUtensorMap K_desc, bfloat16_t* __restrict__ Mout, bfloat16_t* __restrict__ Nout, __grid_constant__ const CUtensorMap V_desc);
extern "C" __global__ void __launch_bounds__(512, 1) kernel_kernel(__grid_constant__ const CUtensorMap A_desc, const float* __restrict__ Beta, const float* __restrict__ G, __grid_constant__ const CUtensorMap K_desc, bfloat16_t* __restrict__ Mout, bfloat16_t* __restrict__ Nout, __grid_constant__ const CUtensorMap V_desc) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  void* ks = ((void*)((char*)buf_dyn_shmem + 0));
  void* vs = ((void*)((char*)buf_dyn_shmem + 32768));
  void* a = ((void*)((char*)buf_dyn_shmem + 65536));
  void* beta = ((void*)((char*)buf_dyn_shmem + 81920));
  void* g = ((void*)((char*)buf_dyn_shmem + 82432));
  void* x = ((void*)((char*)buf_dyn_shmem + 82944));
  __shared__ __align__(16) uint64_t mbarrier_mem[12];
  auto mbarrier = reinterpret_cast<Barrier*>(mbarrier_mem);
  float nt[64];
  float mt[64];
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
    mbarrier[0].init(1);
    mbarrier[1].init(1);
    mbarrier[2].init(1);
    mbarrier[3].init(1);
    mbarrier[4].init(1);
    mbarrier[5].init(1);
    mbarrier[6].init(256);
    mbarrier[7].init(256);
    mbarrier[8].init(256);
    mbarrier[9].init(256);
    mbarrier[10].init(256);
    mbarrier[11].init(256);
  }
  tl::fence_barrier_init();
  __syncthreads();
  if (((int)threadIdx.x) < 256) {
    tl::__sync_thread_partial(3, 256);
    int condval;
    if ((((int)blockIdx.x) == 7)) {
      condval = 0;
    } else {
      condval = 32;
    }
    for (int c = 0; c < condval; ++c) {
      mbarrier[((c & 1) + 6)].wait((((c & 3) >> 1) ^ 1));
      if (((int)threadIdx.x) < 64) {
        float condval_1;
        if ((((c >> 5) + ((int)blockIdx.x)) < 8)) {
          condval_1 = G[((((((int64_t)((int)blockIdx.x)) * (int64_t)98304) + (((int64_t)c) * (int64_t)3072)) + (((int64_t)((int)threadIdx.x)) * (int64_t)48)) + ((int64_t)((int)blockIdx.y)))];
        } else {
          condval_1 = 0x0p+0f/*0.000000e+00*/;
        }
        ((float*)g)[(((c & 1) * 64) + ((int)threadIdx.x))] = condval_1;
        float condval_2;
        if ((((c >> 5) + ((int)blockIdx.x)) < 8)) {
          condval_2 = Beta[((((((int64_t)((int)blockIdx.x)) * (int64_t)98304) + (((int64_t)c) * (int64_t)3072)) + (((int64_t)((int)threadIdx.x)) * (int64_t)48)) + ((int64_t)((int)blockIdx.y)))];
        } else {
          condval_2 = 0x0p+0f/*0.000000e+00*/;
        }
        ((float*)beta)[(((c & 1) * 64) + ((int)threadIdx.x))] = condval_2;
      }
      tl::__sync_thread_partial(3, 256);
      if (tl::tl_shuffle_elect<256>()) {
        mbarrier[(c & 1)].arrive_and_expect_tx(16384);
        tl::fence_proxy_async();
        tl::tma_load(K_desc, mbarrier[(c & 1)], (&(((bfloat16_t*)ks)[((c & 1) * 8192)])), 0, ((((int)blockIdx.x) * 2048) + (c * 64)), (((int)blockIdx.y) / 3), 0);
        tl::tma_load(K_desc, mbarrier[(c & 1)], (&(((bfloat16_t*)ks)[(((c & 1) * 8192) + 4096)])), 64, ((((int)blockIdx.x) * 2048) + (c * 64)), (((int)blockIdx.y) / 3), 0);
      }
      mbarrier[((c & 1) + 8)].wait((((c & 3) >> 1) ^ 1));
      if (tl::tl_shuffle_elect<256>()) {
        mbarrier[((c & 1) + 2)].arrive_and_expect_tx(16384);
        tl::fence_proxy_async();
        tl::tma_load(V_desc, mbarrier[((c & 1) + 2)], (&(((bfloat16_t*)vs)[((c & 1) * 8192)])), 0, ((((int)blockIdx.x) * 2048) + (c * 64)), ((int)blockIdx.y), 0);
      }
      mbarrier[((c & 1) + 10)].wait((((c & 3) >> 1) ^ 1));
      if (tl::tl_shuffle_elect<256>()) {
        mbarrier[((c & 1) + 4)].arrive_and_expect_tx(8192);
        tl::fence_proxy_async();
        tl::tma_load(A_desc, mbarrier[((c & 1) + 4)], (&(((bfloat16_t*)a)[((c & 1) * 4096)])), 0, ((((int)blockIdx.x) * 2048) + (c * 64)), ((int)blockIdx.y), 0);
      }
    }
  } else {
    #pragma unroll
    for (int i = 0; i < 16; ++i) {
      float broadcast_var = 0x0p+0f/*0.000000e+00*/;
      *(float4*)(nt + (i * 4)) = make_float4(broadcast_var, broadcast_var, broadcast_var, broadcast_var);
    }
    #pragma unroll
    for (int i_1 = 0; i_1 < 64; ++i_1) {
      float condval_3;
      if (((((((((int)threadIdx.x) >> 5) * 16) + (((i_1 & 3) >> 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2)) - 128) == ((((i_1 >> 2) * 8) + ((((int)threadIdx.x) & 3) * 2)) + (i_1 & 1)))) {
        condval_3 = 0x1p+0f/*1.000000e+00*/;
      } else {
        condval_3 = 0x0p+0f/*0.000000e+00*/;
      }
      mt[i_1] = condval_3;
    }
    int condval_4;
    if ((((int)blockIdx.x) == 7)) {
      condval_4 = 0;
    } else {
      condval_4 = 32;
    }
    for (int c_1 = 0; c_1 < condval_4; ++c_1) {
      mbarrier[(c_1 & 1)].wait(((c_1 & 3) >> 1));
      mbarrier[((c_1 & 1) + 4)].wait(((c_1 & 3) >> 1));
      {
        tl::GmmaDescriptor desc_a;
        tl::GmmaDescriptor desc_b;
        tl::initialize_wgmma_descriptor<1, 0, 64>(desc_a, (&(((bfloat16_t*)a)[0])));
        tl::increase_descriptor_offset<int>(desc_a, ((c_1 & 1) * 8192));
        tl::initialize_wgmma_descriptor<1, 512, 64>(desc_b, (&(((bfloat16_t*)ks)[0])));
        tl::increase_descriptor_offset<int>(desc_b, ((c_1 & 1) * 16384));
        tl::warpgroup_fence_operand(reinterpret_cast<float*>(xf + 0), 32);
        tl::warpgroup_arrive();
        #pragma unroll
        for (int ki = 0; ki < 4; ++ki) {
          tl::wgmma_ss<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 64, 64, 16, true, true, 1, 1>(uint64_t(desc_a + ((ki * 2048) >> 4)), uint64_t(desc_b + (((((((int)threadIdx.x) & 255) >> 7) * 8192) + (ki * 2048)) >> 4)), ((uint32_t*)(xf + 0)), ((0 < ki) ? 1 : 0));
        }
        tl::warpgroup_commit_batch();
        tl::warpgroup_wait<0>();
        tl::warpgroup_fence_operand(reinterpret_cast<float*>(xf + 0), 32);
      }
      mbarrier[((c_1 & 1) + 10)].arrive();
      tl::__sync_thread_partial(4, 256);
      #pragma unroll
      for (int i_2 = 0; i_2 < 16; ++i_2) {
        float beta_local_cast_1[2];
        bfloat16_t x_local_cast[2];
        *(float2*)(beta_local_cast_1 + 0) = make_float2(((float*)beta)[(((((c_1 & 1) * 64) + (((((int)threadIdx.x) & 127) >> 5) * 16)) + ((i_2 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2))], ((float*)beta)[(((((c_1 & 1) * 64) + (((((int)threadIdx.x) & 127) >> 5) * 16)) + ((i_2 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2))]);
        float broadcast_var_1 = -0x1p+0f/*-1.000000e+00*/;
        uint1 __1;
        float2 __2;
          float2 __3;
            float2 v_ = *(float2*)(beta_local_cast_1 + 0);
            float2 v__1 = make_float2(broadcast_var_1, broadcast_var_1);
            __3.x = (v_.x*v__1.x);
            __3.y = (v_.y*v__1.y);
          float2 v__2 = *(float2*)(xf + (i_2 * 2));
          __2.x = (__3.x*v__2.x);
          __2.y = (__3.y*v__2.y);
        (reinterpret_cast<__nv_bfloat162*>(&__1))[0] = __float22bfloat162_rn(((float2*)(&__2))[0]);
        *(uint1*)(x_local_cast + 0) = __1;
        *(uint1*)(((bfloat16_t*)x) + (((((((((((int)threadIdx.x) >> 5) * 1024) + ((i_2 & 1) * 512)) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + (((((((int)threadIdx.x) & 31) >> 4) + (i_2 >> 3)) & 1) * 32)) + (((((((int)threadIdx.x) & 15) >> 3) + ((i_2 & 7) >> 2)) & 1) * 16)) + (((((((int)threadIdx.x) & 7) >> 2) + ((i_2 & 3) >> 1)) & 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)) - 8192)) = *(uint1*)(x_local_cast + 0);
      }
      #pragma unroll
      for (int i_3 = 0; i_3 < 16; ++i_3) {
        uint2 __4;
        float4 v__3 = *(float4*)(mt + (i_3 * 4));
        (reinterpret_cast<__nv_bfloat162*>(&__4))[0] = __float22bfloat162_rn(((float2*)(&v__3))[0]);
        (reinterpret_cast<__nv_bfloat162*>(&__4))[1] = __float22bfloat162_rn(((float2*)(&v__3))[1]);
        *(uint2*)(operand + (i_3 * 4)) = __4;
      }
      {
        tl::GmmaDescriptor desc_b_1;
        tl::__sync_thread_partial(4, 256);
        tl::initialize_wgmma_descriptor<1, 1, 64>(desc_b_1, (&(((bfloat16_t*)ks)[0])));
        tl::increase_descriptor_offset<int>(desc_b_1, ((c_1 & 1) * 16384));
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
        uint2 __5;
        float4 v__4 = *(float4*)(yz + (i_4 * 4));
        (reinterpret_cast<__nv_bfloat162*>(&__5))[0] = __float22bfloat162_rn(((float2*)(&v__4))[0]);
        (reinterpret_cast<__nv_bfloat162*>(&__5))[1] = __float22bfloat162_rn(((float2*)(&v__4))[1]);
        *(uint2*)(yz_operand + (i_4 * 4)) = __5;
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
      for (int i_5 = 0; i_5 < 16; ++i_5) {
        float4 __6;
          float4 v__5 = *(float4*)(mt + (i_5 * 4));
          float4 v__6 = make_float4(expf(((float*)g)[(((c_1 & 1) * 64) + 63)]), expf(((float*)g)[(((c_1 & 1) * 64) + 63)]), expf(((float*)g)[(((c_1 & 1) * 64) + 63)]), expf(((float*)g)[(((c_1 & 1) * 64) + 63)]));
          __6.x = (v__5.x*v__6.x);
          __6.y = (v__5.y*v__6.y);
          __6.z = (v__5.z*v__6.z);
          __6.w = (v__5.w*v__6.w);
        *(float4*)(mt + (i_5 * 4)) = __6;
      }
      #pragma unroll
      for (int i_6 = 0; i_6 < 16; ++i_6) {
        uint2 __7;
        float4 v__7 = *(float4*)(nt + (i_6 * 4));
        (reinterpret_cast<__nv_bfloat162*>(&__7))[0] = __float22bfloat162_rn(((float2*)(&v__7))[0]);
        (reinterpret_cast<__nv_bfloat162*>(&__7))[1] = __float22bfloat162_rn(((float2*)(&v__7))[1]);
        *(uint2*)(operand + (i_6 * 4)) = __7;
      }
      {
        tl::GmmaDescriptor desc_b_3;
        tl::initialize_wgmma_descriptor<1, 1, 64>(desc_b_3, (&(((bfloat16_t*)ks)[0])));
        tl::increase_descriptor_offset<int>(desc_b_3, ((c_1 & 1) * 16384));
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
      mbarrier[((c_1 & 1) + 6)].arrive();
      mbarrier[((c_1 & 1) + 2)].wait(((c_1 & 3) >> 1));
      #pragma unroll
      for (int i_7 = 0; i_7 < 32; ++i_7) {
        yz_operand[i_7] = ((bfloat16_t)((expf(((float*)g)[(((c_1 & 1) * 64) + 63)]) * yz[i_7]) - (((float)((bfloat16_t*)vs)[(((((((((c_1 & 1) * 8192) + ((i_7 >> 2) * 1024)) + ((((int)threadIdx.x) & 3) * 256)) + ((i_7 & 1) * 128)) + ((((int)threadIdx.x) >> 5) * 16)) + (((i_7 & 3) >> 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2)) - 128)]) * expf((((float*)g)[(((c_1 & 1) * 64) + 63)] - ((float*)g)[(((((c_1 & 1) * 64) + ((i_7 >> 2) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + (i_7 & 1))])))));
      }
      mbarrier[((c_1 & 1) + 8)].arrive();
      #pragma unroll
      for (int i_8 = 0; i_8 < 16; ++i_8) {
        float4 __8;
          float4 v__8 = *(float4*)(nt + (i_8 * 4));
          float4 v__9 = make_float4(expf(((float*)g)[(((c_1 & 1) * 64) + 63)]), expf(((float*)g)[(((c_1 & 1) * 64) + 63)]), expf(((float*)g)[(((c_1 & 1) * 64) + 63)]), expf(((float*)g)[(((c_1 & 1) * 64) + 63)]));
          __8.x = (v__8.x*v__9.x);
          __8.y = (v__8.y*v__9.y);
          __8.z = (v__8.z*v__9.z);
          __8.w = (v__8.w*v__9.w);
        *(float4*)(nt + (i_8 * 4)) = __8;
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
    for (int i_9 = 0; i_9 < 64; ++i_9) {
      Nout[(((((((((((int)blockIdx.x) * 786432) + (((int)blockIdx.y) * 16384)) + ((i_9 >> 2) * 1024)) + ((((int)threadIdx.x) & 3) * 256)) + (((i_9 & 3) >> 1) * 128)) + ((((int)threadIdx.x) >> 5) * 16)) + ((i_9 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2)) - 128)] = ((bfloat16_t)nt[((((i_9 >> 2) * 4) + ((i_9 & 1) * 2)) + ((i_9 & 3) >> 1))]);
    }
    #pragma unroll
    for (int i_10 = 0; i_10 < 64; ++i_10) {
      Mout[(((((((((((int)blockIdx.x) * 786432) + (((int)blockIdx.y) * 16384)) + ((i_10 >> 2) * 1024)) + ((((int)threadIdx.x) & 3) * 256)) + (((i_10 & 3) >> 1) * 128)) + ((((int)threadIdx.x) >> 5) * 16)) + ((i_10 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2)) - 128)] = ((bfloat16_t)mt[((((i_10 >> 2) * 4) + ((i_10 & 1) * 2)) + ((i_10 & 3) >> 1))]);
    }
  }
}


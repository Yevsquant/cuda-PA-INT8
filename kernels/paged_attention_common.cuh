// Shared device helpers for the optimized paged-decode kernels.
//
// The naive kernel (paged_attention_naive.cu) stays self-contained; the
// optimized stages share these reductions and vectorized-load helpers so the
// per-stage .cu files only contain what each stage actually changes.

#pragma once

#include <cuda_fp16.h>
#include <math.h>

namespace pa {

// Block-wide tree reductions over `smem` (size >= blockDim.x).
__device__ inline float block_reduce_max(float val, float* smem) {
    const int tid = threadIdx.x;
    smem[tid] = val;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) smem[tid] = fmaxf(smem[tid], smem[tid + s]);
        __syncthreads();
    }
    const float r = smem[0];
    __syncthreads();
    return r;
}

__device__ inline float block_reduce_sum(float val, float* smem) {
    const int tid = threadIdx.x;
    smem[tid] = val;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid + s];
        __syncthreads();
    }
    const float r = smem[0];
    __syncthreads();
    return r;
}

// Intra-warp reductions via register shuffles (no shared memory).
__device__ inline float warp_reduce_max(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, offset));
    return val;
}

__device__ inline float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_xor_sync(0xffffffff, val, offset);
    return val;
}

// Two-level block reductions: intra-warp shuffle, then a cross-warp combine
// through `smem` (size >= number of warps). Replaces the log-step smem tree —
// register-to-register within a warp and only two __syncthreads total.
__device__ inline float block_reduce_max_warp(float val, float* smem) {
    const int lane = threadIdx.x & 31;
    const int wid = threadIdx.x >> 5;
    const int nwarps = (blockDim.x + 31) >> 5;
    val = warp_reduce_max(val);
    if (lane == 0) smem[wid] = val;
    __syncthreads();
    if (wid == 0) {
        float r = (lane < nwarps) ? smem[lane] : -INFINITY;
        r = warp_reduce_max(r);
        if (lane == 0) smem[0] = r;
    }
    __syncthreads();
    const float r = smem[0];
    __syncthreads();
    return r;
}

__device__ inline float block_reduce_sum_warp(float val, float* smem) {
    const int lane = threadIdx.x & 31;
    const int wid = threadIdx.x >> 5;
    const int nwarps = (blockDim.x + 31) >> 5;
    val = warp_reduce_sum(val);
    if (lane == 0) smem[wid] = val;
    __syncthreads();
    if (wid == 0) {
        float r = (lane < nwarps) ? smem[lane] : 0.f;
        r = warp_reduce_sum(r);
        if (lane == 0) smem[0] = r;
    }
    __syncthreads();
    const float r = smem[0];
    __syncthreads();
    return r;
}

// dot(q, k_token) for one token, reading K vectorized along the x dim.
//   k_base points at k_cache[blk, kv_head, 0, 0, 0]; layout [head_dim/x, block_size, x].
//   x == 8 (16 B = one 128-bit load). q_sh is the head's query in shared fp32.
__device__ inline float qk_dot_vectorized(
    const float* __restrict__ q_sh, const __half* __restrict__ k_base,
    int off, int hd_x, int block_size, int x)
{
    float dot = 0.f;
    for (int outer = 0; outer < hd_x; ++outer) {
        const uint4 raw =
            *reinterpret_cast<const uint4*>(&k_base[(outer * block_size + off) * x]);
        const __half2* h2 = reinterpret_cast<const __half2*>(&raw);
        #pragma unroll
        for (int j = 0; j < 4; ++j) {  // x/2 == 4 half2 per 128-bit load
            const float2 f = __half22float2(h2[j]);
            const int d = outer * x + 2 * j;
            dot += q_sh[d] * f.x + q_sh[d + 1] * f.y;
        }
    }
    return dot;
}

} // namespace pa

// Stage 5 — Warp-level reductions with INT8 KV-cache (paged decode).
//
// Identical structure to Stage 3 (paged_attention_v3_warp.cu); the only change
// is the KV cache is INT8 with a per-token fp32 scale, dequantized inline on
// read. Because the scale is per-token (constant across head_dim) the dequant
// folds cheaply — no per-element multiply:
//   K: dot(q, k_int8·s_k) = s_k · dot(q, k_int8)  → multiply the raw int8 dot
//      once per token by s_k[token].
//   V: prob·(v_int8·s_v) = (prob·s_v)·v_int8      → fold s_v[token] into the
//      per-token softmax weight once, then accumulate the raw int8 V.
//
// Scales layout: k_scales/v_scales [num_blocks, num_kv_heads, block_size] fp32.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>
#include <math.h>

#include "paged_attention_common.cuh"

namespace {

constexpr int TILE = 128;

__global__ void paged_decode_attn_warp_int8_kernel(
    __half* __restrict__ out,
    const __half* __restrict__ q,
    const signed char* __restrict__ k_cache,
    const signed char* __restrict__ v_cache,
    const float* __restrict__ k_scales,
    const float* __restrict__ v_scales,
    const int* __restrict__ block_table,
    const int* __restrict__ context_lens,
    float scale,
    int num_heads, int num_kv_heads, int head_dim,
    int block_size, int max_blocks, int x)
{
    const int seq = blockIdx.x;
    const int head = blockIdx.y;
    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;
    const int kv_head = head / (num_heads / num_kv_heads);
    const int ctx = context_lens[seq];

    extern __shared__ float smem[];
    float* q_sh = smem;                  // [head_dim]
    float* acc_sh = q_sh + head_dim;     // [head_dim]
    float* scores = acc_sh + head_dim;   // [TILE]
    __shared__ float red[32];            // per-warp partials (<=32 warps)
    __shared__ float m_run, l_run;

    const __half* q_ptr = q + ((long)seq * num_heads + head) * head_dim;
    for (int d = tid * 8; d < head_dim; d += nthreads * 8) {
        const uint4 raw = *reinterpret_cast<const uint4*>(&q_ptr[d]);
        const __half2* h2 = reinterpret_cast<const __half2*>(&raw);
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float2 f = __half22float2(h2[j]);
            q_sh[d + 2 * j] = f.x;
            q_sh[d + 2 * j + 1] = f.y;
        }
    }
    for (int d = tid; d < head_dim; d += nthreads) acc_sh[d] = 0.f;
    if (tid == 0) { m_run = -INFINITY; l_run = 0.f; }
    __syncthreads();

    const int hd_x = head_dim / x;
    const int* bt = block_table + (long)seq * max_blocks;

    for (int tile0 = 0; tile0 < ctx; tile0 += TILE) {
        const int ntok = min(TILE, ctx - tile0);

        float local_max = -INFINITY;
        for (int i = tid; i < TILE; i += nthreads) {
            if (i < ntok) {
                const int t = tile0 + i;
                const int blk = bt[t / block_size];
                const int off = t % block_size;
                const signed char* k_base =
                    k_cache + (((long)blk * num_kv_heads + kv_head) * hd_x) * block_size * x;
                const float s_k =
                    k_scales[((long)blk * num_kv_heads + kv_head) * block_size + off];
                const float s =
                    pa::qk_dot_int8(q_sh, k_base, off, hd_x, block_size, x) * s_k * scale;
                scores[i] = s;
                local_max = fmaxf(local_max, s);
            } else {
                scores[i] = 0.f;
            }
        }
        const float m_tile = pa::block_reduce_max_warp(local_max, red);
        const float m_new = fmaxf(m_run, m_tile);
        const float correction = __expf(m_run - m_new);

        float local_sum = 0.f;
        for (int i = tid; i < TILE; i += nthreads) {
            if (i < ntok) {
                const float w = __expf(scores[i] - m_new);
                local_sum += w;
                // Fold the per-token V scale into the softmax weight so the V
                // accumulation below uses raw int8 (s_v constant across head_dim).
                const int t = tile0 + i;
                const int blk = bt[t / block_size];
                const int off = t % block_size;
                const float s_v =
                    v_scales[((long)blk * num_kv_heads + kv_head) * block_size + off];
                scores[i] = w * s_v;
            } else {
                scores[i] = 0.f;
            }
        }
        const float l_tile = pa::block_reduce_sum_warp(local_sum, red);

        const int nblk = (ntok + block_size - 1) / block_size;
        const int base_blk = tile0 / block_size;
        for (int d = tid; d < head_dim; d += nthreads) {
            float acc = acc_sh[d] * correction;
            for (int jb = 0; jb < nblk; ++jb) {
                const int blk = bt[base_blk + jb];
                const signed char* v_base =
                    v_cache + (((long)blk * num_kv_heads + kv_head) * head_dim + d) * block_size;
                const int woff = jb * block_size;
                for (int c0 = 0; c0 < block_size; c0 += 16) {
                    const uint4 raw = *reinterpret_cast<const uint4*>(&v_base[c0]);
                    const signed char* c = reinterpret_cast<const signed char*>(&raw);
                    #pragma unroll
                    for (int j = 0; j < 16; ++j) {
                        acc += scores[woff + c0 + j] * static_cast<float>(c[j]);
                    }
                }
            }
            acc_sh[d] = acc;
        }
        if (tid == 0) {
            l_run = l_run * correction + l_tile;
            m_run = m_new;
        }
        __syncthreads();
    }

    const float inv = 1.f / l_run;
    __half* out_ptr = out + ((long)seq * num_heads + head) * head_dim;
    for (int d = tid; d < head_dim; d += nthreads)
        out_ptr[d] = __float2half(acc_sh[d] * inv);
}

} // namespace

void paged_decode_attn_warp_int8(
    torch::Tensor out, torch::Tensor q,
    torch::Tensor k_cache, torch::Tensor v_cache,
    torch::Tensor k_scales, torch::Tensor v_scales,
    torch::Tensor block_table, torch::Tensor context_lens,
    double scale, int64_t block_size)
{
    TORCH_CHECK(q.is_cuda() && out.is_cuda(), "tensors must be on CUDA");
    TORCH_CHECK(q.scalar_type() == torch::kHalf, "q must be fp16");
    TORCH_CHECK(k_cache.scalar_type() == torch::kChar, "k_cache must be int8");
    TORCH_CHECK(v_cache.scalar_type() == torch::kChar, "v_cache must be int8");
    TORCH_CHECK(k_scales.scalar_type() == torch::kFloat32, "k_scales must be fp32");
    TORCH_CHECK(v_scales.scalar_type() == torch::kFloat32, "v_scales must be fp32");
    TORCH_CHECK(block_table.scalar_type() == torch::kInt, "block_table must be int32");
    TORCH_CHECK(context_lens.scalar_type() == torch::kInt, "context_lens must be int32");
    TORCH_CHECK(context_lens.numel() == 0 || context_lens.min().item<int>() >= 1,
                "context_lens must be >= 1 (empty sequences unsupported)");
    TORCH_CHECK(block_size % 16 == 0 && TILE % block_size == 0,
                "block_size must divide TILE and be a multiple of 16");

    const int num_seqs = q.size(0);
    const int num_heads = q.size(1);
    const int head_dim = q.size(2);
    const int num_kv_heads = k_cache.size(1);
    const int x = k_cache.size(4);
    const int max_blocks = block_table.size(1);
    TORCH_CHECK(x == 8, "int8 K cache requires x == 8");

    const dim3 grid(num_seqs, num_heads);
    const int threads = 128;
    const size_t smem = (2 * head_dim + TILE) * sizeof(float);

    const at::cuda::OptionalCUDAGuard guard(device_of(q));
    auto stream = at::cuda::getCurrentCUDAStream();

    paged_decode_attn_warp_int8_kernel<<<grid, threads, smem, stream>>>(
        reinterpret_cast<__half*>(out.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(q.data_ptr<at::Half>()),
        reinterpret_cast<const signed char*>(k_cache.data_ptr<int8_t>()),
        reinterpret_cast<const signed char*>(v_cache.data_ptr<int8_t>()),
        k_scales.data_ptr<float>(),
        v_scales.data_ptr<float>(),
        block_table.data_ptr<int>(),
        context_lens.data_ptr<int>(),
        static_cast<float>(scale),
        num_heads, num_kv_heads, head_dim,
        static_cast<int>(block_size), max_blocks, x);
}

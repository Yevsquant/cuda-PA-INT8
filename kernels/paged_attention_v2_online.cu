// Stage 2 — Online (single-pass) softmax (FP16 paged decode).
//
// The naive/vec kernels make three sweeps over the context (score, exp/sum,
// V-weighted-sum) and keep a logits[ctx] array in shared memory, which caps both
// occupancy and the max context length. Here we make ONE sweep: KV is processed
// in tiles of TILE tokens, carrying running (m, l) softmax stats and a shared
// accumulator acc[head_dim], rescaled by exp(m_old - m_new) whenever the max
// grows. Shared memory is now O(head_dim + TILE) — constant in ctx.
//
// Work split per tile (both phases use every thread):
//   phase A: thread i computes the score of tile-token i (vectorized Q·K).
//   phase B: thread d accumulates output dim d over the tile's tokens (vec V).

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>
#include <math.h>

#include "paged_attention_common.cuh"

namespace {

constexpr int TILE = 128;  // tokens per tile; multiple of block_size (16)

__global__ void paged_decode_attn_online_kernel(
    __half* __restrict__ out,
    const __half* __restrict__ q,
    const __half* __restrict__ k_cache,
    const __half* __restrict__ v_cache,
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
    float* acc_sh = q_sh + head_dim;     // [head_dim]  running output accumulator
    float* scores = acc_sh + head_dim;   // [TILE]      per-tile softmax weights
    __shared__ float red[256];
    __shared__ float m_run, l_run;       // running max / denominator (block-wide)

    // Vectorized Q load into shared fp32.
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

        // Phase A: scores for this tile, and the tile max.
        float local_max = -INFINITY;
        for (int i = tid; i < TILE; i += nthreads) {
            if (i < ntok) {
                const int t = tile0 + i;
                const int blk = bt[t / block_size];
                const int off = t % block_size;
                const __half* k_base =
                    k_cache + (((long)blk * num_kv_heads + kv_head) * hd_x) * block_size * x;
                const float s =
                    pa::qk_dot_vectorized(q_sh, k_base, off, hd_x, block_size, x) * scale;
                scores[i] = s;
                local_max = fmaxf(local_max, s);
            } else {
                scores[i] = 0.f;  // padding weights -> 0 after exp
            }
        }
        const float m_tile = pa::block_reduce_max(local_max, red);
        const float m_new = fmaxf(m_run, m_tile);
        const float correction = __expf(m_run - m_new);  // 0 on first tile (m_run=-inf)

        // Convert tile scores to weights at the new max; sum the tile denominator.
        float local_sum = 0.f;
        for (int i = tid; i < TILE; i += nthreads) {
            if (i < ntok) {
                const float w = __expf(scores[i] - m_new);
                scores[i] = w;
                local_sum += w;
            } else {
                scores[i] = 0.f;
            }
        }
        const float l_tile = pa::block_reduce_sum(local_sum, red);

        // Phase B: rescale running acc, then add this tile's V contribution.
        const int nblk = (ntok + block_size - 1) / block_size;
        const int base_blk = tile0 / block_size;
        for (int d = tid; d < head_dim; d += nthreads) {
            float acc = acc_sh[d] * correction;
            for (int jb = 0; jb < nblk; ++jb) {
                const int blk = bt[base_blk + jb];
                const __half* v_base =
                    v_cache + (((long)blk * num_kv_heads + kv_head) * head_dim + d) * block_size;
                const int woff = jb * block_size;
                for (int c0 = 0; c0 < block_size; c0 += 8) {
                    const uint4 raw = *reinterpret_cast<const uint4*>(&v_base[c0]);
                    const __half2* h2 = reinterpret_cast<const __half2*>(&raw);
                    #pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        const float2 f = __half22float2(h2[j]);
                        acc += scores[woff + c0 + 2 * j] * f.x;
                        acc += scores[woff + c0 + 2 * j + 1] * f.y;
                    }
                }
            }
            acc_sh[d] = acc;
        }
        if (tid == 0) {
            l_run = l_run * correction + l_tile;
            m_run = m_new;
        }
        __syncthreads();  // m_run/l_run/acc_sh visible to next tile
    }

    const float inv = 1.f / l_run;
    __half* out_ptr = out + ((long)seq * num_heads + head) * head_dim;
    for (int d = tid; d < head_dim; d += nthreads)
        out_ptr[d] = __float2half(acc_sh[d] * inv);
}

} // namespace

void paged_decode_attn_online(
    torch::Tensor out, torch::Tensor q,
    torch::Tensor k_cache, torch::Tensor v_cache,
    torch::Tensor block_table, torch::Tensor context_lens,
    double scale, int64_t block_size)
{
    TORCH_CHECK(q.is_cuda() && out.is_cuda(), "tensors must be on CUDA");
    TORCH_CHECK(q.scalar_type() == torch::kHalf, "q must be fp16");
    TORCH_CHECK(block_table.scalar_type() == torch::kInt, "block_table must be int32");
    TORCH_CHECK(context_lens.scalar_type() == torch::kInt, "context_lens must be int32");
    TORCH_CHECK(block_size % 8 == 0 && TILE % block_size == 0,
                "block_size must divide TILE and be a multiple of 8");

    const int num_seqs = q.size(0);
    const int num_heads = q.size(1);
    const int head_dim = q.size(2);
    const int num_kv_heads = k_cache.size(1);
    const int x = k_cache.size(4);
    const int max_blocks = block_table.size(1);

    const dim3 grid(num_seqs, num_heads);
    const int threads = 128;
    const size_t smem = (2 * head_dim + TILE) * sizeof(float);

    const at::cuda::OptionalCUDAGuard guard(device_of(q));
    auto stream = at::cuda::getCurrentCUDAStream();

    paged_decode_attn_online_kernel<<<grid, threads, smem, stream>>>(
        reinterpret_cast<__half*>(out.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(q.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(k_cache.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(v_cache.data_ptr<at::Half>()),
        block_table.data_ptr<int>(),
        context_lens.data_ptr<int>(),
        static_cast<float>(scale),
        num_heads, num_kv_heads, head_dim,
        static_cast<int>(block_size), max_blocks, x);
}

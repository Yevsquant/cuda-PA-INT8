// Stage 1 — Vectorized loads (FP16 paged decode).
//
// Same structure as the naive kernel (one block per (seq, head), two-pass
// softmax, smem-tree reductions) — the ONLY change is that K, V and Q are read
// with 128-bit (uint4 = 8 fp16) loads instead of one __half at a time. K's `x`
// dim is contiguous and 16 B wide, so a single transaction replaces 8 scalar
// ones; V is vectorized across its contiguous block_size (token) dim.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>
#include <math.h>

#include "paged_attention_common.cuh"

namespace {

__global__ void paged_decode_attn_vec_kernel(
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
    float* q_sh = smem;               // [head_dim]
    float* logits = smem + head_dim;  // [ctx]
    __shared__ float red[256];

    // Load q vectorized (8 fp16 per 128-bit load) into shared fp32.
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
    __syncthreads();

    const int hd_x = head_dim / x;
    const int* bt = block_table + (long)seq * max_blocks;

    // Pass 1: scores via vectorized Q·K, track running max.
    float local_max = -INFINITY;
    for (int t = tid; t < ctx; t += nthreads) {
        const int blk = bt[t / block_size];
        const int off = t % block_size;
        const __half* k_base =
            k_cache + (((long)blk * num_kv_heads + kv_head) * hd_x) * block_size * x;
        const float s = pa::qk_dot_vectorized(q_sh, k_base, off, hd_x, block_size, x) * scale;
        logits[t] = s;
        local_max = fmaxf(local_max, s);
    }
    const float gmax = pa::block_reduce_max(local_max, red);

    // Pass 2: exp(score - max), accumulate denominator.
    float local_sum = 0.f;
    for (int t = tid; t < ctx; t += nthreads) {
        const float e = __expf(logits[t] - gmax);
        logits[t] = e;
        local_sum += e;
    }
    const float gsum = pa::block_reduce_sum(local_sum, red);
    const float inv = 1.f / gsum;

    // Output: parallelize over d, accumulate over tokens. V is read vectorized
    // across the contiguous block_size dim (full blocks; padding slots are zero
    // and masked by the t < ctx guard).
    const int nblk = (ctx + block_size - 1) / block_size;
    __half* out_ptr = out + ((long)seq * num_heads + head) * head_dim;
    for (int d = tid; d < head_dim; d += nthreads) {
        float acc = 0.f;
        for (int jb = 0; jb < nblk; ++jb) {
            const int blk = bt[jb];
            const int t0 = jb * block_size;
            const __half* v_base =
                v_cache + (((long)blk * num_kv_heads + kv_head) * head_dim + d) * block_size;
            for (int c0 = 0; c0 < block_size; c0 += 8) {
                const uint4 raw = *reinterpret_cast<const uint4*>(&v_base[c0]);
                const __half2* h2 = reinterpret_cast<const __half2*>(&raw);
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const float2 f = __half22float2(h2[j]);
                    const int t = t0 + c0 + 2 * j;
                    if (t < ctx) acc += logits[t] * f.x;
                    if (t + 1 < ctx) acc += logits[t + 1] * f.y;
                }
            }
        }
        out_ptr[d] = __float2half(acc * inv);
    }
}

} // namespace

void paged_decode_attn_vec(
    torch::Tensor out, torch::Tensor q,
    torch::Tensor k_cache, torch::Tensor v_cache,
    torch::Tensor block_table, torch::Tensor context_lens,
    double scale, int64_t block_size)
{
    TORCH_CHECK(q.is_cuda() && out.is_cuda(), "tensors must be on CUDA");
    TORCH_CHECK(q.scalar_type() == torch::kHalf, "q must be fp16");
    TORCH_CHECK(block_table.scalar_type() == torch::kInt, "block_table must be int32");
    TORCH_CHECK(context_lens.scalar_type() == torch::kInt, "context_lens must be int32");
    TORCH_CHECK(block_size % 8 == 0, "block_size must be a multiple of 8 for vectorized V");

    const int num_seqs = q.size(0);
    const int num_heads = q.size(1);
    const int head_dim = q.size(2);
    const int num_kv_heads = k_cache.size(1);
    const int x = k_cache.size(4);
    const int max_blocks = block_table.size(1);
    const int max_ctx = context_lens.max().item<int>();

    const dim3 grid(num_seqs, num_heads);
    const int threads = 128;
    const size_t smem = (head_dim + max_ctx) * sizeof(float);

    const at::cuda::OptionalCUDAGuard guard(device_of(q));
    auto stream = at::cuda::getCurrentCUDAStream();

    paged_decode_attn_vec_kernel<<<grid, threads, smem, stream>>>(
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

// Stage 4 — Flash-Decoding split-K (FP16 paged decode).
//
// At decode the batch is 1–8, so a grid of num_seqs*num_heads blocks leaves most
// of the A100's 108 SMs idle while each block serially walks a long context.
// Split-K partitions each sequence's context across num_splits blocks, so the
// grid becomes num_seqs*num_heads*num_splits and the long KV read is done in
// parallel. Each partition block emits an UN-normalized partial (acc, m, l) over
// its token range using the same online sweep as Stage 3; a second reduce kernel
// merges the partials with the log-sum-exp combine.
//
// num_splits = ceil(max_ctx / PARTITION_SIZE), mirroring vLLM v2's heuristic.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>
#include <math.h>

#include "paged_attention_common.cuh"

namespace {

constexpr int TILE = 128;
constexpr int PARTITION_SIZE = 512;  // tokens per split (multiple of TILE)

// One block per (seq, head, split). Walks only its split's token range and
// writes the un-normalized partial: acc = sum exp(s - m)*V, plus (m, l).
__global__ void paged_decode_splitk_partition_kernel(
    float* __restrict__ out_partial,   // [num_seqs, num_heads, num_splits, head_dim]
    float* __restrict__ m_partial,     // [num_seqs, num_heads, num_splits]
    float* __restrict__ l_partial,     // [num_seqs, num_heads, num_splits]
    const __half* __restrict__ q,
    const __half* __restrict__ k_cache,
    const __half* __restrict__ v_cache,
    const int* __restrict__ block_table,
    const int* __restrict__ context_lens,
    float scale,
    int num_heads, int num_kv_heads, int head_dim,
    int block_size, int max_blocks, int x, int num_splits)
{
    const int seq = blockIdx.x;
    const int head = blockIdx.y;
    const int split = blockIdx.z;
    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;
    const int kv_head = head / (num_heads / num_kv_heads);
    const int ctx = context_lens[seq];

    const long part_idx = ((long)seq * num_heads + head) * num_splits + split;

    const int t_start = split * PARTITION_SIZE;
    const int t_end = min(ctx, t_start + PARTITION_SIZE);
    if (t_start >= t_end) {  // this split has no tokens
        if (tid == 0) { m_partial[part_idx] = -INFINITY; l_partial[part_idx] = 0.f; }
        for (int d = tid; d < head_dim; d += nthreads)
            out_partial[part_idx * head_dim + d] = 0.f;
        return;
    }

    extern __shared__ float smem[];
    float* q_sh = smem;                  // [head_dim]
    float* acc_sh = q_sh + head_dim;     // [head_dim]
    float* scores = acc_sh + head_dim;   // [TILE]
    __shared__ float red[32];
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

    for (int tile0 = t_start; tile0 < t_end; tile0 += TILE) {
        const int ntok = min(TILE, t_end - tile0);

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
                scores[i] = w;
                local_sum += w;
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
        __syncthreads();
    }

    // Write the un-normalized partial for this split.
    for (int d = tid; d < head_dim; d += nthreads)
        out_partial[part_idx * head_dim + d] = acc_sh[d];
    if (tid == 0) { m_partial[part_idx] = m_run; l_partial[part_idx] = l_run; }
}

// One block per (seq, head). Log-sum-exp merge over the splits' partials.
__global__ void paged_decode_splitk_reduce_kernel(
    __half* __restrict__ out,          // [num_seqs, num_heads, head_dim]
    const float* __restrict__ out_partial,
    const float* __restrict__ m_partial,
    const float* __restrict__ l_partial,
    int num_heads, int head_dim, int num_splits)
{
    const int seq = blockIdx.x;
    const int head = blockIdx.y;
    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;

    const long base = (long)seq * num_heads + head;
    const float* mp = m_partial + base * num_splits;
    const float* lp = l_partial + base * num_splits;
    const float* op = out_partial + base * num_splits * head_dim;

    // num_splits is small; each thread recomputes the (cheap) merge scalars.
    float gm = -INFINITY;
    for (int s = 0; s < num_splits; ++s) gm = fmaxf(gm, mp[s]);
    float denom = 0.f;
    for (int s = 0; s < num_splits; ++s) denom += lp[s] * __expf(mp[s] - gm);
    const float inv = 1.f / denom;

    __half* out_ptr = out + base * head_dim;
    for (int d = tid; d < head_dim; d += nthreads) {
        float acc = 0.f;
        for (int s = 0; s < num_splits; ++s)
            acc += op[(long)s * head_dim + d] * __expf(mp[s] - gm);
        out_ptr[d] = __float2half(acc * inv);
    }
}

} // namespace

void paged_decode_attn_splitk(
    torch::Tensor out, torch::Tensor q,
    torch::Tensor k_cache, torch::Tensor v_cache,
    torch::Tensor block_table, torch::Tensor context_lens,
    double scale, int64_t block_size)
{
    TORCH_CHECK(q.is_cuda() && out.is_cuda(), "tensors must be on CUDA");
    TORCH_CHECK(q.scalar_type() == torch::kHalf, "q must be fp16");
    TORCH_CHECK(block_table.scalar_type() == torch::kInt, "block_table must be int32");
    TORCH_CHECK(context_lens.scalar_type() == torch::kInt, "context_lens must be int32");
    TORCH_CHECK(block_size % 8 == 0 && TILE % block_size == 0 &&
                PARTITION_SIZE % TILE == 0,
                "block_size must divide TILE and TILE must divide PARTITION_SIZE");

    const int num_seqs = q.size(0);
    const int num_heads = q.size(1);
    const int head_dim = q.size(2);
    const int num_kv_heads = k_cache.size(1);
    const int x = k_cache.size(4);
    const int max_blocks = block_table.size(1);
    const int max_ctx = context_lens.max().item<int>();
    const int num_splits = max(1, (max_ctx + PARTITION_SIZE - 1) / PARTITION_SIZE);

    auto fopts = torch::TensorOptions().dtype(torch::kFloat32).device(q.device());
    auto out_partial = torch::empty({num_seqs, num_heads, num_splits, head_dim}, fopts);
    auto m_partial = torch::empty({num_seqs, num_heads, num_splits}, fopts);
    auto l_partial = torch::empty({num_seqs, num_heads, num_splits}, fopts);

    const int threads = 128;
    const size_t smem = (2 * head_dim + TILE) * sizeof(float);

    const at::cuda::OptionalCUDAGuard guard(device_of(q));
    auto stream = at::cuda::getCurrentCUDAStream();

    const dim3 pgrid(num_seqs, num_heads, num_splits);
    paged_decode_splitk_partition_kernel<<<pgrid, threads, smem, stream>>>(
        out_partial.data_ptr<float>(),
        m_partial.data_ptr<float>(),
        l_partial.data_ptr<float>(),
        reinterpret_cast<const __half*>(q.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(k_cache.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(v_cache.data_ptr<at::Half>()),
        block_table.data_ptr<int>(),
        context_lens.data_ptr<int>(),
        static_cast<float>(scale),
        num_heads, num_kv_heads, head_dim,
        static_cast<int>(block_size), max_blocks, x, num_splits);

    const dim3 rgrid(num_seqs, num_heads);
    paged_decode_splitk_reduce_kernel<<<rgrid, threads, 0, stream>>>(
        reinterpret_cast<__half*>(out.data_ptr<at::Half>()),
        out_partial.data_ptr<float>(),
        m_partial.data_ptr<float>(),
        l_partial.data_ptr<float>(),
        num_heads, head_dim, num_splits);
}

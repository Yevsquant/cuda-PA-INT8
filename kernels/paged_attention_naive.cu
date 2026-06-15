// Naive PagedAttention decode kernel (FP16).
//
// One thread block per (seq, head). Two-pass softmax (find max, then exp/sum),
// then a weighted sum over V. No vectorization, no online softmax, no split-K —
// this is the correctness baseline the optimized stages are measured against.
//
// KV-cache layout (real vLLM, fp16, x = 16/sizeof(dtype) = 8):
//   K: [num_blocks, num_kv_heads, head_dim/x, block_size, x]
//   V: [num_blocks, num_kv_heads, head_dim,   block_size]

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>
#include <math.h>

namespace {

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

__global__ void paged_decode_attn_kernel(
    __half* __restrict__ out,             // [num_seqs, num_heads, head_dim]
    const __half* __restrict__ q,         // [num_seqs, num_heads, head_dim]
    const __half* __restrict__ k_cache,   // [num_blocks, num_kv_heads, head_dim/x, block_size, x]
    const __half* __restrict__ v_cache,   // [num_blocks, num_kv_heads, head_dim, block_size]
    const int* __restrict__ block_table,  // [num_seqs, max_blocks]
    const int* __restrict__ context_lens, // [num_seqs]
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
    __shared__ float red[256];        // reduction scratch (blockDim <= 256)

    // Load q for this (seq, head) into shared memory.
    const __half* q_ptr = q + ((long)seq * num_heads + head) * head_dim;
    for (int d = tid; d < head_dim; d += nthreads) q_sh[d] = __half2float(q_ptr[d]);
    __syncthreads();

    const int hd_x = head_dim / x;
    const int* bt = block_table + (long)seq * max_blocks;

    // Pass 1: scores = scale * dot(q, k_t), track running max.
    float local_max = -INFINITY;
    for (int t = tid; t < ctx; t += nthreads) {
        const int blk = bt[t / block_size];
        const int off = t % block_size;
        // k_cache[blk, kv_head, :, off, :]
        const __half* k_base =
            k_cache + (((long)blk * num_kv_heads + kv_head) * hd_x) * block_size * x;
        float dot = 0.f;
        for (int d = 0; d < head_dim; ++d) {
            const int outer = d / x;
            const int inner = d % x;
            const float kv = __half2float(k_base[(outer * block_size + off) * x + inner]);
            dot += q_sh[d] * kv;
        }
        const float s = dot * scale;
        logits[t] = s;
        local_max = fmaxf(local_max, s);
    }
    const float gmax = block_reduce_max(local_max, red);

    // Pass 2: exp(score - max), accumulate denominator.
    float local_sum = 0.f;
    for (int t = tid; t < ctx; t += nthreads) {
        const float e = __expf(logits[t] - gmax);
        logits[t] = e;
        local_sum += e;
    }
    const float gsum = block_reduce_sum(local_sum, red);
    const float inv = 1.f / gsum;

    // Output: out[d] = (1/sum) * sum_t prob[t] * v[t, d]. Parallelize over d.
    __half* out_ptr = out + ((long)seq * num_heads + head) * head_dim;
    for (int d = tid; d < head_dim; d += nthreads) {
        float acc = 0.f;
        for (int t = 0; t < ctx; ++t) {
            const int blk = bt[t / block_size];
            const int off = t % block_size;
            // v_cache[blk, kv_head, d, off]
            const __half* v_base =
                v_cache + (((long)blk * num_kv_heads + kv_head) * head_dim + d) * block_size;
            acc += logits[t] * __half2float(v_base[off]);
        }
        out_ptr[d] = __float2half(acc * inv);
    }
}

} // namespace

void paged_decode_attention(
    torch::Tensor out,
    torch::Tensor q,
    torch::Tensor k_cache,
    torch::Tensor v_cache,
    torch::Tensor block_table,
    torch::Tensor context_lens,
    double scale,
    int64_t block_size)
{
    TORCH_CHECK(q.is_cuda() && out.is_cuda(), "tensors must be on CUDA");
    TORCH_CHECK(q.scalar_type() == torch::kHalf, "q must be fp16");
    TORCH_CHECK(block_table.scalar_type() == torch::kInt, "block_table must be int32");
    TORCH_CHECK(context_lens.scalar_type() == torch::kInt, "context_lens must be int32");
    TORCH_CHECK(context_lens.numel() == 0 || context_lens.min().item<int>() >= 1,
                "context_lens must be >= 1 (empty sequences unsupported)");

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

    paged_decode_attn_kernel<<<grid, threads, smem, stream>>>(
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

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("paged_decode_attention", &paged_decode_attention,
          "Naive paged decode attention (FP16)");
}

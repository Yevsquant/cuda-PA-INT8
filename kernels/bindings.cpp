// Single pybind11 module for the optimized paged-decode stages. Each stage
// lives in its own .cu and exports one launcher with the shared signature; this
// file links them so the benchmark calls every variant in one process.

#include <torch/extension.h>

void paged_decode_attn_vec(
    torch::Tensor out, torch::Tensor q,
    torch::Tensor k_cache, torch::Tensor v_cache,
    torch::Tensor block_table, torch::Tensor context_lens,
    double scale, int64_t block_size);

void paged_decode_attn_online(
    torch::Tensor out, torch::Tensor q,
    torch::Tensor k_cache, torch::Tensor v_cache,
    torch::Tensor block_table, torch::Tensor context_lens,
    double scale, int64_t block_size);

void paged_decode_attn_warp(
    torch::Tensor out, torch::Tensor q,
    torch::Tensor k_cache, torch::Tensor v_cache,
    torch::Tensor block_table, torch::Tensor context_lens,
    double scale, int64_t block_size);

void paged_decode_attn_splitk(
    torch::Tensor out, torch::Tensor q,
    torch::Tensor k_cache, torch::Tensor v_cache,
    torch::Tensor block_table, torch::Tensor context_lens,
    double scale, int64_t block_size);

void paged_decode_attn_warp_int8(
    torch::Tensor out, torch::Tensor q,
    torch::Tensor k_cache, torch::Tensor v_cache,
    torch::Tensor k_scales, torch::Tensor v_scales,
    torch::Tensor block_table, torch::Tensor context_lens,
    double scale, int64_t block_size,
    c10::optional<torch::Tensor> k_zeros);

void paged_decode_attn_splitk_int8(
    torch::Tensor out, torch::Tensor q,
    torch::Tensor k_cache, torch::Tensor v_cache,
    torch::Tensor k_scales, torch::Tensor v_scales,
    torch::Tensor block_table, torch::Tensor context_lens,
    double scale, int64_t block_size,
    c10::optional<torch::Tensor> k_zeros);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("paged_decode_attn_vec", &paged_decode_attn_vec,
          "Stage 1: vectorized-load paged decode (FP16)");
    m.def("paged_decode_attn_online", &paged_decode_attn_online,
          "Stage 2: online single-pass softmax paged decode (FP16)");
    m.def("paged_decode_attn_warp", &paged_decode_attn_warp,
          "Stage 3: warp-reduction online paged decode (FP16)");
    m.def("paged_decode_attn_splitk", &paged_decode_attn_splitk,
          "Stage 4: Flash-Decoding split-K paged decode (FP16)");
    m.def("paged_decode_attn_warp_int8", &paged_decode_attn_warp_int8,
          "Stage 5: warp-reduction paged decode (INT8 KV cache)");
    m.def("paged_decode_attn_splitk_int8", &paged_decode_attn_splitk_int8,
          "Stage 6: Flash-Decoding split-K paged decode (INT8 KV cache)");
}

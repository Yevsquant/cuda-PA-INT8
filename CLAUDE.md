# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

CUDA PagedAttention decode kernel with INT8 KV-cache quantization, built from scratch. Two independent tracks:

- **Track A (Kernel):** Custom CUDA kernels — PagedAttention decode (FP16), per-token INT8 KV-cache quantization/dequantization fused into attention. Validated against a PyTorch reference implementation, benchmarked against vLLM's `paged_attention_v1/v2`.
- **Track B (System):** vLLM speculative decoding benchmarks (n-gram / EAGLE) with Qwen2.5-1.5B-Instruct on A100, measuring TTFT, TPOT, throughput, draft acceptance rate, and memory.

Motivation: vLLM lacks native INT8 KV-cache support (see vLLM issues #33480, RFC #37319). This project fills that gap at the kernel level.

## Target Environment

- **Hardware:** iCRN, A100-80GB
- **Interactive GPU:** `srun --partition=gpuA100x4 --gres=gpu:1 ...`
- **Python env:** `uv venv`, PyTorch matching cluster CUDA driver, pinned vLLM version
- **Fallback:** RunPod 4090 for kernel development; A100 for final benchmarks only

## Intended Directory Structure

```
kernels/       # CUDA kernel sources (.cu/.cuh)
tests/         # pytest — correctness tests against PyTorch reference
benchmarks/    # Microbenchmark scripts (kernel-level) + vLLM serving benchmarks
report/        # benchmark_report.md → PDF
```

## Build & Test Commands

```bash
# Build CUDA kernels (once build system is set up)
# Kernels are exposed to Python via PyTorch custom ops or pybind11

# Run all tests
pytest tests/

# Run a single test
pytest tests/test_paged_attention.py -k "test_name"

# Profile with Nsight Compute
ncu --set full ./kernel_binary
```

## Key Technical Details

### KV-Cache Layout
vLLM block layout: `[num_blocks, num_kv_heads, head_size/x, block_size, x]` with `block_size=16`. The `x` dimension aligns with 128-bit vectorized loads (`float4`).

### Kernel Optimization Stages (in order)
1. Naive: one thread block per (seq, head), two-pass softmax
2. Vectorized loads (`float4` / 128-bit, aligned to KV layout `x` dim)
3. Online softmax (single-pass, numerically stable)
4. Warp-level reduction (`__shfl_xor_sync`) + shared memory for logits
5. Split-K (Flash-Decoding style): partition KV blocks across thread blocks for long sequences

### INT8 Quantization
- **Per-token dynamic quantization:** each token×head gets one fp32 scale, stored in a separate buffer
- Dequantization is fused inline during KV reads in the attention kernel (no separate pass)
- Compare against: per-tensor static quantization (ablation), FP8, FP16

### Performance Targets
- Reach 80%+ of vLLM's official kernel performance
- A100 DRAM bandwidth utilization: target 70%+ of 1555 GB/s peak
- INT8 KV should halve memory vs FP16 with minimal perplexity degradation

### Precision Evaluation
- Kernel-level: max/mean relative error of INT8 vs FP16 output
- End-to-end: WikiText-2 perplexity + GSM8K subset (200 problems) across FP16/FP8/INT8-per-token/INT8-per-tensor

## GQA Support

All kernels must support Grouped Query Attention (GQA) — `num_q_heads` may differ from `num_kv_heads`. Test with varying GQA ratios.

## Baselines for Comparison

- `vllm._C.paged_attention_v1` / `paged_attention_v2` (import directly)
- PyTorch SDPA (`torch.nn.functional.scaled_dot_product_attention`)
- PyTorch reference implementation (the ground truth for correctness)

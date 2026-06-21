# RBBQ Pivot — Phase 9 Report: Fused W8A8 kernel + honest latency

**Branch:** `feat/rbbq-proposal`  •  **Date:** 2026-06-21  •  **Status:** complete
**Result:** Built and validated fused Triton W8A8 quant/dequant kernels. Two honest
findings: (1) per-token dynamic is **free when fused** (≈ static), so selective gives
**no** latency advantage over all-dynamic; (2) `torch._int_mm` is **6–9× slower than
bf16** on this H200 — so W8A8 via the readily-available int8 GEMM is a latency *loss*,
not a win. The W8A8 latency lever is a tuned Hopper int8 kernel (this repo's Track A),
not quantization scheme.

## What was built

`rbbq/phase9_fused_w8a8.py` — Triton kernels:
- fused **per-token dynamic quant** (rowmax reduce + quantize in one kernel),
- fused **static quant** (scalar scale, no reduce),
- fused **scaled dequant** (int32 × act_scale[m] × weight_scale[n] → bf16).

Around `torch._int_mm`. **Correctness** vs eager fake-quant: dynamic quant matches to
0.01% of values (sub-LSB rounding), fused linear rel-err 6e-4 (bf16-level), static path
exact. This fixes Phase 7's measurement (eager unfused quant/dequant overhead).

## Benchmark (Qwen2.5-1.5B shapes, per decoder layer = 7 linears, µs)

| | T=2048 | ×bf16 | T=128 | ×bf16 |
|---|---|---|---|---|
| bf16 | 363 | 1.0 | 59 | 1.0 |
| eager int8 all-dynamic (Phase 7 style) | 3974 | 11.0 | 1318 | 22.3 |
| **fused** all-static | 2869 | 7.9 | 1005 | 17.0 |
| **fused** selective (down_proj dyn) | 2883 | 7.9 | 1005 | 17.0 |
| **fused** all-dynamic | 2877 | 7.9 | 994 | 16.8 |

Isolated GEMM (T=2048): bf16 vs `torch._int_mm` —
`down_proj` 96 → 858 µs (**8.9×**), `gate` 112 → 673 (**6.0×**), `qkv` 17 → 149 (**8.7×**).
Fused quant ≤ 29 µs, fused dequant ≤ 43 µs (negligible).

## Findings

1. **Fusion works; quant overhead is negligible.** Fused quant/dequant are tens of µs
   vs hundreds for the GEMM. Phase 7's "int8 4× slower" was indeed an eager-fusion
   artifact for the elementwise ops — but fusing them does **not** rescue int8 here,
   because…

2. **The int8 GEMM is the bottleneck: `torch._int_mm` is 6–9× slower than bf16 on
   H200.** PyTorch's stock int8 GEMM does not use Hopper int8 tensor cores efficiently;
   bf16 (with H200's huge bf16 throughput) wins easily. So **W8A8 via available int8
   GEMM is a compute-latency loss**, regardless of quant scheme.

3. **Per-token dynamic is free when fused.** fused_static ≈ fused_selective ≈
   fused_all_dynamic (2869/2883/2877 µs). The rowmax reduction is fully hidden. So
   **selective-granularity provides no latency advantage over all-dynamic** — confirming
   and sharpening Phase 7. Its "fewer dynamic linears" argument is moot for speed.

## Consequence for the method (final, honest position)

Selective-granularity W8A8 is **not a speed optimization**. Its value is:
- **near-FP W8A8 accuracy** (Phases 5–6) at **half the weight/activation memory** — the
  real, hardware-agnostic win, and the one that matters most for memory-bound decode;
- **static-pipeline simplicity**: ~84% of Qwen linears stay fully-static per-tensor
  (no per-token scale machinery), useful for integer-only / accelerator backends.

It is *not* a latency play, because (a) per-token dynamic is free when fused, and
(b) realizing any W8A8 compute speedup on Hopper requires a properly-tuned int8 GEMM —
`torch._int_mm` does not deliver it. **This is exactly where the project rejoins this
repo's Track A (custom CUDA INT8 kernels):** a Hopper-tuned CUTLASS/cuBLASLt int8 GEMM
(or an FP8 path via `_scaled_mm`) is the prerequisite for a W8A8 latency win.

## Next
- Hopper-tuned int8 GEMM (CUTLASS sm_90 / cuBLASLt) to replace `_int_mm`, then re-bench;
  compare to FP8 `_scaled_mm`. This is the real Track-A kernel task.
- Memory-footprint benchmark (the actual W8A8 win): peak bytes, decode throughput.
- Still open: ViT-B/Llama generality; full GLUE/SQuAD/MMLU/GSM8K vs SmoothQuant/OS.

## Reproduce
```bash
conda activate specdec && cd /home/runying2/cuda-PA-INT8
PYTHONPATH=rbbq python rbbq/phase9_bench.py
```

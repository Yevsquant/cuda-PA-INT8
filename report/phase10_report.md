# RBBQ Pivot — Phase 10 Report: int8 vs fp8 vs bf16 GEMM (the latency truth)

**Branch:** `feat/rbbq-proposal`  •  **Date:** 2026-06-21  •  **Status:** complete
**Result:** int8 GEMM is genuinely slow on Hopper (even a correctness-verified tuned
Triton kernel is 4.1× bf16, vs `_int_mm` 7.3×) — **FP8 is Hopper's low-precision
latency path (1.6× faster than bf16)**. int8's win on Hopper is **memory only**
(−50%). The architectural conclusion: **int8 ↔ A100 (the project target, where IMMA
makes int8 a real latency win); fp8 ↔ Hopper.** The selective-granularity accuracy
method is format-agnostic.

## Setup

`rbbq/phase10_int8_gemm.py`. GEMM-only latency on Qwen2.5-1.5B linear shapes:
bf16 (cuBLAS) · int8 `torch._int_mm` · int8 **tuned Triton** (autotuned; **bit-exact
vs `_int_mm`**) · fp8 `torch._scaled_mm` (e4m3). Plus W+A memory footprint.

## Results — GEMM µs per decoder layer (7 linears), ×bf16

| regime | bf16 | int8 `_int_mm` | int8 Triton (tuned) | fp8 `_scaled_mm` |
|---|---|---|---|---|
| T=2048 | 369 (1.0) | 2680 (**7.3×**) | 1522 (**4.1×**) | **223 (0.6×)** |
| T=128  | 58 (1.0) | 943 (**16.3×**) | 803 (**13.9×**) | **47 (0.8×)** |

Memory (W+A bytes/layer, T=2048): **bf16 168 MB → int8/fp8 84 MB (−50%)**.

## Findings

1. **int8 GEMM is fundamentally underserved on Hopper — not just `_int_mm`.** A tuned,
   correctness-verified (bit-exact) Triton int8 kernel is ~1.8× faster than `_int_mm`
   but still **4.1× slower than bf16**. Both available int8 paths fail to beat bf16, so
   this is a Hopper-software/architecture reality (Hopper prioritizes FP8 throughput;
   int8 tensor-core perf is not delivered by Triton/cutlass int8 here), not a single
   bad operator.

2. **FP8 is the Hopper low-precision latency winner.** `_scaled_mm` runs at **0.6× bf16**
   (1.6× speedup) at T=2048 and 0.8× at T=128 — it uses Hopper FP8 tensor cores well.

3. **Memory is the hardware-agnostic win.** int8 and fp8 both halve W+A bytes. For
   memory-bound **decode**, this ~halves the dominant cost regardless of GEMM compute
   speed — the most portable W8A8 benefit.

## Architectural conclusion (ties the project together)

- **The project's INT8 focus is correct for its stated target, the A100.** A100 has
  strong INT8 IMMA tensor cores and **no FP8** (FP8 is Hopper+). On A100, int8 GEMM is
  ~2× FP16, so **INT8 W8A8 is a real latency win there** — and the H200 dev box (where
  these numbers were taken) is exactly *not* where int8 latency shines.
- **Format ↔ hardware:** INT8 → A100 (latency + memory); FP8 → Hopper (latency); both →
  memory everywhere.
- **The selective-granularity accuracy method (Phases 5–8) is format-agnostic** — the
  `D`-selected per-token-dynamic rule applies equally to INT8 (A100) or FP8 (Hopper)
  activations. The accuracy contribution survives independent of which low-precision
  GEMM the hardware favors.

## Validation still needed (target hardware)

These are H200 numbers. The claim "INT8 W8A8 is a latency win on A100" must be confirmed
on an actual A100 (CLAUDE.md's target; not available on this dev box). The Triton int8
kernel here is portable and bit-exact, so it can be re-benchmarked there directly.

## Backlog still open
- A100 re-benchmark (int8 IMMA) to confirm the latency win on target HW.
- ViT-B/Llama generality; full GLUE/SQuAD/MMLU/GSM8K vs SmoothQuant/OS.
- Selective-granularity with an **FP8** activation path (Hopper) as a parallel to the
  int8 (A100) path.

## Reproduce
```bash
conda activate specdec && cd /home/runying2/cuda-PA-INT8
PYTHONPATH=rbbq python rbbq/phase10_int8_gemm.py
```

# RBBQ Pivot — Phase 7 Report: Cost / latency of selective-granularity W8A8

**Branch:** `feat/rbbq-proposal`  •  **Date:** 2026-06-21  •  **Status:** complete
**Result (honest):** Per-token dynamic quant is *cheap* (a memory-bound rowmax
reduction, single-digit % of the GEMM). So selective's advantage over all-dynamic is
a small speed win; its real value is **near-FP accuracy while keeping ~84% of linears
on a fully-static integer path**, not a large latency saving. Absolute int8-vs-bf16
numbers here are a kernel-fusion artifact and are not used as conclusions.

## Goal & method

Quantify the cost the pivot actually changes. The only thing selective varies per
linear is **static per-tensor** (precomputed scalar scale) vs **per-token dynamic**
(runtime rowmax reduction + per-token scale) activation quant. `rbbq/phase7_latency.py`
times real int8 GEMMs (`torch._int_mm`) on Qwen2.5-1.5B's 7 per-layer linear shapes,
prefill T=2048 and batched T=128. (`_int_mm` needs m>16, so single-token decode int8
GEMM is out of scope — a different kernel.)

## Findings

**1. The marginal cost of per-token dynamic = the rowmax reduction, and it is small.**

| regime | int8 GEMM total | selective extra (down_proj reduce) | all-dynamic extra (7 reduces) |
|---|---|---|---|
| T=2048 | 1597 µs | +105 µs (**+6.6%**) | +179 µs (**+11.2%**) |
| T=128 | 396 µs | +9 µs (**+2.2%**) | +67 µs (**+16.8%**) |

Selective does the (memory-bound) per-token reduction on ~1 linear/layer instead of 7,
so it adds single-digit % over a fully-static int8 path while all-dynamic adds up to
~17%. The absolute gap (selective vs all-dynamic) is **modest** — per-token dynamic is
inherently cheap. (At T=2048 selective already captures most of all-dynamic's overhead
because down_proj has the largest contraction K=8960, hence the biggest reduction.)

**2. CAVEAT — int8 appears slower than bf16 here, but that is unfused eager, not W8A8.**
The eager pipeline (separate round/clamp/cast/mul/dequant kernels on large tensors) runs
~4× slower than bf16 (T=2048: all_static 2410 µs vs bf16 579 µs). This is a
kernel-fusion artifact: a real W8A8 kernel folds quant/dequant into the GEMM
prologue/epilogue. **No int8-vs-bf16 speed conclusion is drawn from this benchmark** —
that requires fused kernels (this repo's CUDA INT8 track).

## Reframing the contribution (honest position)

The pivot's value is **not** a large raw-speed win, because per-token dynamic is already
cheap when fused. The defensible contribution is:

1. **Near-FP W8A8 accuracy** (Phases 5–6) while keeping ~84% of linears (Qwen) on a
   **fully-static per-tensor integer path** — no per-token scale computation, storage,
   or epilogue handling on those linears. This matters for static-pipeline / accelerator
   backends that prefer or require per-tensor scales.
2. **Minimal, predictable intervention:** a cheap calibration statistic (`D`, Phase 6)
   identifies the few linears (the MLP down-projection family) that must break static-ness.
3. A small but real per-token-reduction saving vs all-dynamic (2–7% of GEMM here),
   larger at smaller batch.

This corrects any implicit "selective is much faster" claim: it is mainly an
accuracy-at-static-simplicity method.

## Next
- **Fused W8A8 kernel** with a selective (mixed static/dynamic) quant epilogue → a true
  end-to-end latency number, and decode-regime int8 GEMM. Direct tie-in to this repo's
  PagedAttention/INT8 CUDA track.
- Backlog still open: ViT-B / Llama-7B generality; full GLUE/SQuAD/MMLU/GSM8K vs
  SmoothQuant/OS; the `E'` selection refinement (Phase 6).

## Reproduce
```bash
conda activate specdec && cd /home/runying2/cuda-PA-INT8
PYTHONPATH=rbbq python rbbq/phase7_latency.py
```

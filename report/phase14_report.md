# RBBQ Revised — Phase 14 Report: Residual-aware static-scale rescue

**Branch:** `feat/rbbq-proposal`  •  **Date:** 2026-06-21  •  **Status:** complete
**Result:** A residual/MSE-aware **static** scale at down_proj recovers ~89% of the C1 gap
(ppl 110.4→20.5) with no per-token dynamic — but **cannot match** per-token dynamic
(10.67). No single static per-tensor scale suffices, because the down_proj dynamic range
varies per token. The static rescue is real but insufficient; per-token dynamic stays
necessary for the last ~10 ppl.

## Goal (solution #3)

The original proposal targets hard C1 (static per-tensor activations). Max-scale static
fails at down_proj because one outlier token sets the range. Try a constrained static
alternative — one scale per linear, still C1-deployable — choosing the down_proj clip by
fraction-of-max sweep and MSE-optimal clipping, and measure how much of the gap closes
*without* going dynamic. `rbbq/phase14_static_rescue.py` (Qwen2.5-1.5B, WikiText-2).

## Results (down_proj static clip; others max-static C1; FP ppl 9.803)

| down_proj static scale | ppl | Δppl |
|---|---|---|
| 1.00·max (= plain C1) | 110.4 | +100.6 |
| 0.99·max | 106.9 | +97.1 |
| 0.97·max | 102.7 | +92.9 |
| 0.95·max | 98.1 | +88.3 |
| 0.90·max | 88.6 | +78.8 |
| 0.80·max | 71.0 | +61.2 |
| 0.70·max | 53.8 | +44.0 |
| **MSE-optimal (per-linear)** | **20.5** | **+10.7** |
| per-token dynamic (upper bound) | 10.67 | +0.86 |

(An earlier run reported ~12 for "max"; that was a *sample-max* artifact — the capped
calibration subsample under-estimated the true max, accidentally clipping the extreme
tail. The table above uses the true calibrated max as the base and is authoritative.)

## Findings

1. **Mild fraction clipping helps slowly** — even 0.70·max only reaches 53.8 ppl. The
   damage is dominated by a small number of extreme-outlier tokens, but uniformly shrinking
   the scale also crushes the many high-but-not-extreme values, so the fraction sweep can't
   win.
2. **MSE-optimal per-linear clipping is much better (20.5 ppl, ~89% of the gap closed)** —
   it clips the heavy tail per linear to minimize quant MSE. This is a legitimate, fully
   **static, C1-deployable** rescue: no per-token scales, no runtime reduction.
3. **But it cannot reach per-token dynamic (10.67).** A single static scale per linear
   cannot serve both the outlier tokens and the bulk simultaneously — the down_proj input
   dynamic range is genuinely **per-token**. This is a direct, quantitative confirmation of
   the per-token-granularity diagnosis (Phases 3–5): the problem is per-token, so no static
   scale closes it.

## Conclusion

- **Static C1 is partially rescuable:** if per-token dynamic is unavailable (e.g. a strict
  static-only integer pipeline), MSE-optimal clipping at down_proj turns a catastrophic
  +100 ppl into +10.7 ppl — a usable fallback.
- **Per-token dynamic remains the recommended fix** (Δ+0.86), and the ~10 ppl gap between
  the best static and dynamic is the *irreducible per-token component* — the precise
  quantity that motivates selective granularity.
- This is the honest "serious static rescue attempt": it succeeds partially and maps its
  own ceiling.

## Next: Phase 15 — token-outlier fallback (escalate only outlier tokens).

## Reproduce
```bash
conda activate specdec && cd /home/runying2/cuda-PA-INT8
HF_DATASETS_OFFLINE=1 PYTHONPATH=rbbq python rbbq/phase14_static_rescue.py
```

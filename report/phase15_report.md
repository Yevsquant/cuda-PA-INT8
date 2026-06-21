# RBBQ Revised — Phase 15 Report: Token-outlier fallback at down_proj

**Branch:** `feat/rbbq-proposal`  •  **Date:** 2026-06-21  •  **Status:** complete
**Result:** Routing only the **outlier tokens** (≈0.6%) through FP16 at down_proj brings
static int8 from 110.4 → **11.64 ppl** — far past the static-clip rescue (20.5, Phase 14)
and approaching per-token dynamic (10.67), while 99.4% of tokens stay static int8. The
W8A8 damage is concentrated in a small fraction of tokens.

## Goal (solution #4)

A C1↔all-dynamic middle ground: static int8 activations at down_proj for normal tokens;
for tokens with per-token `rowmax > k·median_rowmax`, route that token's activation through
FP16 (int8 weight kept). The static scale is `thr/127` (= `k·median/127`) so non-escalated
tokens fit exactly. Sweep `k`. `rbbq/phase15_token_fallback.py` (Qwen2.5-1.5B, WikiText-2).

## Results (FP ppl 9.803; down_proj hybrid, others static C1)

| k | escalated-token fraction | ppl | Δppl |
|---|---|---|---|
| 16 | 0.02% | 50.2 | +40.4 |
| 8 | 0.02% | 14.7 | +4.9 |
| 4 | **0.6%** | **11.64** | +1.84 |
| 2 | 8.8% | 10.89 | +1.08 |
| per-token dynamic (ref) | — | 10.67 | +0.86 |
| MSE-static (Phase 14) | 0% | 20.5 | +10.7 |

## Findings

1. **The damage is concentrated in a tiny token-fraction.** Escalating ≈0.6% of tokens to
   FP16 (k=4) recovers to +1.84 ppl — already better than the best pure-static scale
   (Phase 14, +10.7) and within ~1 ppl of full per-token dynamic. At 8.8% (k=2) it reaches
   +1.08, essentially matching dynamic. A handful of outlier tokens cause the W8A8 failure.
2. **`k` couples two effects** (honest caveat): `thr = k·median` sets both the escalation
   threshold *and* the static scale `thr/127`. k=8 vs k=16 escalate the same 0.02% of tokens
   but differ a lot (14.7 vs 50.2) because the larger-k static scale is coarser for the
   bulk. So the knob trades static-granularity-for-the-bulk against escalation-fraction
   jointly; both improve as k drops.
3. **A viable mostly-static deployment.** Static int8 for >99% of tokens + an FP16 escape
   hatch for the ~0.6% outliers needs no per-token scale machinery on the common path — a
   middle ground between strict C1 and all-dynamic, targeting exactly the residual-branch
   output where the failure lives.

## Down_proj static-handling ladder (consolidated)

| method | ppl | tokens needing >static |
|---|---|---|
| max-static (plain) | 110.4 | 0 |
| MSE-optimal static (Ph14) | 20.5 | 0 |
| token-outlier FP16, 0.6% (Ph15) | 11.64 | 0.6% |
| token-outlier FP16, 8.8% (Ph15) | 10.89 | 8.8% |
| per-token dynamic (Ph4/5) | 10.67 | 100% |
| FP | 9.80 | — |

A clean monotone trade: the more per-token adaptivity you allow at down_proj, the closer to
FP — and almost all of it is bought by the first ~1% of tokens.

## Next: Phase 16 — FP8 selective path (hardware-aware), the final component.

## Reproduce
```bash
conda activate specdec && cd /home/runying2/cuda-PA-INT8
HF_DATASETS_OFFLINE=1 PYTHONPATH=rbbq python rbbq/phase15_token_fallback.py
```

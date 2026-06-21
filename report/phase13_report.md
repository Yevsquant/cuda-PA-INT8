# RBBQ Revised — Phase 13 Report: RBBQ-A + Selective ablation

**Branch:** `feat/rbbq-proposal`  •  **Date:** 2026-06-21  •  **Status:** complete
**Result:** Exact `LN→Linear` folding on the foldable linears **adds value on top of
selective** (Qwen ppl 10.67→10.19, closing ~half the residual gap), while folding **alone
is insufficient** (89.2 — it can't reach down_proj). The strongest static-friendly method
is **RBBQ-A+Selective**: exact folds where legal + per-token dynamic at down_proj.

## Goal (solution #2)

Keep Variant A's exact folding as a *component*, not the whole method: fold SmoothQuant/
RBBQ-A scales only on the foldable linears (`q/k/v`, `gate/up`), do **not** attempt a
cross-norm inverse at `down_proj`, and use per-token dynamic there. Ablate. In **pre-norm**
Qwen the RMSNorm output feeds only qkv/gate-up (not the residual), so folding into the
RMSNorm weight is **algebraically exact** — unlike post-norm BERT. `rbbq/phase13_fold_selective.py`.

## Ablation (Qwen2.5-1.5B, WikiText-2, C1 W8A8; FP ppl 9.803)

| arm | folds (qkv/gate-up) | down_proj | ppl | Δppl |
|---|---|---|---|---|
| plain C1 | – | static | 110.4 | +100.6 |
| folds_only | exact SmoothQuant | static | 89.2 | +79.4 |
| selective_only | – | dynamic | 10.67 | +0.86 |
| **folds + selective** | exact SmoothQuant | dynamic | **10.19** | **+0.39** |
| all_dynamic | – | dynamic | 10.02 | +0.22 |

## Findings

1. **Folding alone cannot fix W8A8** (89.2 ppl): the exact `LN→Linear` smoothing improves
   the foldable linears but down_proj stays static and dominates — re-confirming Phases 2–4
   from the other direction (the foldable boundary is the wrong place for the *bottleneck*).
2. **Folding adds value on top of selective** (0.86 → 0.39 ppl gap, ~55% of the residual
   closed). Once down_proj is handled by per-token dynamic, the remaining error lives in the
   static-quantized qkv/gate-up inputs, which exact folding reduces — getting within +0.17
   ppl of all-dynamic while keeping 6/7 linear types static.
3. **The combination is principled and disciplined.** RBBQ-A+Selective uses exact algebra
   only where legal (pre-norm RMSNorm fold, verified exact by construction) and per-token
   dynamic only at the proven causal site — honoring the original proposal's fold discipline
   without any norm-crossing claim.

## Method statement (RBBQ-A+Selective)

> Static per-tensor W8A8 by default. Apply exact SmoothQuant/RBBQ-A folding on the foldable
> `LN→Linear` boundaries (qkv, gate/up); apply per-token dynamic activation quant at the
> `mlp_out` (down_proj/fc2) family. No cross-norm inverse. Result: ≈ all-dynamic accuracy
> while the majority of linears keep static per-tensor activations.

This is the recommended deployable variant: it strictly dominates selective-alone and folds-
alone, and is the honest synthesis of Variant A (exact, Phase 2) and selective granularity
(Phases 5–12).

## Caveat

Folding is exact only at pre-norm `LN→Linear` (Qwen/Llama/Mistral, ViT). In **post-norm
BERT** the LN output is the residual stream, so these folds are *not* free (Phase 0/3) —
there, selective-alone (no folding) is the safe variant. So RBBQ-A+Selective's folding
component applies to pre-norm architectures.

## Next: Phase 14 — residual-aware static-scale search (the "static rescue").

## Reproduce
```bash
conda activate specdec && cd /home/runying2/cuda-PA-INT8
HF_DATASETS_OFFLINE=1 PYTHONPATH=rbbq python rbbq/phase13_fold_selective.py
```

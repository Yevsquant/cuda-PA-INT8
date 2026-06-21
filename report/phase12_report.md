# RBBQ Revised — Phase 12 Report: RBBQ-C consolidation + residual→D linkage

**Branch:** `feat/rbbq-proposal`  •  **Date:** 2026-06-21  •  **Status:** complete
**Result:** The locked group-aware RBBQ-C policy works (Qwen ppl 110.4→10.49, 33/196
dynamic). The residual→selector linkage is **honest and precise**: channel-domain
residual diagnostics (output dominance) *mislead* (they flag `o_proj`), while the
actual damage and the per-token input spread `D` both flag `down_proj`. So the residual
premise localizes the failing *region* (MLP-add branch output) and justifies the
group-aware prior; **`D` is the operative per-linear selector.**

## Goal (solution #1 finish)

Make the original proposal's residual instrumentation and the new `D` selector cohere
into one canonical method (RBBQ-C), and test whether the residual diagnostic predicts the
same linear as `D`. `rbbq/phase12_rbbqc.py` (Qwen2.5-1.5B, WikiText-2, C1 W8A8).

## Linkage analysis (per-linear, by type)

| type | residual diagnostic: output channel dominance (median) | selector: `D` (median) |
|---|---|---|
| attn_in (q/k/v) | 91 | 2.18 |
| attn_out (o_proj) | **405** | 2.33 |
| mlp_in (gate/up) | 30 | 2.72 |
| **mlp_out (down_proj)** | 118 | **7.69** |

**They disagree, and `D` is right.** Output channel-dominance ranks `o_proj` highest, but
Phase 4 showed keeping `o_proj` FP16 barely helps while `down_proj` is the causal site —
and `D` flags `down_proj` cleanly (7.69 vs ≤2.7). This is the same lesson as Phase 6:
channel/energy-domain diagnostics conflate "concentrates energy" with "breaks W8A8"; the
per-token **input** spread `D` (static-vs-dynamic scale mismatch) is the operative signal.

**Honest linkage statement:** the residual-stream premise (Bondarenko/OS: failure at the
residual-add branch outputs) survives and **motivates the group-aware prior** — always
consider the `mlp_out` family, the repo-proven causal site across BERT/Qwen/Mistral. But
the residual *channel* diagnostics do not rank linears correctly; **`D` selects.** The two
are complementary: residual premise → region/group; `D` → linear.

## RBBQ-C locked policy

**Policy:** per-token dynamic activation quant iff `group == mlp_out` OR `D ≥ thr`
(thr=5); static per-tensor elsewhere. Weights per-channel on dynamic linears, per-tensor
on static.

| Qwen2.5-1.5B (FP ppl 9.803) | ppl | dynamic linears |
|---|---|---|
| plain C1 (all static) | 110.4 | 0 |
| **RBBQ-C** | **10.49** | 33/196 (17%) |
| all-dynamic | 10.02 | 196/196 |

RBBQ-C recovers to within +0.47 ppl of all-dynamic and +0.69 of FP, at 17% dynamic — and
the policy is now a single locked rule, not a swept threshold.

## Three-family status (canonical method)

RBBQ-C is the headline method; its accuracy is established across families in prior phases
(re-stated here as the method's results, not scattered diagnostics):
- BERT/MNLI: plain 9.01 pp gap → selective 0.22 pp (Phase 5).
- Qwen/WikiText-2: +100.6 → +0.69 ppl (this phase, locked policy).
- Mistral-7B/WikiText-2: +139.9 → +0.22 ppl (Phase 11).

## Refinement to the revised claim

> The residual-stream **diagnosis** localizes W8A8 failure to the MLP residual-add output
> (`fc2`/`down_proj`) and motivates a **group-aware prior**; the per-token input spread
> `D` is the operative per-linear **selector**. Channel-domain residual metrics (`r[c]`,
> output dominance) are diagnostic of the region but **not** valid linear selectors.

## Next: Phase 13 — RBBQ-A + selective ablation (exact folds where legal).

## Reproduce
```bash
conda activate specdec && cd /home/runying2/cuda-PA-INT8
HF_DATASETS_OFFLINE=1 PYTHONPATH=rbbq python rbbq/phase12_rbbqc.py
```

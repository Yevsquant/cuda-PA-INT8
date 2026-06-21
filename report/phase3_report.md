# RBBQ Phase 3 Report — Variant B refuted; constructive granularity fix

**Branch:** `feat/rbbq-proposal`  •  **Date:** 2026-06-21  •  **Status:** complete
**Verdict:** The RBBQ branch-rebalancing thesis does **not** address the dominant
W8A8 damage on BERT. Variant B (weight-side) cannot help. The actual bottleneck —
static per-tensor quantization of two activation inputs — is fully fixed by a
cheap selective per-token-dynamic scheme.

## Goal and method

Phase 2 localized ~96% of the C1 (static per-tensor) W8A8 damage to out_proj/fc2
(the non-foldable MLP-add region). Variant B is a **weight-side** method (fold a
per-output-channel scale into W_down/out_proj, calibrated cross-norm inverse). So
before building it, a decomposition diagnostic (`rbbq/phase3_variant_b.py`,
`FlexQuantLinear` with per-linear control of weight/act quant) measured **what
Variant B could possibly recover.**

## Diagnostic — out_proj/fc2 treatment, MNLI (FP = 0.8457, 9815 ex)

| arm | out_proj/fc2 treatment | acc | gap |
|---|---|---|---|
| plain | all C1 quant | 0.7556 | 9.01 |
| diag_a | quant **weight** (per-tensor), FP16 act | 0.8397 | **0.60** |
| diag_w | FP16 weight, quant **act** (static) | 0.7620 | **8.37** |
| diag_wc | per-**channel** weight, quant act | 0.7624 | 8.33 |
| **fix_dyn** | per-token **dynamic** act + per-ch weight | 0.8421 | **0.36** |
| oracle | both FP16 | 0.8421 | 0.36 |

## Findings

1. **The damage is activation-quant, not weight-quant.** Quantizing the *weights*
   of out_proj/fc2 costs only 0.60 pp (`diag_a`); quantizing their *activation
   inputs* statically costs 8.37 pp (`diag_w`). Per-channel weights change nothing
   (`diag_wc` 8.33). The killer is **static per-tensor quantization of the inputs
   to out_proj (attention context) and fc2 (GELU output)** — outlier-heavy tensors
   downstream of softmax/GELU (non-linear, non-foldable).

2. **It is a per-token granularity problem, fully fixable cheaply.** Switching
   *only* out_proj and fc2 (2 of 6 linears/layer) to per-token **dynamic**
   activation quant (`fix_dyn`) reaches **0.36 pp — identical to the oracle**. No
   folding, no inverse, no branch balancing required.

3. **Variant B cannot help, and the branch-rebalancing thesis misses the target.**
   Variant B rebalances the update-branch *output projection rows* (a weight-side
   transform). But `diag_a` shows the weight side is already near-lossless. The
   bottleneck is the *activation inputs* of those linears, which branch
   rebalancing at the residual add does not touch. The Phase-1 channel-dominance
   of the update branch was real, but it is carried by **outlier activations**
   (per-token dynamic range), not by weight rows or by a fixable branch energy
   imbalance.

## Decision: Variant B not implemented

Implementing the full Variant B (γ-fold inverse + `ε_inv`) would be building
toward a hypothesis the diagnostic already refutes (the weight side it targets is
near-lossless). That contradicts the evidence and the goal, so it was not built.
The reusable `FlexQuantLinear` (with the `out_fold` hook) remains in the script if
a future, activation-side variant wants it.

## Recommendation for the project

- **On BERT, RBBQ as proposed is refuted.** W8A8 is solved by **selective
  per-token-dynamic activation quant on the two update-branch output linears**
  (out_proj, fc2) — a "selective granularity" method (mild runtime cost, no graph
  transform). This is the honest headline result and a clean, defensible
  contribution in its own right, distinct from the original proposal.
- The original proposal's baselines (SmoothQuant, OS, PEG) all target *foldable*
  activation outliers; this result shows the residual W8A8 gap on BERT lives in
  *non-foldable* post-nonlinearity activations that those methods structurally
  cannot reach but selective per-token quant trivially does.

## Open question that could still rescue the branch thesis (Phase 4 gate)

The BERT diagnostic does **not** settle decoder LLMs. Phase 1 showed Qwen-2.5's
**identity (residual-stream) branch** carries massive-activation channels
(dom_id ~135 000×) feeding the **LN→qkv** path — which *is* foldable and *is*
branch-relevant, unlike BERT's post-GELU/softmax inputs. Before abandoning RBBQ
entirely, run the same weight/act decomposition on Qwen W8A8:
- if the LLM damage is **weight-side / residual-stream** and foldable → the
  branch-balancing idea may still have a target there;
- if it is again **activation-granularity** → RBBQ is refuted across families and
  the project should pivot fully to selective per-token quant.

## Reproduce
```bash
conda activate specdec && cd /home/runying2/cuda-PA-INT8
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
  PYTHONPATH=rbbq python rbbq/phase3_variant_b.py --task mnli
```
(Use offline flags — datasets 5.0 re-checks the hub tree and intermittently 504s.)

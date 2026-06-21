# RBBQ Phase 1 Report — Residual-Branch Energy Instrumentation (GATE)

**Branch:** `feat/rbbq-proposal`  •  **Date:** 2026-06-21  •  **Status:** complete, **GATE PASSED**

Phase 1 goal: at every residual add `y = identity + update`, measure per-channel
branch energy and the imbalance `r[c] = E_id[c]/E_up[c]`. Gate question: **is
`r[c]` heavy-tailed?** If flat, the RBBQ premise fails. It is decisively *not*
flat.

## Setup

- `rbbq/phase1_branch_energy.py`. Two architectures spanning both folding regimes:
  - **BERT-base** (post-norm LayerNorm), 12 layers → 24 adds; 512 SST-2 sentences.
  - **Qwen2.5-1.5B-Instruct** (pre-norm RMSNorm, Llama-family proxy), 28 layers →
    56 adds; 128×256-token WikiText-2 chunks; bf16 (fp16 → NaN, per repo memory).
- Per add: `r[c]` summary (`frac_imbalanced` = fraction with `|log2 r|>2`,
  `spread_log2r`), **channel dominance** `max_c E[c]/median_c E[c]` per branch,
  **crush rate** (fraction of add-output values quantizing to `|level|<=1` under
  per-tensor int8 max-scaling).
- Figures: `report/figs/phase1_bert.png`, `report/figs/phase1_qwen.png`.
  Raw stats: `report/phase1_results.json`.

## Results — aggregate

| Model | frac_imbalanced | spread_log2r | mean dom_id | max dom_id | mean crush |
|---|---|---|---|---|---|
| BERT-base (post-norm) | 0.54 | 0.84 | 634 | 1796 | 0.51 |
| Qwen2.5-1.5B (pre-norm) | 0.72 | 2.15 | 132 483 | 207 847 | 0.94 |

Both blow past the gate: a large fraction of channels are >2 octaves imbalanced,
channel dominance is 2–5 orders of magnitude above the median, and 50–94% of all
residual values are crushed into the lowest int8 levels.

## Results — where the imbalance concentrates (the key finding)

Splitting by add type (attention residual add vs MLP residual add):

| | frac_imb | crush | dom_id | dom_up |
|---|---|---|---|---|
| **BERT** attn-add | 0.73 | 0.21 | 108 | 12 |
| **BERT** mlp-add  | 0.36 | **0.82** | 1161 | **37 805** |
| **Qwen** attn-add | 0.93 | **0.93** | **135 481** | 400 |
| **Qwen** mlp-add  | 0.51 | 0.94 | 129 484 | 9 121 |

- **BERT: the MLP residual add is where it breaks.** The MLP **update** branch
  injects a few channels with energy ~38 000× the median (`dom_up`, peaking at
  369 239 in `L10_add1`), driving an 82% crush rate. The attention add is
  comparatively benign. This **directly explains Phase 0 finding 2**: SmoothQuant
  smooths qkv/fc1 inputs and never touches the `out_proj`/`fc2` region, so it
  could not recover the static-W8A8 drop — the damage is at the MLP add it
  doesn't see.
- **Qwen: the identity (residual-stream) branch carries massive activations.**
  `dom_id` ~135 000× median at the attention add — the known Qwen "massive
  activation" channels living in the residual stream — with 93% crush. Pre-norm
  RMSNorm is *worse* than post-norm BERT here, which is exactly the headline
  Llama-family target.

So the imbalance is real, large, and **branch- and add-localized**, not diffuse —
precisely the structure RBBQ exploits.

## Gate verdict — PASSED

`r[c]` is strongly heavy-tailed in both architectures; channel dominance and
crush rates confirm the Bondarenko / Outlier-Suppression residual-outlier picture
at the level of individual residual adds. Greenlight Phase 2.

## Implications for Phase 2 (RBBQ-A)

1. **Prioritize the MLP residual add** (BERT) and **both adds** (Qwen). The
   branch-energy criterion should protect the channels the update branch injects
   into the stream, which magnitude-only SmoothQuant under-weights.
2. The most-damaged linears (`out_proj`, `fc2`) are *not* LN-fed, so for an exact
   fold RBBQ-A must either (a) smooth them with an explicit scale (exact, small
   runtime op, as in the Phase 0 harness) or (b) restrict folding to the LN-fed
   linears and measure how much of the add imbalance that captures. This tension
   is the core Phase 2 design question and ties back to `rbbq_method.md`'s
   foldable-boundary analysis.
3. Add a harder GLUE task (CoLA/MNLI) in Phase 2 — SST-2 under-resolves (Phase 0).

## How to reproduce

```bash
conda activate specdec
cd /home/runying2/cuda-PA-INT8
python rbbq/phase1_branch_energy.py     # BERT ~secs, Qwen ~1-2 min on H200
```

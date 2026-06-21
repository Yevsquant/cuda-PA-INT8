# RBBQ Phase 2 Report — Variant A (exact foldable boundary)

**Branch:** `feat/rbbq-proposal`  •  **Date:** 2026-06-21  •  **Status:** complete
**Verdict:** RBBQ-A (exact, foldable) is the *wrong lever*; redirect to Variant B (Phase 3).
The RBBQ premise is **confirmed**, but the fixable damage is not at the foldable boundary.

## Goal

Per the plan: implement Variant A — branch-energy criterion at the exact foldable
`LN→Linear` boundary (qkv, fc1) — and test whether it beats SmoothQuant at W8A8
**C1** (static per-tensor), on SST-2 and a harder task (MNLI). Plus a
mixed-precision oracle (`out_proj`/`fc2` kept FP16) to localize the damage.

`rbbq/phase2_rbbq_a.py` (reuses Phase 0 `QuantLinear`/calibration and Phase 1
branch-energy hooks). RBBQ-A scale:
`s_c = a_c^0.5 / w_c^0.5 · clip(r[c],0.1,10)^λ`, with `r[c]=E_id/E_up` from the
residual add that produces this linear's input stream (`qkv_i ← mlp-add_{i-1}`,
`fc1_i ← attn-add_i`).

## Results — C1 static per-tensor W8A8

| | SST-2 acc | SST-2 gap | **MNLI acc** | **MNLI gap** |
|---|---|---|---|---|
| FP reference | 0.9243 | — | 0.8457 | — |
| plain W8A8 | 0.9128 | 1.15 | 0.7556 | **9.01** |
| SmoothQuant (qkv/fc1) | 0.9117 | 1.26 | 0.7789 | 6.68 |
| RBBQ-A λ=0.5 (qkv/fc1) | 0.9071 | 1.72 | 0.7657 | 8.00 |
| **oracle_mp** (out_proj/fc2 FP16) | 0.9174 | 0.69 | **0.8421** | **0.36** |

RBBQ-A λ sweep on MNLI (gap, pp): `−0.5→6.72, −0.25→6.20, 0.0→6.68 (=SmoothQuant),
+0.25→6.30, +0.5→8.00`.

**SST-2 is within noise** (872 val ex; 1 ex ≈ 0.11pp) and cannot rank these
methods — MNLI (9815 ex) is the reliable signal. All conclusions below are MNLI.

## What the numbers say

1. **96% of the W8A8 damage is at `out_proj`/`fc2`** (the MLP-residual-add region,
   non-foldable in post-norm BERT). The oracle takes the gap from **9.01 → 0.36
   pp** by keeping just those two linears per layer in FP16. This confirms Phase 1
   (the MLP update branch injects ~38 000× channels) at the task-accuracy level.

2. **SmoothQuant works but is structurally capped.** It recovers 2.3 pp
   (9.01→6.68) by smoothing qkv/fc1, but cannot touch `out_proj`/`fc2`, so it
   leaves ~6.3 pp (of the 6.68) on the table — exactly the oracle headroom.

3. **The branch-energy criterion is a weak lever at the foldable boundary.** Best
   λ (−0.25) beats SmoothQuant by only ~0.5 pp, and there is **no robust
   direction** (both small ±λ help marginally; λ=0 reproduces SmoothQuant
   exactly, a clean sanity check). This matches theory: the linear's input is
   `LN(stream)`, whose channel structure is set by the LayerNorm `γ`, not by the
   raw-stream branch energy `r[c]` measured at the add. The branch decomposition
   is meaningful *at the add*, not *post-LN* — so Variant A cannot use it well.

## Conclusion → redirect to Variant B

The RBBQ thesis (residual-branch imbalance causes the PTQ damage) is **confirmed**
by the oracle. But the damage lives at the **MLP residual add**, which in post-norm
BERT is **not** a foldable boundary (Phase 0 finding). Therefore:

- **Exact Variant A at qkv/fc1 is the wrong place** — no criterion there can close
  a 6 pp gap that lives in `out_proj`/`fc2`.
- **The method must act at the add → Variant B (Phase 3):** rebalance the update
  branch by folding a per-channel scale into the update-branch output projection
  (`W_down`/`out_proj`) and applying the calibrated cross-norm inverse
  (`rbbq_method.md` §Variant B), accepting and measuring the inversion error
  `ε_inv`. **Headroom target: the oracle gap of 0.36 pp on MNLI.**

## Hand-off to Phase 3

- Reusable: `phase2_rbbq_a.py` (`quantize`, `build_scales`, `branch_energies`,
  `evaluate` with label perm), `oracle_mp` as the upper-bound reference.
- **Gotcha (saved to memory):** `textattack/bert-base-uncased-MNLI` has no real
  `id2label`; output index → GLUE index permutation is **(2,0,1)** (FP 0.8457).
  Without it FP looks like 6.5%.
- Phase 3 build: (a) fold update-branch scale into `out_proj`/`fc2` weights;
  (b) absorb inverse into the following LayerNorm `γ` (the post-norm wall →
  approximate); (c) report `ε_inv` and MNLI/SST-2 gap vs the 0.36 pp oracle.
  Decision to surface: pre-norm (Qwen/Llama) gives a cleaner inverse path than
  post-norm BERT — consider leading Phase 3 on Qwen.

## Reproduce
```bash
conda activate specdec && cd /home/runying2/cuda-PA-INT8
PYTHONPATH=rbbq python rbbq/phase2_rbbq_a.py --tasks sst2 mnli
```

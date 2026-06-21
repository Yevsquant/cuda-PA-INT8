# RBBQ Revised Plan — Residual-Guided Selective-Granularity W8A8

Supersedes the actuator of `rbbq_proposal.md` while preserving its motivation,
instrumentation, W8A8 scope, architecture comparison, and success criteria. Grounded in
Phases 0–11 (`rbbq_capstone.md`) and the five directions in `pontential_sol.md`.

## Revised claim (the thesis)

> RBBQ's original branch-**rebalancing** actuator is refuted (Phases 2–4), but its
> residual-stream **premise survives**: residual-add imbalance localizes the W8A8 failure
> to the MLP output projection (`fc2`/`down_proj`), and it is **activation-side**. The
> corrected method — **Residual-Guided Selective-Granularity W8A8 (RBBQ-C)** — uses the
> residual diagnostics plus the input per-token spread `D` to select the branch-output
> linears that require per-token-dynamic activation quant, keeps static per-tensor W8A8
> elsewhere, and folds exactly only where algebraically legal.

Do **not** claim diagonal branch rebalancing works. The evidence is the opposite.

## What is preserved vs replaced

| From original proposal | Status |
|---|---|
| Motivation (residual outliers; Bondarenko/OS) | **kept** |
| Residual-add instrumentation `E_id,E_up,r[c]` | **kept** (Phase 1) — now a *diagnosis*, not an actuator input |
| W8A8 scope, C1/C2 configs, success criteria | **kept** |
| Baselines (SmoothQuant/OS/PEG/mixed precision) | **kept** |
| Actuator: folded diagonal branch rebalancing | **replaced** → selective granularity |

## Method: RBBQ-C (core, solution #1 — largely DONE)

1. Static per-tensor W8A8 everywhere by default (the hard C1 setting).
2. Residual diagnosis identifies the poisoning branch output: `r[c]` + channel-dominance
   at each add point to `fc2`/`down_proj` (Phases 1, 3, 4).
3. Selection: per linear input, `D = max_token(rowmax)/median_token(rowmax)`; apply
   per-token dynamic activation quant iff `D ≥ thr`. **Group-aware prior:** consider the
   `mlp_out` (`fc2`/`down_proj`) family first — the repo-proven causal site across BERT,
   Qwen, Mistral.
4. Evidence already in hand: Qwen 32/196 dynamic ≈ all-dynamic (Phase 5); `D` beats
   sensitivity metrics (Phases 6, 8); transfers to Mistral-7B (Phase 11).

**Remaining for RBBQ-C:** make the *residual-diagnostic → selection* link explicit (show
`r[c]`/branch-energy at the add predicts the same linears as `D`), and lock the
group-aware threshold policy. → Phase 12.

## Components / ablations (the NEW work)

### Phase 12 — RBBQ-C consolidation (solution #1 finish)
Assemble the canonical one-knob method; add the explicit residual→`D` linkage; group-aware
threshold; re-report the 3-family table as the headline method (not scattered diagnostics).
*Verify:* residual branch-energy ranking ≈ `D` ranking; selective ≤ 0.3 pp / small-ppl gap
on all three families with the group-aware policy.

### Phase 13 — RBBQ-A + Selective (solution #2)
Exact `LN→Linear` folding (SmoothQuant/RBBQ-A) on the **foldable** linears
(`q/k/v`, `fc1`, `gate/up`) **plus** per-token dynamic at `fc2`/`down_proj`. Do *not* try a
cross-norm inverse at `down_proj`. *Verify (ablation):* exact-folds-alone vs
exact-folds + selective-dynamic vs selective-alone — does legal folding add anything on top
of selective? Reuses Phase 2 (SmoothQuant) + Phase 5 (selective).

### Phase 14 — Residual-aware static-scale search (solution #3, the "static rescue")
Keep C1 deployable (one static scale per group), but choose the `down_proj`/`fc2` activation
scale by minimizing **downstream residual error**, not raw local max. Objectives to sweep:
percentile/MSE clipping, residual cosine after the add, static-vs-dynamic mismatch, tiny-set
PPL. *Verify:* how much of the C1 gap closes with a *static* scale (an honest attempt to
rescue C1 before conceding per-token dynamic).

### Phase 15 — Token-outlier fallback (solution #4)
Hybrid at `fc2`/`down_proj`: static per-tensor for normal tokens; for tokens with
`rowmax > k · median_rowmax`, quantize that token dynamically (or route its activation through
BF16/FP16). *Verify:* accuracy vs fraction-of-tokens-escalated; a middle ground between C1
and all-dynamic that still targets the residual-branch output.

### Phase 16 — Hardware-aware / FP8 path (solution #5, Hopper-tractable now)
Same `D` selection rule, two formats: **INT8→A100**, **FP8→Hopper**. Build the FP8 selective
path on this H200 (`torch._scaled_mm` + the Phase-9 fused kernels); measure accuracy + latency.
*Verify:* FP8 selective matches INT8 selective accuracy and is ≤ bf16 latency on H200 (Phase 10
showed fp8 GEMM 0.6× bf16). Frame selectivity as accuracy/memory, **not** a latency trick
(Phases 7, 9).

## Success criteria (revised, inherited from the proposal)

- **Accuracy:** sub-0.5 pp gap from FP16/BF16 at W8A8 on ≥ 2 families — **already met** by
  selective (BERT/MNLI 9.01→0.36 pp; Qwen +100→+0.86 ppl; Mistral +140→+0.16 ppl).
- **Selectivity:** match all-dynamic while keeping ≥ 80 % of linears static per-tensor
  (Qwen 84 %, Phase 5).
- **Discipline:** every fold algebraically exact (Phase 13); no claim that crosses a norm.
- **Honesty:** latency framed as format×hardware + memory (Phases 7, 9, 10), not a speed win.

## Deferred / blocked (state explicitly)

- **ViT-B (vision):** the one untested architecture class — blocked on ImageNet (gated/
  uncached on this box). Partial signal (`D`-stat + localization on a cached CLIP/timm ViT)
  possible if desired.
- **A100 INT8 latency confirmation:** needs A100 (CLAUDE.md reserves it for final runs); the
  Phase-10/Phase-9 kernels are portable for a one-shot re-benchmark there.
- **Full SmoothQuant/OS/PEG matrix on LLMs:** incremental; core comparisons already made.

## Roadmap summary

| Phase | Solution | New? | Tractable here |
|---|---|---|---|
| 12 RBBQ-C consolidation | #1 | finish | ✅ |
| 13 RBBQ-A + selective ablation | #2 | yes | ✅ |
| 14 residual-aware static rescue | #3 | yes | ✅ |
| 15 token-outlier fallback | #4 | yes | ✅ |
| 16 FP8 selective path | #5 | yes | ✅ (H200) |
| — ViT-B | scope | yes | ❌ ImageNet |
| — A100 int8 latency | #5 | yes | ❌ hardware |

Recommended order: 12 → 13 → 14/15 (the static-rescue and fallback are the freshest research
questions) → 16. One phase per session, report per phase, as established.

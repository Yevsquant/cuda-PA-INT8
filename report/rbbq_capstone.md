# RBBQ Capstone — From Residual-Branch Quantization to Selective-Granularity W8A8

**Branch:** `feat/rbbq-proposal` • consolidates Phases 0–10 • 2026-06-21

This document synthesizes the full investigation: a proposed method (Residual-Balanced
Branch Quantization) was **refuted** with rigorous mechanism, and a constructive
alternative (**selective-granularity W8A8**) with a validated selection rule and an
honest format↔hardware latency story emerged in its place. Per-phase detail is in
`report/phase{0..10}_report.md`.

---

## 1. The original hypothesis (and why it was plausible)

RBBQ proposed that Transformer PTQ breaks at the **residual add**: the identity branch
`x` and update branch `F(x)` have mismatched per-channel energy, so quantizing their sum
crushes the quieter branch. The method: estimate the per-channel imbalance and fold a
pair of branch rescalings that equalize the add, inverting downstream "where
algebraically possible." Grounded in Bondarenko (residual outliers), Outlier Suppression
(LN-γ amplification), and residual-dominance reproductions.

**The branch imbalance is real** (Phase 1 gate **passed**): `r[c]=E_id/E_up` is heavy-
tailed on both BERT (post-norm) and Qwen2.5 (pre-norm RMSNorm); channel dominance reaches
10²–10⁵× the median; int8 "crush rate" 0.51 (BERT) / 0.94 (Qwen); and the imbalance is
**localized to the MLP residual add**, not diffuse.

## 2. Why the method fails (the refutation, Phases 2–4)

The method's algebra only folds **exactly** at the `LN→Linear` boundary (Phase 0 method
doc). But the damage is not there:

- **Phase 2 (BERT/MNLI):** a mixed-precision oracle keeping `out_proj`/`fc2` in FP16 cuts
  the W8A8 gap from **9.01 → 0.36 pp** — ~96% of the damage is at the non-foldable MLP-add
  output projections. Exact foldable smoothing (SmoothQuant, or RBBQ-A's branch criterion)
  tops out ~6.2 pp, far from the oracle.
- **Phase 3 (BERT decomposition):** that damage is **activation-quant** (FP16-weight+quant-
  act still 8.37 pp; quant-weight+FP16-act only 0.60 pp; per-channel weight irrelevant).
  Variant B is a *weight-side* rebalancing — it cannot reach an activation-side bottleneck,
  so it was **not built** (refuted premise). Per-token dynamic act on `out_proj`/`fc2`
  alone → 0.36 pp = oracle.
- **Phase 4 (Qwen confirms):** identical pattern — damage localizes to `down_proj`
  (ppl 110→10.5 by keeping it FP16), is activation-side (FP16-weight+quant-act 104 ppl;
  quant-weight+FP16-act 10.7), and per-token dynamic on `down_proj` alone → 10.67 ≈ FP 9.80.
  Crucially **not** at the foldable LN→qkv path SmoothQuant targets.

**Verdict:** the residual-branch *imbalance* is real but is carried by **per-token
activation outliers downstream of GELU/SwiGLU** at the MLP down-projection — non-foldable,
weight-irrelevant, and not fixable by branch rebalancing. RBBQ is refuted on both families.

## 3. The constructive pivot: selective-granularity W8A8 (Phases 5–8)

The diagnostics handed us a method: keep cheap static per-tensor quant on most linears;
apply **per-token dynamic** only where activation outliers demand it.

- **Selection rule (Phase 5):** a cheap calibration statistic, the per-token spread
  `D = max_token(rowmax)/median_token(rowmax)`, flags exactly those linears. Quantize
  per-token-dynamic iff `D ≥ thr`.
  - Qwen: **32/196 linears dynamic (16%, the down_proj family) → ppl 10.57** vs plain
    static 110.4, all-dynamic 10.02, FP 9.80 — ~99% of the gap recovered, 84% of linears
    left on cheap static.
  - BERT: 28–37/72 dynamic → 0.8435–0.8455 ≈ all-dynamic (outliers more distributed).
- **The rule is principled, not lucky (Phases 6, 8):** a "more principled" local
  static-vs-FP output-error metric `E` is **much worse** (Qwen k=13: picks qkv → ppl 105.8
  vs `D`'s down_proj → 12.05). End-to-end sensitivity ≠ local quant error (qkv error is
  absorbed by softmax; down_proj error propagates). The corrected `E'` (static-vs-dynamic
  output error) fixes `E` but still loses to `D` — and `D` is the cheapest (an input
  reduction, no matmul). **Selection rule settled: use `D`.**

## 4. The honest latency / cost story (Phases 7, 9, 10)

- **Per-token dynamic is cheap and free when fused.** Triton fused quant/dequant kernels
  (Phase 9, validated bit-close to eager) make per-token dynamic ≈ static
  (fused_static ≈ fused_dynamic). So selective gives **no latency edge** over all-dynamic;
  it is **not** a speed optimization.
- **int8 GEMM is the real cost variable, and it's hardware-dependent.** On the H200 dev box,
  even a correctness-verified tuned Triton int8 GEMM is **4.1× slower than bf16**
  (`_int_mm` 7.3×), while **fp8 `_scaled_mm` is 1.6× faster than bf16** (Phase 10).
- **Format ↔ hardware (the unifying conclusion):**
  - **INT8 ↔ A100** — IMMA tensor cores, no FP8 → int8 W8A8 is a real ~2× latency win
    there; this is the project's stated target, and explains why the H200 numbers look bad.
  - **FP8 ↔ Hopper** — the low-precision latency path on H200.
  - **Memory (−50% W+A) is the hardware-agnostic win**, dominant for memory-bound decode.

## 5. Net contributions

1. **A clean negative result with mechanism:** residual-branch rebalancing cannot fix W8A8,
   because the damage is activation-granularity at the MLP down-projection (post-nonlinearity
   per-token outliers), on both post-norm (BERT) and pre-norm-RMSNorm (Qwen) families.
2. **Selective-granularity W8A8:** a cheap `D`-based rule selects the few linears needing
   per-token dynamic, reaching near-FP accuracy while keeping ~84% of linears on a fully-
   static integer path. Format-agnostic (int8 on A100 / fp8 on Hopper).
3. **A negative methodological lesson:** local quantization-error metrics mislead for linear
   selection; the static-vs-dynamic input-scale mismatch is the right, cheapest signal.
4. **An honest latency map:** per-token dynamic is free when fused; the W8A8 latency win is
   format×hardware (int8→A100, fp8→Hopper), with memory the portable benefit.

## 6. Limitations / what is not yet established

- **Generality:** shown on BERT-base + Qwen2.5-1.5B only. ViT-B (vision) and larger LLMs
  (7–13B) untested — the next experiment.
- **Target-hardware latency:** the int8 latency win is *argued* for A100 but measured only
  on H200 (where int8 loses). Needs an A100 re-benchmark (CLAUDE.md reserves A100 for final
  runs); the Triton int8 kernel is portable for this.
- **Baselines:** SmoothQuant compared on BERT; a full SmoothQuant/OS/PEG sweep on Qwen and
  more GLUE/SQuAD/MMLU/GSM8K tasks remains.

## 7. Open work (priority order)

1. **Generality experiment (next session):** ViT-B (ImageNet) and a 7B LLM — does the
   down-projection concentration and the `D` rule transfer across vision and scale?
2. A100 int8 GEMM re-benchmark (target HW) to confirm the latency win.
3. Full accuracy matrix vs SmoothQuant / Outlier Suppression / per-tensor / all-dynamic.
4. An **FP8** selective-granularity path (Hopper) parallel to int8 (A100).

## 8. Artifact index

| Phase | Report | Code |
|---|---|---|
| 0 spec+harness | `phase0_report.md`, `rbbq_method.md` | `rbbq/phase0_bert_sst2.py` |
| 1 branch-energy gate | `phase1_report.md` | `rbbq/phase1_branch_energy.py` |
| 2 Variant A | `phase2_report.md` | `rbbq/phase2_rbbq_a.py` |
| 3 Variant B refuted | `phase3_report.md` | `rbbq/phase3_variant_b.py` |
| 4 Qwen gate | `phase4_report.md` | `rbbq/phase4_qwen_decomp.py` |
| 5 selective | `phase5_report.md` | `rbbq/phase5_selective.py` |
| 6 sensitivity | `phase6_report.md` | `rbbq/phase6_sensitivity.py` |
| 7 cost | `phase7_report.md` | `rbbq/phase7_latency.py` |
| 8 E' rule | `phase8_report.md` | `rbbq/phase8_eprime.py` |
| 9 fused kernel | `phase9_report.md` | `rbbq/phase9_fused_w8a8.py`, `phase9_bench.py` |
| 10 int8/fp8/bf16 | `phase10_report.md` | `rbbq/phase10_int8_gemm.py` |

Env: conda `specdec` (torch 2.10+cu128, H200); datasets via `nyu-mll/glue` /
`Salesforce/wikitext`; `HF_HUB_OFFLINE=1` to avoid hub 504s.

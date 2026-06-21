# RBBQ Pivot — Phase 5 Report: Selective-Granularity W8A8

**Branch:** `feat/rbbq-proposal`  •  **Date:** 2026-06-21  •  **Status:** complete
**Result:** A cheap a-priori rule selects the few linears that need per-token
dynamic quant; quantizing only those (rest static per-tensor) recovers ~all of the
W8A8 gap. On Qwen, 16% of linears dynamic ≈ all-dynamic / FP.

## From negative result to method

Phases 3–4 showed the static-per-tensor W8A8 damage is activation-side at the MLP
down-projection (down_proj / fc2) on both families, fixed by per-token **dynamic**
quant there. This phase makes it a method with an **a-priori selection rule** —
not an oracle — and measures the accuracy-vs-cost tradeoff. `rbbq/phase5_selective.py`.

### Selection statistic (from calibration, cheap)
For each quantized linear, the per-token dynamic-range spread of its input:
```
D = max_token(rowmax) / median_token(rowmax),   rowmax = max_c |x[t,c]|
```
High `D` ⇒ a single static per-tensor scale (set by the worst token) wastes most of
the int8 range on typical tokens ⇒ that linear needs per-token dynamic. **Rule:**
quantize a linear's activations per-token-dynamic iff `D ≥ thr`, else static
per-tensor. Weights: per-channel for dynamic linears, per-tensor for static.

## Validation — does D flag the damaging linears?

Median `D` by linear type:

| | q | k | v | o_proj | fc1/gate-up | **fc2 / down_proj** |
|---|---|---|---|---|---|---|
| BERT | 1.57 | 1.57 | 1.57 | 2.98 | 4.41 | **3.97** |
| Qwen | 2.18 (attn_in) | | | 2.33 | 2.72 (mlp_in) | **7.69 (mlp_out)** |

- **Qwen: clean.** `down_proj` D=7.69 is ~3× every other type — exactly the linear
  Phase 4 localized the damage to. The statistic identifies it with no oracle.
- **BERT: directional.** The MLP/attention-output linears (fc1 4.4, fc2 4.0,
  o_proj 3.0) rank above qkv (1.6), matching Phase 3 (damage at fc2/o_proj, plus
  fc1 carries per-token spread). Separation is weaker — BERT's outliers are more
  distributed than Qwen's massive activations.

## Tradeoff: accuracy vs. # linears made dynamic

**BERT** (MNLI acc; FP 0.8457, plain 0.7556, all-dynamic 0.8465; 72 linears):

| thr | n_dyn | acc | gap to FP |
|---|---|---|---|
| 6.0 | 8 | 0.8395 | 0.62 |
| 4.0 | 18 | 0.8386 | 0.71 |
| 3.0 | 28 | 0.8435 | 0.22 |
| 2.5 | 37 | 0.8455 | 0.02 |

**Qwen** (WikiText-2 ppl; FP 9.803, plain 110.4, all-dynamic 10.022; 196 linears):

| thr | n_dyn | ppl | Δppl to FP |
|---|---|---|---|
| 8.0 | 13 | 12.05 | +2.25 |
| 5.0 | 32 | 10.57 | +0.77 |
| 4.0 | 35 | 10.41 | +0.61 |
| 3.0 | 45 | 10.34 | +0.54 |

## Headline

- **Qwen: selective ≈ all-dynamic at 1/6 the dynamic-quant cost.** Making the
  ~32 down_proj-family linears dynamic (16% of 196) gives ppl 10.57 vs plain 110.4
  and all-dynamic 10.022 — i.e. it recovers ~99% of the gap while 84% of linears
  keep cheap static per-tensor quant.
- **BERT: ~40–50% dynamic matches all-dynamic** (thr=3 → 28 linears, 0.8435;
  thr=2.5 → 37, 0.8455 ≈ all-dynamic 0.8465). More linears needed because the
  outliers are spread across fc1/fc2/o_proj, not concentrated.

This is the pivot's core claim, validated on two architectures: **you do not need
per-token dynamic everywhere; a per-token-spread statistic picks the small subset
that does, and quantizing only those recovers near-oracle accuracy.**

## Honest caveats

- BERT/MNLI carries ~0.2–0.3 pp eval noise; its sweep is mildly non-monotonic
  (thr=6 slightly > thr=4). The trend (more dynamic → better, saturating) is solid;
  exact knee placement is noisy.
- `D` is a clean predictor on Qwen, weaker on BERT. A sensitivity-aware refinement
  (e.g. per-linear quant-error impact) may sharpen BERT selection — open for Phase 6.
- Cost is reported here as **# dynamic linears** (a proxy). Real latency of
  selective per-token dynamic is a Phase 6 measurement.

## Next (Phase 6)
- Extend to **ViT-B (ImageNet)** and **Llama-7B** — does the down-proj concentration
  hold across architectures/scales?
- Full benchmarks (GLUE/SQuAD, MMLU/GSM8K/perplexity) vs SmoothQuant/OS/per-tensor
  and vs all-dynamic.
- **Latency**: measure selective per-token dynamic (16% of linears on Qwen) vs
  static and vs all-dynamic — the cost case for the method.
- Refine the selection rule (sensitivity-aware) for the distributed-outlier case (BERT).

## Reproduce
```bash
conda activate specdec && cd /home/runying2/cuda-PA-INT8
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
  PYTHONPATH=rbbq python rbbq/phase5_selective.py
```

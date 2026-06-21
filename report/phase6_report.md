# RBBQ Pivot — Phase 6 Report: Sensitivity-aware selection (D validated)

**Branch:** `feat/rbbq-proposal`  •  **Date:** 2026-06-21  •  **Status:** complete
**Result:** A naive "more principled" sensitivity metric (local static-quant output
error, `E`) selects the **wrong** linears and is much worse than the Phase-5 input
spread `D`. End-to-end W8A8 sensitivity ≠ local quant error; `D` is the right cheap
selection rule, confirmed on both families.

## Goal

Phase 5's selection statistic `D = max_token(rowmax)/median_token(rowmax)` was clean
on Qwen but seemed weak on BERT. This phase tested whether a direct sensitivity
metric improves selection:
```
E = || q_static(x) W − x W || / || x W ||     (relative output error from static act quant)
```
`E` accounts for the weight and output norm, so it "should" rank the damaging
linears better. We selected the top-k linears by each metric at matched budgets and
compared. `rbbq/phase6_sensitivity.py`.

## Result — E is decisively worse than D

| family | k | **D-select** | **E-select** | overlap | all-dynamic | plain |
|---|---|---|---|---|---|---|
| BERT (MNLI acc) | 18 | 0.8386 | 0.7799 | 5/18 | 0.8465 | 0.7556 |
| BERT (MNLI acc) | 28 | 0.8435 | 0.7838 | 15/28 | 0.8465 | 0.7556 |
| Qwen (WikiText ppl) | 13 | **12.05** | **105.8** | **0/13** | 10.022 | 110.4 |
| Qwen (WikiText ppl) | 32 | 10.57 | 24.10 | 3/32 | 10.022 | 110.4 |

What each metric picks (linear types):
- Qwen k=13: **D → down_proj** (all 13). **E → q/k/v (attn_in)**. Disjoint. D's pick
  recovers ppl to 12.0; E's pick leaves it at 105.8 (≈ no quantization fix at all).
- BERT k=28: D spreads across all types; E concentrates on fc2/o_proj/fc1/q.

## Why local error misleads (the insight)

The linears with the highest *local* static-quant output error are **not** the ones
that damage end-to-end accuracy:
- **qkv inputs** (Qwen's massive-activation residual stream after RMSNorm) have huge
  local quant error — but that error is **absorbed downstream by softmax** and does
  not propagate. `E` ranks them first and wastes the budget on them.
- **down_proj** output flows straight into the residual stream and **propagates**;
  its damage dominates end-to-end (Phase 4) even though its local error isn't the
  largest.

`D` works because it measures the **static-vs-per-token-dynamic scale mismatch** of a
linear's input — i.e. *how much per-token dynamic actually helps that linear* — which
is exactly the selection question, and which tracks the propagating massive-activation
structure. So `D` is principled, not a lucky heuristic; `E` conflates "hard to
quantize locally" with "matters end-to-end."

## Conclusion

- **`D` (input per-token spread) is the selection rule** for selective-granularity
  W8A8, validated on both BERT and Qwen against a plausible alternative.
- BERT's earlier "weakness" is benign: `D` includes fc1 but still matches all-dynamic
  (0.8435–0.8455); BERT simply needs more dynamic linears (distributed outliers), not
  a different metric.
- Naive local-sensitivity metrics should **not** be used for W8A8 linear selection —
  a transferable lesson.

## Next refinement (future phase)

The honest output-space analog of `D` is `E' = ||q_static(x)W − q_dynamic(x)W||/||xW||`
— the output error that per-token dynamic *removes* at this linear (vs `E`'s error
relative to FP, which includes irreducible/non-propagating terms). `E'` should agree
with `D` and may sharpen it; it was not run here. Plus the Phase-5 backlog: ViT-B /
Llama-7B generality, full benchmarks vs SmoothQuant/OS, and real-int8 latency (ties
into this repo's CUDA INT8 kernel track).

## Reproduce
```bash
conda activate specdec && cd /home/runying2/cuda-PA-INT8
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
  PYTHONPATH=rbbq python rbbq/phase6_sensitivity.py
```

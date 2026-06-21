# RBBQ Pivot — Phase 8 Report: E' selection refinement (D wins, settled)

**Branch:** `feat/rbbq-proposal`  •  **Date:** 2026-06-21  •  **Status:** complete
**Result:** The corrected output-space metric `E'` fixes Phase 6's broken `E` (it now
finds the right linears) but still **loses to the input-spread `D`** on both families.
`D` is best on accuracy *and* cheapest (input reduction, no matmul). The selection-rule
question is settled: **use `D`.**

## Goal

Phase 6 showed the naive sensitivity metric `E = ||q_static(x)W − xW||/||xW||` (local
static-vs-FP error) misranks linears. The principled fix is the error per-token dynamic
*removes*:
```
E' = || q_static(x) W − q_dynamic(x) W || / || x W ||,
```
the output-space analog of `D` (input static-vs-dynamic scale mismatch). This phase
compares `D`, `E`, `E'` at matched budgets `k`. `rbbq/phase8_eprime.py`.

## Results (top-k by each metric → per-token dynamic; rest static C1)

**BERT** (MNLI acc; FP 0.8457, plain 0.7556, all-dynamic 0.8465):

| k | D | E | E' |
|---|---|---|---|
| 18 | **0.8386** | 0.7799 | 0.7992 |
| 28 | **0.8435** | 0.7838 | 0.8022 |

**Qwen** (WikiText-2 ppl; FP 9.803, plain 110.4, all-dynamic 10.022):

| k | D | E | E' |
|---|---|---|---|
| 13 | **12.05** | 105.8 | 17.95 |
| 32 | **10.57** | 24.10 | 12.86 |

## Reading

1. **`E'` >> `E`** — the framing was the issue, not the idea. Comparing static to
   *dynamic* (not to FP) makes the metric correctly identify the damaging group: at
   Qwen k=13, `E'` picks `mlp_out` (down_proj), same group as `D`, recovering ppl to
   17.95 vs `E`'s 105.8. So "the error per-token dynamic removes" is the right notion of
   per-linear quant sensitivity.

2. **`D` still wins.** Even within the right group, `D` ranks *which* linears matter
   better (Qwen k=13: `D` 12.05 vs `E'` 17.95; BERT k=28: `D` 0.8435 vs `E'` 0.8022).
   `D` directly measures the static-vs-dynamic input-scale mismatch — the benefit of
   going dynamic — without the weight-norm weighting that `E'`'s output error imposes.

3. **`D` is also the cheapest.** `D` needs only an input reduction at calibration;
   `E'` needs two extra matmuls per linear. `D` wins on accuracy *and* cost.

## Conclusion — selection rule settled

For selective-granularity W8A8, **select per-token-dynamic linears by the input
per-token spread `D = max_token(rowmax)/median_token(rowmax)`**. Increasingly
"principled" output-error metrics (`E` → `E'`) climb toward `D` but never beat it, and
cost more. The simplest statistic is the right one. This closes the selection-rule
thread opened in Phase 5.

## Backlog still open (future phases)
- Generality: ViT-B (ImageNet) and Llama/Mistral-7B — does the down-proj concentration
  and the `D` rule transfer across architecture and scale?
- Full benchmarks (GLUE/SQuAD, MMLU/GSM8K/perplexity) vs SmoothQuant / Outlier
  Suppression / per-tensor / all-dynamic.
- Fused W8A8 kernel with a selective (mixed static/dynamic) epilogue → true end-to-end
  latency (ties to this repo's PagedAttention/INT8 CUDA track).

## Reproduce
```bash
conda activate specdec && cd /home/runying2/cuda-PA-INT8
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
  PYTHONPATH=rbbq python rbbq/phase8_eprime.py
```

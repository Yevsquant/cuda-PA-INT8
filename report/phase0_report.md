# RBBQ Phase 0 Report — Spec + Repro

**Branch:** `feat/rbbq-proposal`  •  **Date:** 2026-06-20  •  **Status:** complete, gate met

Phase 0 goal (from `rbbq_proposal.md`): produce the formal method spec, and stand
up + validate the W8A8 + SmoothQuant baseline harness on BERT-base/SST-2.

## Deliverables

| Deliverable | File |
|---|---|
| Formal method & folding derivations (both variants, pre/post-norm) | `report/rbbq_method.md` |
| W8A8 fake-quant + SmoothQuant harness | `rbbq/phase0_bert_sst2.py` |
| Results | `report/phase0_results.json` |

## Environment (for the next session)

- Conda env **`specdec`** (`conda activate specdec`): torch 2.10.0+cu128,
  transformers 5.11.0, datasets 5.0.0. GPU: **H200** (sm_90), CUDA available.
- RBBQ is a **separate track** but reuses this working torch env (do NOT build a
  fresh env — the repo's memory notes a libcudart pinning trap). New code lives
  under `rbbq/`.
- **Gotcha:** datasets 5.0 rejects the legacy `glue` id; use
  `load_dataset("nyu-mll/glue", "sst2")`.
- Checkpoint: `textattack/bert-base-uncased-SST-2` (cached).

## Results (SST-2 validation, 872 ex)

| Config | Description | Acc | Δ vs FP |
|---|---|---|---|
| `fp`    | fp32 reference | **0.9243** | — |
| `c1`    | static per-tensor act + per-tensor weight (HARD) | 0.9128 | −1.15 pp |
| `c1_sq` | c1 + SmoothQuant (qkv, fc1) | 0.9117 | −1.26 pp |
| `c2`    | per-token dynamic act + per-channel weight (EASY) | 0.9232 | −0.11 pp |
| `c2_sq` | c2 + SmoothQuant | 0.9220 | −0.23 pp |

## Gate status — MET

- **Repro gate:** FP accuracy **0.9243 (806/872)** reproduces the canonical
  TextAttack SST-2 dev number (92.43%) exactly. The eval harness is correct.
- **Harness gate:** all five configs run end-to-end; W8A8 fake-quant and
  explicit SmoothQuant smoothing both function; results serialize to JSON.

## Findings worth carrying forward

1. **Post-norm BERT has no free foldable boundary.** Every LayerNorm output is
   simultaneously (a) a linear's input and (b) the residual added to the next
   sublayer. Folding a per-channel smooth scale into the LN therefore breaks the
   residual add — exactly the wall `rbbq_method.md` predicts for post-norm. The
   harness applies smoothing as an **explicit, exact** per-channel pre-linear
   scale (`x/s @ (W·s)`); accuracy is identical to folded SmoothQuant, but the
   "is it free?" question is real and deferred to Phase 5 (latency).

2. **SmoothQuant does not recover the c1 drop on this task** (−1.26 vs −1.15;
   flat-to-slightly-worse). Because smoothing targets the qkv/fc1 (LN-fed)
   inputs, the fact that it doesn't help suggests the damaging outliers are NOT
   there but in the **residual stream feeding `attention.output.dense` / FFN
   `output.dense`** — which is precisely the residual-branch imbalance RBBQ
   targets. This is motivating, not discouraging, but note SST-2 is an easy,
   robust task (only −1.15 pp even for plain c1), so it is a weak discriminator;
   harder GLUE tasks (CoLA, MNLI, QNLI) and the LLM tier will be more telling.

3. **c2 (dynamic per-token + per-channel) is already near-lossless** (−0.11 pp).
   As anticipated, the easy config masks outlier effects; c1 is where the method
   must prove itself. Confirms the resolved decision to gate on c1.

## Implications for the plan

- Phase 1 instrumentation should measure branch energy `E_id/E_up/r[c]` at the
  **residual adds feeding out_proj/fc2**, not only at the qkv/fc1 inputs, given
  finding 2.
- Add a harder GLUE task to the prove-out tier early (SST-2 alone under-resolves
  method differences). CoLA (Matthews corr) or MNLI recommended.
- `rbbq/phase0_bert_sst2.py` is the reusable spine: `QuantLinear`,
  `collect_perchannel_max`, `smooth_scales`, calibration. Phase 2 (RBBQ-A) swaps
  `smooth_scales`' criterion for the branch-energy one.

## How to reproduce

```bash
conda activate specdec
cd /home/runying2/cuda-PA-INT8
python rbbq/phase0_bert_sst2.py            # ~8s total on H200
```

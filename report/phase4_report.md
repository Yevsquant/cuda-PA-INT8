# RBBQ Phase 4 Report — LLM gate: branch thesis refuted on Qwen too

**Branch:** `feat/rbbq-proposal`  •  **Date:** 2026-06-21  •  **Status:** complete
**Verdict:** The RBBQ branch-rebalancing thesis is refuted on **decoder LLMs as
well**. The W8A8 bottleneck is the *same* as BERT — activation-granularity at the
MLP down-projection input — and is **not** at the foldable LN→qkv path. Selective
per-token-dynamic quant on one linear per layer fixes it.

## Goal (the Phase 3 gate)

Phase 3 refuted RBBQ on BERT but flagged decoder LLMs as possibly different: Qwen's
massive activations (Phase 1: `dom_id ~135 000×`) live in the residual stream
feeding the foldable LN→qkv/gate-up path, where branch rebalancing could matter.
This phase runs the same group + weight/act decomposition on Qwen2.5-1.5B, WikiText-2
perplexity, C1 (static per-tensor W8A8). `rbbq/phase4_qwen_decomp.py`.

## Group localization — which linears carry the damage

FP bf16 ppl = **9.803**. Quantize all 7 linears/layer (C1), keep one group FP16:

| arm | kept FP16 | ppl | Δppl |
|---|---|---|---|
| plain | — | 110.4 | +100.6 |
| fp16_attn_in | q/k/v_proj (LN-fed) | 93.6 | +83.8 |
| fp16_attn_out | o_proj | 104.7 | +94.9 |
| fp16_mlp_in | gate/up_proj (LN-fed) | 97.2 | +87.4 |
| **fp16_mlp_out** | **down_proj** | **10.54** | **+0.74** |

**Damage is concentrated at `down_proj`** (the MLP update-branch output projection)
— the direct analog of BERT's `fc2`. Keeping the *foldable* LN-fed groups (qkv,
gate/up) in FP16 barely helps (still 93–97 ppl). This is the opposite of what
foldable-smoothing (SmoothQuant) lore predicts, and matches BERT exactly.

## Weight/activation split on down_proj

| arm | down_proj treatment | ppl | Δppl |
|---|---|---|---|
| mlp_out_w | FP16 weight, quant act | 104.0 | +94.2 |
| mlp_out_a | quant weight (per-tensor), FP16 act | 10.70 | +0.90 |
| **mlp_out_dyn** | per-token **dynamic** act + per-ch weight | **10.67** | +0.86 |

Identical to BERT: the damage is **activation-quant** (static per-tensor on the
SwiGLU/massive-activation inputs), **not weight-quant** (per-tensor weight quant of
down_proj is nearly free, +0.90). Per-token **dynamic** act on just `down_proj`
recovers ppl to 10.67 ≈ the FP16 oracle (10.54).

## Conclusion — RBBQ refuted across both families

The residual-branch *imbalance* is real (Phase 1 gate passed on both models), but on
**both** BERT (post-norm) and Qwen (pre-norm RMSNorm decoder LLM) the W8A8 damage is:

- localized to the **MLP down-projection** (update-branch output linear),
- **activation-side**, driven by per-token outliers downstream of the
  GELU/SwiGLU nonlinearity (non-foldable),
- **not** weight-side, **not** at the foldable LN→qkv boundary, and **not** fixable
  by rebalancing the residual branches.

RBBQ (Variant A exact-foldable, Variant B weight-side rebalancing) cannot reach this
bottleneck. The branch-rebalancing thesis is **refuted**.

## What actually works (the constructive result)

**Selective per-token-dynamic activation quantization on the MLP down-projection**
(and BERT's out_proj/fc2) closes the W8A8 gap to the FP16 oracle on both families:
- BERT/MNLI: 9.01 pp → 0.36 pp (= oracle)
- Qwen/WikiText-2: +100.6 ppl → +0.86 ppl (≈ oracle +0.74)

with no graph transform, no folding/inverse, and per-tensor static quant retained on
the other ~5 linears/layer. This is a clean, defensible contribution that aligns with
the modern LLM-quant understanding (massive activations / down_proj are the hard part)
and is distinct from — and a corrective to — the original branch-balancing proposal.

## Recommended pivot (if the project continues)

Reframe around **"selective-granularity W8A8"**: a principled rule for which linears
need per-token-dynamic (the update-branch output projections, where post-nonlinearity
activations carry per-token outliers) vs static per-tensor (the rest). Then:
- characterize the rule across BERT/ViT/Llama-7-13B,
- full GLUE/SQuAD + MMLU/GSM8K/perplexity vs SmoothQuant/OS/per-tensor,
- latency: per-token dynamic on 1 of 7 linears/layer is cheap — quantify it.

## Reproduce
```bash
conda activate specdec && cd /home/runying2/cuda-PA-INT8
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
  PYTHONPATH=rbbq python rbbq/phase4_qwen_decomp.py
```

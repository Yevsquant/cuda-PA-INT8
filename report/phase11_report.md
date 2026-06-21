# RBBQ Pivot — Phase 11 Report: Generality on Mistral-7B (different family, 7B)

**Branch:** `feat/rbbq-proposal`  •  **Date:** 2026-06-21  •  **Status:** complete
**Result:** The Phase 3–5 findings **transfer cleanly** to Mistral-7B-Instruct-v0.2 (a
different family at 7B scale): W8A8 damage localizes to the MLP down-projection, is
activation-side, and is fixed by per-token-dynamic / the `D` rule. The conclusion now
holds across post-norm encoder (BERT), pre-norm decoder (Qwen-1.5B), and a different
pre-norm decoder family at scale (Mistral-7B).

## Setup

`rbbq/phase11_generality.py` — reuses the Phase 4 harness verbatim (Mistral has the same
`{q,k,v,o}_proj` / `{gate,up,down}_proj` layout). WikiText-2 perplexity, C1 static
per-tensor W8A8. The Phase 4/5 code ran unmodified except the model id, which is itself
evidence of structural generality.

## Results (FP bf16 ppl = 6.329)

| arm | down_proj treatment | ppl | Δppl |
|---|---|---|---|
| plain C1 | all static per-tensor | 146.2 | +139.9 |
| fp16_mlp_out | down_proj FP16 | 6.474 | **+0.15** |
| mlp_out_w | FP16 weight, quant act | 140.8 | +134.4 |
| mlp_out_a | quant weight, FP16 act | 6.558 | **+0.23** |
| mlp_out_dyn | per-token dynamic act | 6.493 | +0.16 |
| D_selective | top-D linears dynamic | 6.552 | +0.22 |
| all_dynamic | all dynamic | 6.380 | +0.05 |

`D` by linear type (median): **mlp_out 8.37** > mlp_in 4.71 > attn_out 3.18 > attn_in 3.16.

## Findings — all three questions answer YES

1. **Damage localizes to `down_proj`.** Keeping just the MLP down-projection in FP16 cuts
   the gap from **139.9 → 0.15 ppl**. Same as BERT (fc2) and Qwen (down_proj).
2. **It is activation-side.** FP16-weight + quant-act stays broken (140.8); quant-weight +
   FP16-act recovers (6.56). Weight quant of down_proj is nearly free.
3. **Per-token dynamic / `D` fixes it.** down_proj per-token dynamic → 6.49; the `D`-rule
   selective → 6.55 — both ≈ the FP16 oracle (6.47) and near all-dynamic (6.38). `D`
   cleanly ranks `mlp_out` highest (8.37 vs ~3.2 for attention).

Minor caveat: top-`D` here also pulled in some `mlp_in` linears (mlp_in `D`=4.71 is 2nd),
so `D_selective` (6.55) is a hair behind the down_proj-targeted `mlp_out_dyn` (6.49) — the
same threshold-tuning nuance noted in Phase 5; a group-aware threshold closes it.

## Generality matrix (now established)

| model | norm | type | MLP | scale | damage at down-proj? | act-side? | `D` flags it? |
|---|---|---|---|---|---|---|---|
| BERT-base | post-norm LN | encoder | GELU | 110M | ✅ (fc2) | ✅ | directional |
| Qwen2.5-1.5B | pre-norm RMS | decoder | SwiGLU | 1.5B | ✅ | ✅ | ✅ (clean) |
| **Mistral-7B** | pre-norm RMS | decoder | SwiGLU | 7B | ✅ | ✅ | ✅ (clean) |

The selective-granularity W8A8 result spans both norm placements, encoder/decoder,
GELU/SwiGLU, and 0.1B–7B scale.

## Still open
- **ViT-B (vision)** — the one architecture class untested; blocked on ImageNet (gated/
  uncached). The genuinely different modality test.
- A100 int8 latency confirmation (Phase 10); full accuracy matrix vs SmoothQuant/OS.

## Reproduce
```bash
conda activate specdec && cd /home/runying2/cuda-PA-INT8
HF_DATASETS_OFFLINE=1 PYTHONPATH=rbbq python rbbq/phase11_generality.py
```

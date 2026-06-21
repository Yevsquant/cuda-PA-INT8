# RBBQ Revised — Phase 16 Report: FP8 selective path (hardware-aware, final)

**Branch:** `feat/rbbq-proposal`  •  **Date:** 2026-06-21  •  **Status:** complete
**Result:** **FP8 (e4m3) all-static is essentially lossless** on Qwen (+0.09 ppl) and needs
**no** selective granularity — its floating-point range absorbs the down_proj per-token
outliers that wreck INT8 static (110 ppl). So selective granularity is specifically the
**INT8/A100** solution; on **Hopper, FP8 all-static** sidesteps the problem (and is faster
than bf16, Phase 10). This completes the format↔hardware story.

## Goal (solution #5, final component)

The `D` selection rule is format-agnostic. INT8 is the A100 path; FP8 is the Hopper path
(Phase 10: fp8 `_scaled_mm` 0.6× bf16; int8 GEMM 4–7×). Test the FP8 accuracy question:
does FP8's wider dynamic range tolerate the per-token down_proj outliers, i.e. does FP8 even
*need* selective granularity? `rbbq/phase16_fp8_selective.py` (Qwen2.5-1.5B, WikiText-2).

## Results (FP ppl 9.803)

| method | ppl | Δppl |
|---|---|---|
| **FP8 all-static** | **9.893** | **+0.090** |
| FP8 selective (dynamic @ down_proj) | 9.891 | +0.088 |
| FP8 all-dynamic | 9.878 | +0.075 |
| — INT8 plain (all static) | 110.4 | +100.6 |
| — INT8 RBBQ-C selective | 10.49 | +0.69 |
| — INT8 all-dynamic | 10.02 | +0.22 |

## Findings

1. **FP8 all-static is near-lossless and selectivity-free.** +0.09 ppl with a single static
   per-tensor scale per linear, including down_proj. FP8 static ≈ FP8 selective ≈ FP8
   dynamic — the selective machinery buys nothing in FP8.
2. **The mechanism: floating-point gives uniform *relative* precision.** FP8 e4m3 (4 exp
   bits) keeps ~3 mantissa bits of relative precision at every magnitude, so a per-tensor
   scale set by an outlier token still represents the small bulk values with constant
   relative error. INT8 is fixed-point: once the scale is set by a huge per-token outlier,
   the bulk collapses to a few levels. This is exactly why INT8 static fails (110) and FP8
   static does not (9.89), and why per-token *dynamic* (or selectivity) is an INT8-only
   need.
3. **Selective granularity is the INT8/A100 contribution.** On Hopper, the right move is
   simply **FP8 all-static** — simpler than selective, near-lossless, and faster than bf16.

## Final deployment story (format ↔ hardware)

| hardware | format | method | ppl gap | latency vs bf16 |
|---|---|---|---|---|
| **A100** | INT8 | **RBBQ-C selective** (per-token dynamic @ down_proj; folds where legal) | +0.69 | win (IMMA; needs A100 confirm) |
| **Hopper** | FP8 | **all-static** (no selectivity needed) | +0.09 | 0.6× (Phase 10) |
| any | INT8/FP8 | — | — | memory −50% |

The `D` rule and the residual diagnosis remain available, but are **only needed for INT8**.
FP8's numeric format dissolves the per-token-outlier problem that motivated the whole method.

## Honest closing position

- The original RBBQ residual-rebalancing actuator is refuted (Phases 2–4).
- Its residual *diagnosis* survives and localizes the W8A8 failure to down_proj (Phases
  1–4, 11–12).
- The corrected method — **Residual-Guided Selective-Granularity W8A8 (RBBQ-C)**, optionally
  with legal folds (Phase 13), static-clip rescue (Phase 14), or token-outlier fallback
  (Phase 15) — solves the **INT8** failure on A100.
- On **Hopper**, FP8 all-static is the simpler, near-lossless answer; selectivity is
  unnecessary there.

## Remaining (hardware-blocked)
- A100 INT8 accuracy+latency confirmation (target HW; portable kernels ready, Phases 9–10).
- ViT-B vision generality (ImageNet blocked).

## Reproduce
```bash
conda activate specdec && cd /home/runying2/cuda-PA-INT8
HF_DATASETS_OFFLINE=1 PYTHONPATH=rbbq python rbbq/phase16_fp8_selective.py
```

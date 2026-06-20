# Residual-Balanced Branch Quantization (RBBQ)

Research proposal. Status: plan / pre-implementation.

## Motivation

Several lines of work locate the place where Transformer post-training
quantization (PTQ) breaks not in generic matrix multiplication but in the
**residual stream**:

- **Bondarenko et al.** — residual outliers; specific embedding dimensions
  develop very large values that dominate quantization ranges.
- **Outlier Suppression (Wei et al.)** — LayerNorm `γ` acts as an amplifier of
  these outlier channels; "gamma migration" moves the difficulty elsewhere.
- **Recent reproductions** — residual-driven *channel dominance*: a few
  channels carry most of the energy of the stream.

RBBQ targets the residual **add** directly. At every `x + F(x)`, the identity
branch `x` and the update branch `F(x)` can have badly mismatched per-channel
energy. Whichever branch owns the outlier channels sets the dynamic range, and
the quieter branch is crushed to a handful of quantization levels. The proposal:
estimate the per-channel energy imbalance at each residual add, then learn or
calibrate a pair of folded, branch-specific rescalings that equalize the add
before quantization and invert the transform afterward **where algebraically
possible**.

## Hypothesis (Phase 1 gate)

For a pre-norm block:

```
a = x + Attn(LN1(x))     # identity x, update Attn(·)
y = a + MLP(LN2(a))      # identity a, update MLP(·)
```

Define per-channel energies and imbalance:

```
E_id[c] = E_t[ x[c]^2 ]
E_up[c] = E_t[ f[c]^2 ]          f = Attn(LN1(x)) or MLP(LN2(a))
r[c]    = E_id[c] / E_up[c]
```

**Claim:** `r[c]` is heavy-tailed across channels — a few channels are dominated
by one branch. If `r[c]` is flat, the premise is dead and this is a negative
result we want to know early.

## Method — and the exactness boundary (the crux)

A per-channel diagonal `D` can be moved **exactly** only between two linear ops.
The path `LN-affine(γ,β) → Linear(W)` is linear in the channel scale:

```
(γ ⊙ z) W  =  ((γ⊙s) ⊙ z) (diag(1/s) W)      # exact — what SmoothQuant/OS+ exploit
```

But the residual add is *before* LN, and the normalization statistic (RMS or
mean/var) does **not** commute with a per-channel `D`. So we cannot literally
rescale the two branches at the add and invert across LN exactly. This forces
two variants:

- **Variant A — Exact / foldable (safe).** Do not rescale at the add. Choose the
  branch *output-projection* scales (`W_O` for attention, `W_down` for MLP) by a
  branch-energy-balancing criterion, and fold the inverse into the next
  `LN-γ → Linear` boundary, where it is exact. The FP model stays
  bit-equivalent; only the quantization grid changes. This is a strict
  generalization of SmoothQuant/OS+ with the scale objective being inter-branch
  balance rather than max-activation matching.

- **Variant B — Aggressive / calibrated-inverse.** Literally rebalance `x` and
  `f` at the add with separate diagonals and absorb the inverse across LN with a
  *calibrated* (not exact) correction. Report the residual inversion error
  explicitly as a metric.

The paper's strength is A-vs-B head-to-head: how much accuracy does giving up
exactness buy, and is it worth the backend cost.

**Architecture caveat.** BERT is **post-norm** (`LN(x + sublayer)`): the add
feeds directly into LN, so the foldable boundary sits in a different place than
in pre-norm ViT/Llama. Folding feasibility differs per family; post-norm and
pre-norm get separate derivations, not one formula.

## Positioning vs baselines

| Method | What it scales | Criterion | Exact? |
|---|---|---|---|
| Plain PTQ (per-tensor / per-token min-max) | nothing | — | yes |
| PEG (Bondarenko, per-embedding-group) | per-group ranges | grouping | yes |
| Outlier Suppression / OS+ | γ migration + shift | outlier channel | yes |
| SmoothQuant | act ↔ weight | max-activation | yes |
| Selective mixed precision (LLM.int8()-style) | keep outlier channels FP16 | magnitude | yes |
| **RBBQ-A (ours)** | branch out-proj | **inter-branch energy balance** | yes |
| **RBBQ-B (ours)** | both branches at add | energy balance | approx (calibrated) |

Novelty must be defended as a **new scaling criterion targeting the add
structure**, not a new quant format. SmoothQuant is the key baseline to beat;
RBBQ-A reduces to SmoothQuant-like *form*, so the delta is purely the criterion.

## Phased plan with verifiable gates

**Phase 0 — Spec + repro.** Formal method doc (both variants, per-architecture
folding derivations). Reproduce one published SmoothQuant W8A8 number on
BERT-base/SST-2 within ±0.3pp. → *Verify: repro matches paper.*

**Phase 1 — Instrumentation & motivation (GATE).** Hook every residual add;
measure per-channel `E_id`, `E_up`, `r[c]`, and layerwise saturation rate on
BERT-base + ViT-B + Llama-7B with a 512-sample calibration set.
→ *Verify: imbalance plots; if flat, stop and report negative result.*

**Phase 2 — RBBQ-A (exact).** Branch-energy scale selection folded at the
foldable boundary. → *Verify: (a) FP fold bit-equivalent to original
(max abs diff < 1e-5); (b) W8A8 accuracy ≥ SmoothQuant on BERT-base/SST-2.*

**Phase 3 — RBBQ-B (approx) + ablation.** Calibrated cross-norm inverse; log
inversion error. → *Verify: A-vs-B ablation; accuracy gain vs inversion-error
tradeoff.*

**Phase 4 — Full matrix.** Run the grid below. → *Verify: success criterion.*

**Phase 5 — Latency.** Confirm RBBQ-A scales are compile-time folded (zero
runtime ops); benchmark end-to-end vs plain W8A8. → *Verify: latency penalty
within target; for B, measure runtime correction cost.*

**Phase 6 — Writeup.**

## Experimental matrix (staged to control compute)

- **Prove-out tier (first):** BERT-base/GLUE+SQuAD, ViT-B/ImageNet,
  Llama-7B/{MMLU, GSM8K, WikiText PPL}.
- **Scale tier (only if prove-out succeeds):** BERT-large, ViT-L, Llama-13B.
- **Quant config:** W8A8 across **two co-headline configs**, run as a full grid
  for every method so the only variable is the scaling criterion:
  - **C1 — static per-tensor** activations + per-tensor weights. The hard
    setting where branch imbalance hurts and outlier methods differentiate;
    this is where RBBQ must prove itself.
  - **C2 — per-token dynamic** activations + per-channel weights. The modern
    strong W8A8; confirms RBBQ does not regress on the easy setting.
  - **Ordering within each tier:** validate C1 first (it gates the method), then
    run C2. C2's compute is only spent once C1 shows signal.

## Metrics

- Task accuracy / EM-F1 / PPL vs FP16-BF16 reference (gap = headline).
- Layerwise **saturation rate** (fraction of activations clipped at the int8
  range) — mechanism evidence.
- **Residual cosine similarity**: cos(FP16 residual output, quantized residual
  output) per layer — tracks error accumulation down the stream.
- End-to-end latency / throughput.

## Success criterion

Consistent **sub-0.5 pp average gap** from FP16/BF16 at W8A8 on **≥2 model
families**, with only a small latency penalty. RBBQ-A must hit this with **zero**
runtime overhead (everything folded) to be compelling; RBBQ-B is justified only
if it adds accuracy beyond A.

## Risks & mitigations

- **Inexact folding across norm** → Variant A sidesteps it; B quantifies it.
  Make the inversion error a reported metric, don't paper over it.
- **Backend inconsistency for the transformed graph** → restrict A to folds that
  compile away; verify the fused/exported graph is numerically identical
  pre/post fold on the actual inference backend before trusting accuracy.
- **Post-norm vs pre-norm** → separate derivations; BERT may need a different
  fold site than ViT/Llama.
- **Compute blowup** (ImageNet + 13B + GLUE) → tiered scope; negative-result
  gate at Phase 1.

## Resolved decisions

1. **Scope — separate track, new tooling.** RBBQ is full-model W8A8 (weights +
   activations across all linears), distinct from the repo's KV-cache fake-quant
   infra. Build on SmoothQuant/Brevitas for the quant + folding machinery,
   `lm-eval-harness` for MMLU/GSM8K/perplexity, HuggingFace for GLUE/SQuAD, and
   `timm` for ViT/ImageNet. Keeps baselines fair and avoids forcing
   KV-cache-specific code to do something it wasn't built for.
2. **Quant config — both, full grid.** Co-headline C1 (static per-tensor) and C2
   (per-token dynamic), per the Experimental matrix above. C1 gates the method;
   C2 confirms no regression on the easy setting.

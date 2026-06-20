# RBBQ — Formal Method & Folding Derivations

Companion to `rbbq_proposal.md`. This document pins down the algebra that
decides what "equalize before the add, invert afterward where algebraically
possible" actually means, for pre-norm (ViT/Llama) and post-norm (BERT)
architectures. It is the Phase 0 spec deliverable.

## Notation

- Hidden dim `d`. Per-token activation is a row vector `x ∈ R^d`; a batch is
  `X ∈ R^{T×d}`. Channel index `c`, token index `t`.
- Linear: `Lin_W(X) = X Wᵀ + b`, with `W ∈ R^{out×in}`.
- LayerNorm with affine `(γ, β)`: `LN(x) = z(x) ⊙ γ + β`, where
  `z(x) = (x − μ(x)) / σ(x)` and `μ, σ` are per-token mean/std over channels.
- RMSNorm: `RMS(x) = (x / √(mean_c x_c² + ε)) ⊙ γ`.
- `D = diag(d_1..d_d)`, `S = diag(s_1..s_d)` are per-channel diagonal scales.

## Three folding identities

**I1 — Linear input scaling (exact).** For any diagonal `S`,
```
(X S)(S⁻¹ Wᵀ) = X Wᵀ.
```
A per-input-channel scale on activations is absorbed exactly into the weight
columns. (This is the SmoothQuant move: shift difficulty from activation to
weight.)

**I2 — Affine-norm output scaling (exact).** Scaling an LN output per channel by
`s` is identical to rescaling its affine parameters:
```
s ⊙ LN(x) = z(x) ⊙ (γ ⊙ s) + (β ⊙ s).
```
So a per-channel scale living on the `LN → Linear` boundary can be pushed **into
LN's `γ, β`** (I2) or **into the next Linear's weights** (I1). Both exact. This
boundary — between an affine-norm output and a linear input — is the only place
a per-channel scale folds for free. Call it the **foldable boundary**.

**I3 — Normalization does not commute with a per-channel diagonal (the wall).**
```
RMS(x ⊙ s) = (x ⊙ s) / √(mean_c (x_c s_c)²)   ≠   s' ⊙ RMS(x)
```
for any fixed diagonal `s'` independent of `x`, because the denominator mixes
channels. Only a **global scalar** `α` commutes (`RMS(αx) = RMS(x)`; LN is also
invariant to global scale after affine). LayerNorm's mean-subtraction and
std-division mix channels the same way. **Consequence:** a per-channel scale
applied to the *input* of a norm cannot be moved to its *output* exactly.

## Where the residual add sits, and why inversion is only partial

Pre-norm block:
```
a = x + Attn(LN1(x))           u_attn = Attn(LN1(x)) = A · W_Oᵀ
y = a + MLP(LN2(a))            u_mlp  = MLP(LN2(a))  = G · W_downᵀ
```
The update branch always ends in a Linear (`W_O` for attention, `W_down` for
MLP). The identity branch is the running stream `x` = embedding + Σ prior branch
outputs.

Suppose we want to rebalance the two branches at the add with a per-channel `D`,
moving the stream into a rescaled basis `ã = D⁻¹ a = D⁻¹x + D⁻¹u`:

- **Write side (exact).** `D⁻¹u` folds into the branch's final linear by I1:
  `u' = G (D⁻¹ W_down)ᵀ = D⁻¹ u`. Free. Likewise `W_O`, and the token embedding
  for the very first contribution. So *producing* the stream in basis `D⁻¹` is
  exact and zero-cost.
- **Read side (the wall).** Every consumer of the stream is a norm
  (`LN1/LN2/RMSNorm`). To use the stream it must invert the basis:
  `LN(D · ã)` should equal `LN(x)`. By I3 this is impossible for per-channel `D`.
  The norm input cannot un-see a per-channel scale.

This is the exact content of "invert where algebraically possible": **exact on
the write side, impossible on the read side at each norm.** It forces two
variants.

## Variant A — exact (foldable boundary, branch-aware criterion)

Do **not** change the residual basis. Place the scale only at the foldable
`LN → Linear` boundary, where I1/I2 hold exactly, so the FP model is
bit-identical and only the quantization grid of that linear's input moves. This
is structurally SmoothQuant; the contribution is the **scale criterion**.

For the linear consuming `LN(stream)`, SmoothQuant picks the per-channel smooth
scale from activation magnitude, `s_c ∝ max_t |LN(stream)_{t,c}|^α / max|W|^{1−α}`.
RBBQ-A instead decomposes the stream channel energy into identity vs update
contributions measured *at the add*:
```
E_id[c] = E_t[ x_c² ],   E_up[c] = E_t[ u_c² ],   r[c] = E_id[c] / E_up[c].
```
and biases `s_c` to protect channels where the **update** branch carries the
task signal but the **identity** branch dominates magnitude (large `r[c]`) — the
channels plain magnitude-smoothing under-weights. Exact functional equivalence
is inherited from I1/I2; nothing here crosses a norm.

This is the safe headline method. Its bar: beat SmoothQuant at equal folding
cost and zero runtime overhead.

## Variant B — approximate (rebalance at the add, calibrated inverse)

Literally rebalance `x` and `u` at the add with per-channel `D` chosen to
equalize `E_id` and `E_up` (e.g. `d_c = (E_id[c]/E_up[c])^{1/2}` applied to the
update branch, or a symmetric split). Fold `D` exactly on the write side (I1).
On the read side, absorb `D` into the following norm's `γ` (`γ → γ ⊙ D`) and
**accept the changed normalization denominator** — the residual of I3. Report
the inversion error explicitly:
```
ε_inv = || LN_with_Dγ(D·ã) − LN(x) ||  (per layer, over calibration tokens).
```
B can balance where A only approximates, at the cost of exactness; it is
justified only if it buys accuracy beyond A net of `ε_inv`.

## Post-norm (BERT) — same foldability, different measurement point

BERT encoder layer:
```
h1  = LN_attn(x + SelfAttn(x))         # add THEN norm
out = LN_out (h1 + FFN(h1))            # add THEN norm
```
The adds feed **directly into a norm**. Implications:

- **Branch energy is measured at the norm input** (`x + SelfAttn(x)` and
  `h1 + FFN(h1)`) — same `E_id/E_up/r[c]` definitions, just located pre-LN.
- **Foldable boundary is unchanged.** `LN_attn → Linear1(FFN intermediate)` and
  `LN_out → next layer's Q/K/V` are both affine-norm → linear, so I1/I2 apply
  exactly. Variant A works identically.
- **Variant B hits the wall sooner** (the norm is immediately after the add),
  but the algebra of the calibrated `γ`-absorption is the same.

So the only architecture-specific piece is *where we hook to measure branch
energy*, not the folding feasibility. ViT-B (pre-norm) follows the pre-norm
derivation; BERT follows post-norm; Llama is pre-norm with RMSNorm (no mean
subtraction, no `β`) — I1/I2/I3 all still hold with `β = 0`.

## What this buys the experiment design

- Variant A is the falsifiable headline: same graph as SmoothQuant, so any
  accuracy delta is attributable purely to the branch-energy criterion, and
  latency is identical (everything compile-time folded).
- Variant B quantifies the price of exactness via `ε_inv` — a first-class
  reported metric, not a swept-under detail.
- The Phase 1 gate (`r[c]` heavy-tailed) is exactly the quantity both variants
  consume, so Phase 1 instrumentation directly feeds Phases 2–3.

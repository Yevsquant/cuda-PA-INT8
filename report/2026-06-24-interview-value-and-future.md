# Codebase value (interview + résumé) and future directions

**Date:** 2026-06-24 · **Repo:** `cuda-PA-INT8` · **Branch:** `feat/rbbq-proposal`
**Hardware:** kernels/bridge/spec-decode on A100-80GB; RBBQ quantization research on H200.

All numbers below are traceable to a report in `report/`. Where a claim is *argued* rather
than *measured*, it is labeled as such. Read this as "what can I defend under questioning,"
not "what sounds good."

The repo is three separable bodies of work:

1. **Track A** — a from-scratch CUDA PagedAttention decode kernel (5 optimization stages) with
   per-token INT8 KV-cache quantization (inline fused dequant).
2. **Track B** — a vLLM speculative-decoding benchmark study (n-gram; FP8 vs BF16 KV).
3. **RBBQ arc (Phases 0–16)** — a W8A8 quantization investigation: a hypothesis refuted with
   mechanism, then a constructive method (selective-granularity W8A8) validated on 3 model
   families, with an honest format×hardware latency framing.

---

## Part 1 — What the value is

The work spans three layers that are usually owned by different people: low-level CUDA
(memory roofline, vectorized loads, warp reductions, split-K), ML-systems (vLLM, paged KV,
speculative decode), and quantization research (falsifiable hypothesis, ablation, a reported
negative result). It is reproducible end-to-end and the claims are caveated rather than
rounded up.

The honest framing of the negative result is itself a selling point: RBBQ's branch-rebalancing
fix was refuted with a mechanism, and replaced with a cheaper method. Being able to show *why*
an idea failed is stronger evidence of judgment than a polished win would be.

What this is **not**: the INT8 bridge is a standalone batch=1 harness, not a vLLM integration;
the spec-decode study is small-scale and directional; the INT8-on-A100 latency advantage is
argued from the hardware, not yet measured on an A100. These bound every claim below.

---

## Part 2 — The three stories, with precise numbers

### Story A — CUDA PagedAttention decode kernel + INT8 KV

**What:** five stages, each correctness-checked against a PyTorch reference (atol/rtol 2e-2)
before timing — naive → vectorized (`float4`/128-bit) → online softmax → warp reduction
(`__shfl_xor_sync`) → split-K (Flash-Decoding). Two INT8 variants (`warp_int8`, `splitk_int8`)
use per-token dynamic symmetric quant with **inline fused dequant**: the per-token scale
factors out of the QK dot product (`dot(q, k_int8·s_k) = s_k·dot(q, k_int8)`), so there is no
separate dequant pass. GQA/MQA tested (8/8, GQA-4×, 8/1, 16/4).

**Numbers (single A100-80GB; HBM peak taken as 1555 GB/s):**
- **1591 GB/s effective bandwidth at batch=64/ctx=4096 (warp kernel)** — 102% of the 1555 GB/s
  byte-model peak. The >100% is a byte-model / cache artifact, not a physics violation (vLLM's
  own kernels show the same); honest reading: "saturates HBM." split-K ≈ 1332 GB/s (86%) at
  batch=32/ctx=4096. Clears the project's 70%-of-peak target.
- **≈100–115% of vLLM `paged_attention_v1`** at moderate batch. In the low-batch
  latency-bound regime (batch=1/ctx=4096) split-K is **11× over naive** (16 → 195 GB/s) and
  1.23× vLLM v1. Which variant wins is regime-dependent — split-K at low batch, warp/online
  at saturation — which is why all variants are kept (mirrors vLLM's v1/v2 split).
- **INT8 KV = 0.516× the FP16 cache** end-to-end on Qwen2.5-1.5B (a **48.4% reduction**;
  int8 data + fp32 per-token scales, approaching but not reaching the 50% floor).
- **Quality:** in the end-to-end bridge, perplexity **1.438 (int8 per-token) vs 1.444 (fp16)**
  — *measured on the model's own FP16 greedy continuation* (batch=1, 5 prompts × 64 tokens),
  so it is a self-consistency metric, not WikiText PPL. Greedy-token match vs fp16 is lower
  (0.662) but that is an argmax-cascade artifact over long greedy runs; perplexity is the
  chosen quality metric. At the kernel level, per-token mean rel-err vs FP16 is consistently
  below per-tensor (the `max` column is noise-dominated by near-zero denominators).

**Debugging example worth telling:** the FP16 trust-gate (bridge must reproduce HF greedy)
initially failed. Rather than assume a kernel bug, it was localized by a kernel↔reference swap
plus a teacher-forced per-step logit comparison; root cause was that **Qwen's
`generation_config` applies `repetition_penalty=1.1` even under `do_sample=False`**, so the
*reference* wasn't pure greedy. The kernel, cache, and loop were correct.

### Story B — vLLM speculative-decoding study

**What:** a resumable matrix on vLLM 0.19.1 (V1): method (baseline / n-gram) × KV dtype
(auto / fp8) × num-spec-tokens (2/4/8) × concurrency (1/8/32) × dataset (sharegpt / humaneval),
online `vllm bench serve`, median of ≤3 runs/cell. **48/60 cells** (EAGLE-3 excluded — no valid
Qwen2.5-1.5B head exists).

**What the data shows (this model/hardware only; directional):**
- **n-gram acceptance ≈ 11–41%, falling as spec length grows** (e.g. sharegpt auto: 30.5% @
  spec_tok=2 → 13.8% @ spec_tok=8).
- **Speculation can reduce throughput when the batch is already busy:** at concurrency 8
  (sharegpt) baseline is 1778 tok/s; n-gram auto is 1601 / 1691 / 1766 across spec_tok 2/4/8 —
  all below baseline. At concurrency 1 it is roughly neutral-to-positive.
- **FP8 KV raised acceptance but lowered throughput and raised TPOT** (sharegpt conc-1,
  spec_tok=2: acc 30.5%→40.9%, tok/s 221.8→203.4, TPOT 4.92→5.64 ms).
- Ran inside an **8 GiB host-RAM** pod: one server resident at a time, disk checkpointing,
  streaming aggregation → resumable after interruption. Fixed real build/runtime issues:
  nvcc-parallelism OOM (`MAX_JOBS` cap), stale JIT build locks, and a flashinfer FP8 kernel
  **link** failure (`-lcuda` unresolved by the conda linker → CUDA driver stub on
  `LIBRARY_PATH`).
- **Caveat:** "GPU MB" per cell is vLLM's steady server allocation (≈0.9 reservation), not a
  true per-config delta.

### Story C — RBBQ arc (Phases 0–16)

**Question:** why does static W8A8 (8-bit weight + 8-bit activation) PTQ break Transformers,
and what is the cheapest fix?

1. **Hypothesis (RBBQ):** PTQ breaks at the residual add — identity branch `x` and update
   branch `F(x)` have mismatched per-channel energy; fold a branch rescaling to equalize it.
2. **Gate passes (Phase 1):** the imbalance is real and heavy-tailed (channel dominance
   10²–10⁵× the median), localized to the MLP residual add, on BERT and Qwen.
3. **Refutation (Phases 2–4):** the fix cannot work. The damage is **activation-side**, at the
   **non-foldable MLP down-projection** (`fc2`/`down_proj`), downstream of GELU/SwiGLU — shown
   by a mixed-precision oracle and a weight-vs-activation decomposition (BERT: quant-act
   +8.37 pp vs quant-weight +0.60 pp). Branch rebalancing is weight-side, so it cannot reach
   it. Refuted on both families.
4. **Constructive method (Phases 5–8): selective-granularity W8A8.** A cheap calibration
   statistic `D = max_token(rowmax)/median_token(rowmax)` flags the linears needing per-token
   dynamic quant. **Qwen (WikiText-2, FP ppl 9.803): 32/196 linears dynamic (16%) → ppl 10.57**
   vs plain static **110.4** and all-dynamic **10.02** — ~99% of the gap recovered, 84% of
   linears left on static per-tensor. *On BERT the rule is weaker/directional* (outliers more
   distributed; ~28–37/72 linears needed), an honest limit of `D`.
5. **Methodology (Phases 6, 8):** a local quant-error metric `E` is *worse* than `D` (it picks
   qkv, whose error softmax absorbs) — end-to-end sensitivity ≠ local quant error, and `D` is
   also the cheapest signal (an input reduction, no matmul).
6. **Latency, honest (Phases 7, 9, 10):** per-token dynamic is **free when fused** (Triton
   kernels validated bit-close to eager), so selective is an accuracy/memory method, **not** a
   speed trick. On **H200**, a tuned Triton int8 GEMM is **4.1× slower than bf16** (`_int_mm`
   7.3×), while fp8 `_scaled_mm` is **1.6× faster** — so the int8 latency win is hardware-bound.
7. **Generality (Phase 11):** transfers to **Mistral-7B** (down_proj +139.9 ppl → +0.16 with
   per-token dynamic). The pattern now spans post-norm encoder (BERT), pre-norm decoder
   (Qwen-1.5B, Mistral-7B), GELU/SwiGLU, 0.1B–7B.
8. **Rescue ablations (Phases 14–15):** MSE-optimal *static* clip at down_proj recovers ~89%
   of the gap (110.4 → 20.5) but cannot match dynamic (10.67) — direct proof the range is
   genuinely per-token. Routing only **~0.6% of tokens** (the outliers) to FP16 reaches
   **11.64 ppl** — the damage is concentrated in a tiny token fraction.
9. **Format×hardware close (Phase 16):** **FP8 (e4m3) all-static is near-lossless (+0.09 ppl)
   and needs no selectivity** — floating-point relative precision absorbs the down_proj
   outliers that wreck INT8 static (110.4). Conclusion: **selective-granularity is specifically
   the INT8/A100 answer; on Hopper, FP8 all-static is simpler and near-lossless.**

**The defensible one-liner:** "I refuted my own quantization hypothesis with a mechanism, then
built a cheaper method — a one-statistic rule that makes 16% of linears dynamic and recovers
~99% of the W8A8 accuracy gap on Qwen — and showed it's an INT8-specific need that FP8
dissolves."

---

## Part 3 — Résumé bullets (grounded; pick by role)

**CUDA / performance**
- Built a from-scratch CUDA PagedAttention decode kernel across five optimization stages
  (128-bit vectorized loads, online softmax, warp-shuffle reduction, split-K), reaching
  **1591 GB/s (~HBM-saturating) at batch=64** and **100–115% of vLLM `paged_attention_v1`**;
  every stage validated against a PyTorch reference.
- Added per-token INT8 KV-cache quantization with **inline fused dequant**, cutting KV memory
  to **0.516× FP16 (−48%)** with perplexity unchanged on the FP16 continuation (1.438 vs
  1.444) for Qwen2.5-1.5B; full GQA/MQA support.

**ML-systems / inference**
- Ran a resumable 48-cell vLLM speculative-decoding matrix (n-gram; FP8 vs BF16 KV;
  TTFT/TPOT/throughput/acceptance), showing low-acceptance speculation **reduces** throughput
  on compute-bound batches and that FP8 KV trades higher acceptance for higher TPOT.
- Built the pipeline to run in an **8 GiB host-RAM** pod (sequential servers, disk
  checkpointing, streaming aggregation) and fixed CUDA/vLLM build failures (nvcc OOM, stale JIT
  locks, a flashinfer FP8 **link** failure fixed via the CUDA driver stub).

**Quantization / applied research**
- Investigated static W8A8 PTQ failure: **refuted a residual-branch-rebalancing hypothesis
  with mechanism** (damage is activation-side at the non-foldable MLP down-projection), then
  built **selective-granularity W8A8** — a per-token-spread statistic selects the **16% of
  linears** needing dynamic quant, recovering **~99% of the gap (110.4 → 10.6 ppl, FP 9.8)** on
  Qwen with 84% of linears static.
- Validated across **BERT, Qwen2.5-1.5B, Mistral-7B**, and mapped the format×hardware tradeoff
  (INT8→A100; **FP8 all-static near-lossless on Hopper, +0.09 ppl**); fused Triton kernels show
  dynamic quant is free when fused, so the method is accuracy/memory, not a speedup.

---

## Part 4 — Questions this work prepares you for

- *Optimizing a memory-bound kernel* → the roofline ladder; why split-K wins at low batch and
  warp at saturation.
- *A bug you debugged* → the `repetition_penalty`-under-greedy trust-gate (root cause in the
  reference, not the kernel).
- *A time you were wrong* → the RBBQ refutation, with the oracle + weight/activation decomposition.
- *How you choose what to quantize* → `D` vs the local-error metric `E`; sensitivity ≠ local error.
- *INT8 vs FP8* → fixed-point vs floating-point relative precision; why the choice is hardware-bound.

Be ready to volunteer the caveats (batch=1 bridge, small spec-decode study, A100 latency
unmeasured) — stating them first is more convincing than defending them under follow-up.

---

## Part 5 — Future of the codebase

Ordered by value-to-effort. **(blocked)** = needs hardware/data not on the current box.

**Near-term, high-value**
1. **A100 INT8-GEMM latency confirmation (blocked on A100).** The int8→A100 win is currently
   argued and measured only on H200 (where int8 loses). The Phase 9/10 Triton kernels are
   portable; one A100 run converts the single biggest open claim from argued to measured.
2. **Fused write-quant kernel for the INT8 KV path.** The decode write-path quant is currently
   torch, so the bridge's INT8 *latency* is conservative; a fused write-quant kernel makes it a
   latency story, not only a memory story.
3. **FP8 KV-cache kernel variant** for a direct comparison against vLLM's FP8 KV (pairs the
   Track-A kernel with the Phase-16 FP8 finding).

**Medium-term (research completeness)**
4. **ViT-B (vision) generality (blocked on ImageNet).** The one untested architecture class; a
   cached CLIP/timm ViT could give a partial `D`-stat + localization signal sooner.
5. **Full baseline matrix** vs SmoothQuant / Outlier Suppression / per-tensor / all-dynamic on
   Qwen + more tasks (GLUE/SQuAD/MMLU/GSM8K); SmoothQuant is currently compared mainly on BERT.
6. **Group-aware `D` threshold** (Phase 11 noted it can pull in some `mlp_in` linears) to make
   the selection fully automatic.

**Longer-term / upstream**
7. **vLLM integration of the INT8 KV path** — what RFC #37319 proposes; the inline-dequant read
   path + a fused write-quant kernel is a concrete starting PR and the highest-visibility outcome.
8. **INT8-Q** (quantize the query) and **split-K `num_splits` auto-tuning** (currently fixed
   PARTITION_SIZE=512 vs vLLM v2's context/batch-aware count) to close the ~56%-occupancy gap.
9. **Write up the RBBQ arc** (refutation + selective-granularity + format×hardware) as a short
   paper or blog post — it is a self-contained negative-plus-constructive result.

**Next step in one line:** get on an A100, turn the int8 latency claim from argued to measured,
write the fused write-quant kernel, then upstream the INT8 KV path against vLLM RFC #37319.

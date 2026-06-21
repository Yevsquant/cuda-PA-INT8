Assumption: by “remaining the original proposal,” you mean preserve the RBBQ story as a residual-stream W8A8 proposal, not abandon it for an unrelated quantization project. Under that constraint, do **not** claim branch rebalancing itself works. The repo evidence says it does not: the damage is activation-side at `fc2/down_proj`, not fixable by folded branch rescaling ([rbbq_capstone.md](/Users/runyingchen/UIUC/003/cuda-PA-INT8/report/rbbq_capstone.md:46)).

**Recommended framing:** keep RBBQ as a falsifiable residual-diagnosis method, then replace the actuator.

1. **RBBQ-C: Residual-Guided Selective Granularity**

Keep the original residual-add instrumentation: measure `E_id`, `E_up`, `r[c]` as proposed ([rbbq_proposal.md](/Users/runyingchen/UIUC/003/cuda-PA-INT8/report/rbbq_proposal.md:105)). But instead of forcing diagonal branch rebalancing, use the residual analysis to identify which branch output poisons W8A8, then apply selective activation granularity.

Method:
- Static per-tensor W8A8 everywhere by default.
- For each linear input, compute `D = max_token(rowmax) / median_token(rowmax)`.
- Use per-token dynamic activation quant only when `D >= threshold`.
- Prefer group-aware policy: always consider `fc2/down_proj/mlp_out` first, because the repo shows that is the causal site across BERT, Qwen, and Mistral.

This already has strong evidence: Qwen recovers near all-dynamic with only 32/196 dynamic linears ([phase5_report.md](/Users/runyingchen/UIUC/003/cuda-PA-INT8/report/phase5_report.md:52)), and Mistral-7B confirms the same mechanism ([phase11_report.md](/Users/runyingchen/UIUC/003/cuda-PA-INT8/report/phase11_report.md:31)).

2. **RBBQ-A+Selective: Exact Folds Where Legal, Dynamic Where Necessary**

Preserve Variant A exactly as originally written: only fold scales at legal `LN -> Linear` boundaries ([rbbq_proposal.md](/Users/runyingchen/UIUC/003/cuda-PA-INT8/report/rbbq_proposal.md:62)). But make it a component, not the whole method.

Pipeline:
- Apply SmoothQuant/RBBQ-A style exact folding on foldable linears: `q/k/v`, `fc1`, `gate/up`.
- Do **not** try to repair `down_proj` through cross-norm inverse.
- Use per-token dynamic activation quant at `fc2/down_proj`.
- Report the ablation: exact folds alone vs exact folds + selective dynamic.

This keeps the original algebra discipline and avoids contradicting the negative result.

3. **Residual-Aware Static Scale Search for the C1 Setting**

The original proposal cares about hard C1: static per-tensor activations ([rbbq_proposal.md](/Users/runyingchen/UIUC/003/cuda-PA-INT8/report/rbbq_proposal.md:131)). Since max-scale static quant fails because one token sets the range, try a constrained static alternative before giving up C1 entirely.

Method:
- For selected `down_proj/fc2` inputs, choose activation scale by minimizing downstream residual error, not raw local quant error.
- Candidate objectives:
  - residual cosine after the add,
  - output PPL/accuracy on a tiny calibration set,
  - static-vs-dynamic mismatch minimized over calibration tokens,
  - percentile/MSE clipping with residual-stream validation.
- Keep one static scale per layer or per group, so it remains deployable as C1.

This is weaker than per-token dynamic, but it gives the proposal a serious “static rescue” attempt.

4. **Token-Outlier Fallback Instead of Fully Dynamic Linears**

A middle ground between C1 and all-dynamic:

- Use static per-tensor activation quant for most tokens.
- If a token’s rowmax exceeds `k * median_rowmax`, quantize that token dynamically or route it through BF16/FP16 activation.
- Apply only to `fc2/down_proj` families.

This remains close to the original proposal because it targets the residual-branch output where the failure occurs, but it handles the actual per-token granularity problem.

5. **Hardware-Aware Final Method**

Do not sell selective granularity as a latency trick. The repo already says fused per-token dynamic is roughly as cheap as static, while the real latency variable is format and hardware ([rbbq_capstone.md](/Users/runyingchen/UIUC/003/cuda-PA-INT8/report/rbbq_capstone.md:69)).

Final deployment story:
- A100: INT8 W8A8 selective path.
- Hopper/H200: FP8 selective path.
- Both: same `D` selection rule.
- Validate A100 because current H200 INT8 GEMM numbers are unfavorable ([phase10_report.md](/Users/runyingchen/UIUC/003/cuda-PA-INT8/report/phase10_report.md:44)).

**What I would write as the revised claim**

“RBBQ’s original branch-rebalancing actuator is refuted, but its residual-stream premise survives: residual-add imbalance localizes the W8A8 failure to MLP output projections. The corrected method is Residual-Guided Selective-Granularity W8A8: use residual diagnostics plus input per-token spread `D` to select the branch-output linears requiring per-token dynamic activation quantization, while retaining static W8A8 elsewhere and exact folding where algebraically legal.”

That keeps the proposal’s motivation, instrumentation, W8A8 scope, architecture comparison, and success criteria, while avoiding the false claim that residual-branch rebalancing fixes the problem.
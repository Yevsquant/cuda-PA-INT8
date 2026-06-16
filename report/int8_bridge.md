# INT8 KV-cache end-to-end bridge — Qwen2.5-1.5B-Instruct

One honest end-to-end data point for the custom per-token INT8 KV-cache kernel on real Qwen weights. A standalone harness (not vLLM) routes Qwen2's **decode-step** attention through the custom split-K paged-decode kernels; prefill runs in bf16 SDPA. Three cache variants share the *identical* prefill+greedy-decode loop — only the decode-step KV dtype/kernel differ — so the numbers below isolate quantization. Each variant ran in its own process (8 GB host-RAM rule); fp16 is the reference.

- Model: `Qwen/Qwen2.5-1.5B-Instruct` (bf16), batch=1, single sequence, contiguous blocks, greedy.
- Decode length: 64 tokens/prompt; 5 prompts (chat + code).
- Kernel: `paged_decode_attn_splitk{,_int8}`, head_dim 128, GQA-6 (12 Q / 2 KV).

## Results (medians across prompts)

| variant | TTFT (ms) | TPOT (ms) | greedy-match vs fp16 | ppl (fp16 cont.) | KV @2048 tok (MiB) | KV ratio |
|---|---|---|---|---|---|---|
| fp16 | 28.5 | 28.97 | 1.000 | 1.444 | 56.0 | 1.000 |
| int8_per_token | 36.8 | 34.80 | 0.662 | 1.438 | 28.9 | 0.516 |
| int8_per_tensor | 37.0 | 31.53 | 0.609 | 1.439 | 28.9 | 0.516 |

## Headline

- **Memory:** INT8 (per-token) KV cache is **0.516×** the FP16 cache at 2048 tokens — a **48.4%** reduction (int8 data + fp32 per-token scales). This number is real and independent of the kernel's latency.
- **Precision:** per-token greedy-match vs fp16 = 0.662; per-tensor = 0.609. Per-token ppl 1.438 vs fp16 1.444.

## Caveats (do not hide these)

- **Not in vLLM.** Standalone harness; not comparable to the §1–§5 serving matrix; batch=1 latency only.
- **No fused write kernel.** Per-step KV quantization is torch on the write path, which inflates TPOT — the INT8 latency here is *conservative* vs an idealized fused-write impl. The memory number is the real headline.
- **Q is fp16 at the kernel boundary** (kernel is `__half`), not bf16 — a minor precision caveat.
- **Prefill runs in bf16/SDPA**; only the decode path exercises the custom kernel. The fp16 variant matches HF greedy token-for-token (test gate 2), validating the cache+loop before INT8.

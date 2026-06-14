# INT8 KV-cache quantization — kernel-level results

Hardware: NVIDIA A100-SXM4-80GB. Peak HBM bandwidth assumed 1555 GB/s.

INT8 KV cache with **per-token** dynamic symmetric quantization (`s = max(|x|)/127`, one fp32 scale per token×kv-head, constant across head_dim). Dequant is fused inline on KV reads: the per-token scale factors out of the dot product (`dot(q, k_int8·s_k) = s_k·dot(q, k_int8)`) and folds into the per-token softmax weight for V — no per-element multiply. **Per-tensor** static quant (one scalar per K and V tensor) is the ablation baseline.

Two INT8 kernels: `warp_int8` (Stage 5, from the FP16 warp kernel) and `splitk_int8` (Stage 6, Flash-Decoding split-K). Both validated against a dequant reference within 2e-2 across GQA / context / batch.

## Performance and memory

GB/s counts DRAM bytes at the variant's element width (INT8 = half of FP16). `% KV mem` is the resident KV-cache size vs FP16 — INT8 stores 1 byte/element plus a small fp32 per-token scale stream, so it lands just above the 50% floor (the excess shrinks toward 50% as head_dim/kv-head grows). `rel-err vs fp16` is the mean relative error of the INT8 kernel output vs the FP16 `warp` kernel.

#### batch=8, ctx=2048, heads=8/8 (GQA 1x)

| variant | µs | GB/s | % peak BW | KV MB | % KV mem | rel-err vs fp16 |
|---|---:|---:|---:|---:|---:|---:|
| warp | 245.0 | 274 | 18% | 67.1 | 100% | — |
| splitk | 124.8 | 538 | 35% | 67.1 | 100% | — |
| warp_int8 | 240.4 | 140 | 9% | 34.6 | 52% | 0.0277 |
| splitk_int8 | 121.6 | 276 | 18% | 34.6 | 52% | 0.0277 |

#### batch=32, ctx=2048, heads=8/8 (GQA 1x)

| variant | µs | GB/s | % peak BW | KV MB | % KV mem | rel-err vs fp16 |
|---|---:|---:|---:|---:|---:|---:|
| warp | 277.7 | 967 | 62% | 268.4 | 100% | — |
| splitk | 254.8 | 1054 | 68% | 268.4 | 100% | — |
| warp_int8 | 269.0 | 499 | 32% | 138.4 | 52% | 0.0274 |
| splitk_int8 | 199.6 | 673 | 43% | 138.4 | 52% | 0.0274 |

#### batch=8, ctx=4096, heads=8/1 (GQA 8x)

| variant | µs | GB/s | % peak BW | KV MB | % KV mem | rel-err vs fp16 |
|---|---:|---:|---:|---:|---:|---:|
| warp | 296.4 | 57 | 4% | 16.8 | 100% | — |
| splitk | 102.6 | 164 | 11% | 16.8 | 100% | — |
| warp_int8 | 264.5 | 32 | 2% | 8.7 | 52% | 0.0247 |
| splitk_int8 | 104.9 | 80 | 5% | 8.7 | 52% | 0.0247 |

#### batch=32, ctx=4096, heads=16/4 (GQA 4x)

| variant | µs | GB/s | % peak BW | KV MB | % KV mem | rel-err vs fp16 |
|---|---:|---:|---:|---:|---:|---:|
| warp | 523.5 | 513 | 33% | 268.4 | 100% | — |
| splitk | 438.6 | 612 | 39% | 268.4 | 100% | — |
| warp_int8 | 487.0 | 276 | 18% | 138.4 | 52% | 0.0253 |
| splitk_int8 | 393.6 | 341 | 22% | 138.4 | 52% | 0.0253 |

## Kernel-level precision (per-token vs per-tensor ablation)

Max / mean relative error of the INT8 kernel output vs the FP16 kernel output, on random N(0,1) K/V. The `max` is inflated by a few near-zero output elements in the denominator; the `mean` is the representative metric. Per-token quant beats per-tensor on both — the headline ablation result.

| config | per-token max | per-token mean | per-tensor max | per-tensor mean |
|---|---:|---:|---:|---:|
| ctx=128, 8/8 | 2.4261 | 0.0365 | 2.6207 | 0.0491 |
| ctx=500, 8/8 | 1.1912 | 0.0292 | 2.1629 | 0.0531 |
| ctx=500, 8/2 | 1.1582 | 0.0301 | 2.2581 | 0.0505 |
| ctx=2048, 16/4 | 1.0009 | 0.0277 | 1.2929 | 0.0473 |

## Summary

- **Memory:** INT8 KV cache is ~50% of FP16 (1 B/elem + small fp32 per-token scales), confirming the 2× memory reduction.
- **Precision:** per-token dynamic quant gives mean relative error a few percent vs FP16 and is consistently lower than per-tensor static quant — the expected ablation outcome.
- **Correctness:** both INT8 kernels match the dequant reference within 2e-2 across the full GQA / context / batch sweep (incl. long-context split-K).


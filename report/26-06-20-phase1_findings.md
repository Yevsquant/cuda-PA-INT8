# Phase 1 结果：真实分布 vs 合成数据的 fake-quant 误差

> 配套：`benchmarks/dump_kv.py`（抓 post-RoPE Q/K/V）、`benchmarks/fakequant_eval.py`（fake-quant + Phase-0 指标）。数据：Qwen2.5-1.5B-Instruct，6 段真实文本，layers {0,14,27}，单步 decode（最后一个 query token 注意全序列）。

## 结论一句话

合成 N(0,1) 基线**严重低估**了真实分布下的 INT8 退化，且退化**集中在浅层**——那里 post-RoPE K 有极端的离群通道。这正是审稿意见第一条的实锤，也直接给 Phase 2/3（SmoothQuant / Hadamard 压 K 离群通道）提供了动机。

## 数据（per-token 对称 INT8，rel-L2 越低越好）

| layer | K 离群比 (max/median 通道) | 真实 per-token rel-L2 | 合成 per-token rel-L2 | 真实/合成 |
|------:|------:|------:|------:|------:|
| 0  | **83×** | **0.132** | 0.009 | **~15×** |
| 14 | 5.1× | 0.020 | 0.009 | ~2.3× |
| 27 | 6.0× | 0.009 | 0.009 | ~1× |

per-tensor 一律更差（layer 0 真实 0.222），与 Phase 0 的消融结论一致：per-token 优于 per-tensor。

## 怎么读这张表

- **离群比 = K 各通道在序列上的 amax，最大通道 / 中位通道**。>1 说明少数通道独大，就是 post-RoPE 的系统性离群通道签名。
- **layer 0 离群比 83×**：一个通道幅度是中位通道的 80 多倍。对称 per-token 把 scale 撑大，正常通道被压没 → rel-L2 冲到 0.132，是合成数据的 ~15 倍。
- **浅层重、深层轻**：layer 27 离群比只有 6×，真实误差与合成基本持平。说明「3% / 0.9% 的合成基线」对深层尚可，对浅层是过度乐观的。

## 踩到的坑（已修，值得记住）

Qwen2.5 **必须用 bfloat16**，不能 fp16。深层有 massive activations，幅度超过 fp16 的 65504 → inf → 整层 NaN（第一版 dump 在 layer 14/27 全 NaN）。改 `dtype=torch.bfloat16` 后正常。我们的 kernel 是 fp16 存储，但本研究测的是激活**数值分布**，与 kernel 存储精度无关，用模型原生 bf16 才能拿到有效激活。

## 对后续的指向

- Phase 2/3 的 SmoothQuant / Hadamard 应**优先验证浅层**（layer 0 最甜），那里离群最严重、提升空间最大。
- Phase 0 的合成阈值（cos>0.998, rel-L2<0.05）对深层够用；浅层真实 rel-L2 已达 0.13，是接下来要打下来的目标。

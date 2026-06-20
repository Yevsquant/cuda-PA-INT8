# Phase 5 结果：端到端 WikiText-2 困惑度（把 rel-L2 翻译成 PPL）

> 配套：`benchmarks/ppl_eval.py`。Qwen2.5-1.5B-Instruct，bf16，WikiText-2 test 前 60K token，滑窗 max_len=2048/stride=1024。fake-quant 注入点同 Phase 1（`eager_attention_forward` 钩子），把 post-RoPE K/V 换成 quant→dequant。SmoothQuant α=0.85。

## 这是整个项目的「硬证据」

之前都是 rel-L2（kernel 级误差）。审稿意见反复说:「真正该关心的是困惑度」。现在量出来了:

| mode | PPL | ΔPPL |
|---|---:|---:|
| baseline（不量化 KV） | 9.342 | +0.0% |
| INT8 per-tensor | 11.752 | **+25.8%** |
| INT8 per-token（当前对称方案） | 11.070 | **+18.5%** |
| INT8 per-token + **SmoothQuant** | 9.614 | **+2.9%** |
| INT8 per-token + Hadamard | 9.625 | +3.0% |
| INT8 per-token + 非对称 K | 10.266 | +9.9% |

## 三个关键结论

**1. 合成基线严重骗人,真实困惑度退化巨大。** Phase 0 在 N(0,1) 上测的对称 per-token rel-L2 只有 0.9%,看着人畜无害。但端到端,**对称 per-token INT8 KV 让困惑度涨 +18.5%**——这是会被一眼看穿的退化。这彻底坐实了审稿意见的核心:**不在真实分布 + 端到端上测,就会把问题低估一个数量级。**

**2. SmoothQuant 几乎把损失全收回:+18.5% → +2.9%。** 这是项目的主结果。治住 post-RoPE K 的离群通道(Phase 2/3 的发现),直接体现在困惑度上——从「不可用」回到「几乎无损」。Hadamard 同样把损失收回到 +3.0%,与 SmoothQuant 端到端打平。

**3. 非对称单独 +9.9%(把对称的 +18.5% 砍了快一半),但不如通道型方法。** 与 Phase 4 一致:K 的病主要是离群通道(SmoothQuant/Hadamard 治),不是偏斜(非对称治);非对称便宜(kernel +1~2%)但只能算辅助。

## 一个有意思的 nuance:为什么端到端 Hadamard ≈ SmoothQuant?

Phase 3 在 **layer 0 单步 decode** 上,SmoothQuant(rel-L2 0.031)远胜 Hadamard(0.074)。但端到端 PPL 两者几乎相等(+2.9% vs +3.0%)。原因:

- SmoothQuant 的大胜**集中在浅层**(layer 0 离群比 71×),深层增益小。
- Hadamard 是**每-token 旋转,层层均匀起效**。
- PPL 对**所有 28 层所有位置**求和,浅层的单点优势被摊薄,于是两者趋同。

**教训:单层 probe 和全模型指标是两个镜头,不能只看一个下结论。** 这恰好呼应 Phase 0「换更稳的指标」的初衷。

## 工程一致性

fake-quant 路径与 kernel 严格对齐:所有 INT8 mode 的 V 都走 per-token 对称;SmoothQuant/Hadamard 只变换 Q,K(对分数精确无损);非对称仅 K。所以这张 PPL 表能代表 kernel 真实部署的精度。

## 口径说明

- 60K token 子集(全 test 约 245K)。相对 ΔPPL 在子集上稳定,绝对值会随语料长度略动。
- baseline 跑 bf16(模型原生);「不量化 KV」指 KV 不做 INT8。kernel 实际存 fp16,精度结论一致。
- SmoothQuant 在端到端用**动态 per-batch 标定**(用当前序列的 per-channel absmax),无需离线校准集,实现更简。

## 项目全景(Phase 0→5)

| 阶段 | 交付 | 关键数字 |
|---|---|---|
| 0 | cosine/rel-L2 指标 + 离群压力测试 | 干掉「除零噪声」的旧指标 |
| 1 | 真实 post-RoPE K/V dump + fake-quant | layer 0 真实 rel-L2 0.13(合成 0.009 的 15×) |
| 2 | SmoothQuant(不改 kernel) | layer 0 −76% |
| 3 | Hadamard + 对比 | SmoothQuant 胜系统性离群;不叠加 |
| 4 | 非对称 K(唯一 kernel 改动) | +1~2% 开销;SQ 上叠加 +11% |
| 5 | **端到端 PPL** | **对称 +18.5% → SmoothQuant +2.9%** |

## GSM8K(200 题)下游任务准确率

PPL 是语言建模指标;再加一个**下游任务**看量化是否真的答错题。同一个 fake-quant 钩子,batched 贪心生成,`repetition_penalty=1.0`(显式关掉 Qwen 的 1.1 贪心陷阱)。

| mode | GSM8K acc | Δ |
|---|---:|---:|
| baseline | 0.385 | +0.000 |
| INT8 per-token(对称) | 0.305 | **−0.080** |
| INT8 + SmoothQuant | 0.375 | −0.010 |
| INT8 + 非对称 K | 0.375 | −0.010 |

**与 PPL 完全同向:对称 INT8 KV 掉 8 个点(38.5%→30.5%,相对 −21%),SmoothQuant 收回到 baseline 1 点以内。** 非对称在 GSM8K 上也回到 37.5%。

口径与注意:
- n=200,准确率标准误约 ±3.4 点。所以「对称 −8 点」≈ 2.3 个标准误,**显著**;而 SmoothQuant/非对称/baseline 三者 1 点内的差异**在噪声内**——结论是这两种方法都把损失收回到「与 baseline 无统计差异」。
- baseline 38.5% 偏保守:零样本(无 few-shot)+ 模型常不按 `####` 输出 + 「取末位数字」抽取偶尔失手。但所有 mode 用**同一口径**,相对比较有效。
- 加大 n / 加 few-shot 能抬高绝对值并缩小误差棒,但不会改变「对称掉点、SmoothQuant 收回」的结论。

## 两个端到端指标一致

| 指标 | 对称 INT8 | +SmoothQuant |
|---|---|---|
| WikiText-2 PPL | +18.5% | +2.9% |
| GSM8K acc | −8.0 点 | −1.0 点(噪声内) |

困惑度和下游任务**互相印证**:对称 per-token INT8 KV 有真实可测的退化,SmoothQuant 基本无损地修好。这就是整个项目要给审稿人的答案。

# Phase 6 结果:vLLM 定位 + FP8 对照 + INT8 后端集成设计

> 配套:`benchmarks/ppl_eval.py`(加 `fp8_e4m3` mode)、`benchmarks/vllm_kv_eval.py`(vLLM 原生 kv_cache_dtype GSM8K)。前置:`bench_paged_decode.py`(已有的 kernel-vs-vLLM 性能基准,含 INT8)。

## 一句话

vLLM **没有 INT8 KV**(本项目要填的空),它唯一的 sub-fp16 KV 选项是 **FP8-E4M3,而 FP8 在 Qwen2.5-1.5B 上灾难性崩盘**(GSM8K 0.63→0.04)。我们的 **INT8+SmoothQuant 近乎无损**(PPL +2.9%)。两个独立引擎(我们的 HF fake-quant 与 vLLM 原生 kernel)在 FP8 崩盘上**互相印证**。

## 核心证据:FP8-E4M3 KV 崩盘,跨引擎一致

| 引擎 / 方法 | 指标 | 结果 |
|---|---|---|
| **vLLM 原生 kv_cache_dtype=auto**(bf16 KV) | GSM8K acc (n=100) | **0.630** |
| **vLLM 原生 kv_cache_dtype=fp8**(E4M3 KV) | GSM8K acc (n=100) | **0.040** ← 崩盘 |
| 我们的 HF fake-quant `fp8_e4m3`(per-token) | WikiText-2 PPL | **135.4(+1349%)** ← 印证 |
| 我们的 HF fake-quant `int8_per_token` | WikiText-2 PPL | 11.07(+18.5%) |
| 我们的 HF fake-quant `int8_smoothquant` | WikiText-2 PPL | 9.61(+2.9%) |

两条独立证据链(vLLM 真 FP8 kernel 的 GSM8K、我们 HF 的 PPL)都指向同一结论:**FP8-E4M3 KV 对这个小模型是不可用的。**

## 机理:为什么 FP8 比 INT8 还差(KV cache 场景)

直觉反常——FP8 不是「抗离群」吗?但在 **KV cache + softmax** 这个具体场景:

- **E4M3 只有 3 位尾数**,相对误差 ~6%(half-ULP 6.25%),且**对每个元素都是这个相对误差**,无论大小。
- **INT8 per-token** scale = max/127,**最大的那些 K 分量**相对误差 <0.8%;小分量绝对误差小。
- 注意力分数 = Σ q·k,**softmax 对最大的几个 q·k 项指数级敏感**。FP8 在这些主导分量上带 ~6% 误差,被 softmax 放大 → 崩。INT8 恰好把精度**钉在最大分量上**,主导项准 → 稳。
- 验证:Phase 6 实测 **K 走 FP8 单独就让 loss 4.0→4.8;V 走 FP8 几乎无损**(2.31)。问题全在 K——正是 softmax 敏感的那一侧。

> **这是一个支持 INT8 的实证结论**:FP8 的「均匀相对精度」不适合 softmax 敏感的 K;INT8 的「max 锚定精度」更稳。

## 重要 caveat(诚实口径)

- **模型越小越敏感**。社区普遍报告 FP8 KV 在 7B+ 大模型上基本无损。Qwen2.5-1.5B 小,对 KV 精度敏感,所以 FP8 崩得厉害。所以结论是「**对小/边缘模型,INT8 KV 比 FP8 更安全**」,不是「FP8 KV 普遍无用」。
- HF 的 GSM8K baseline(0.385)低于 vLLM(0.630):HF eager + 左 padding 批量贪心是个较弱的 harness,但**各 mode 同口径**,相对比较有效;vLLM 的 0.63 才是该模型的真实水平。
- FP8 sim 用 per-token max→448 的 E4M3,与 int8_per_token 同粒度,是公平的同粒度数据类型对比。

## 性能侧(已有基准)

kernel 级性能对照 vLLM 早已在 `bench_paged_decode.py` 落地:我们的 INT8 v5/v6 与 `vllm._custom_ops.paged_attention_v1/v2`、SDPA 同台,带 DRAM 字节/带宽/KV 显存核算。INT8 变体 KV 显存 ~50%、DRAM 字节减半。性能目标(达到 vLLM kernel 80%+、带宽 70%+)在那份报告里追踪,不在本阶段重复。

## vLLM 原生 INT8 KV 后端:集成设计(future work)

vLLM 没有 INT8 KV 的 hook 点(`kv_cache_dtype` 只有 auto/fp8),这正是 RFC #37319 指出的空白。把本项目的 kernel 接进去需要:

1. **新 cache dtype `int8`**:在 CacheConfig/cache engine 注册;block 按 int8 分配(显存减半),并**并行分配一份 per-token scale cache**(以及非对称时的 zero-point cache),布局对齐 `[num_blocks, num_kv_heads, block_size]`(与我们 reference 的 scale 缓冲一致)。
2. **自定义 AttentionBackend**:实现 vLLM 的 attention backend 接口,decode 路径调用我们的 `paged_attention_v5/v6`(已支持 INT8 + 可选 zero-point)。vLLM 自带 kernel 不支持 per-token int8 scale,所以必须走自定义 backend。
3. **写入端量化 kernel**:在 KV append 时把新 token 的 K/V per-token 量化进 int8 cache(一个轻量 quant kernel)。
4. **SmoothQuant 折叠进权重**:SmoothQuant 对 QK 是 `Q'=Q·s, K'=K/s`。Q、K 来自 `q_proj/k_proj` 线性层,所以把 per-channel `s` **离线折叠进 k_proj 权重(除 s)和 q_proj 权重(乘 s)**,推理期零开销、无需运行时变换——这是 SmoothQuant「不改 kernel 数据通路」性质的自然落地。
5. **非对称(可选)**:zero-point cache + 我们已实现的 `S_q` 修正项(kernel 已支持),开销 +1~2%。

难点是 vLLM 的 paged 布局 / block manager / backend 接口是版本绑定的(本环境 vllm==0.19.1),属于较大且脆的集成工程,作为 future work;但路径清晰,且每一块都对应本项目已验证的组件(v5/v6 kernel、per-token scale、SmoothQuant 折叠、`S_q` 修正)。

## 结论:整个项目闭环

- **动机被实证**:vLLM 唯一的 sub-fp16 KV(FP8)在小模型上崩盘(0.63→0.04),**确实需要 INT8 KV**。
- **方案被验证**:INT8 per-token + SmoothQuant 端到端近乎无损(PPL +2.9%,GSM8K −1pt),kernel 已实现并通过正确性测试,性能对照 vLLM 已有基准。
- **集成路径清晰**:原生 vLLM INT8 后端的设计已列出,每一步都落在已验证组件上。

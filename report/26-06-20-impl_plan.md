# 实施计划：方案 B + D + Section 3（INT8 KV 量化进阶）

> 配套文档：`report/activation_quant_improvements_cn.md`。本文是把那份分析落到代码的工程计划。

## 关于 vLLM 的决定：解耦，最后再集成

本轮所有工作（方案 B、D、Section 3）都是**量化精度**工作，vLLM 在这里只会碍事：

- **Fake-quant 注入**（3.1 / 3.3）要在 RoPE 之后拦截 K/V，替换成 quant→dequant 版本。在 HF transformers 的 eager attention 里这是 5 行 monkeypatch；在 vLLM 的融合 CUDA 算子里要跟框架硬刚。
- **PPL / GSM8K** 是普通的 HF 评测循环，vLLM 帮不上。
- **B 和 D** 是对 Q/K 做的变换，验证对象是我们自己的 kernel，不是 vLLM。

vLLM 只保留两个窄角色，且都在最后：
1. `vllm._C.paged_attention_v1/v2` 作为 **kernel 算子**的速度基线（轻量 import，已在用），保留。
2. 可选的收尾项：把 INT8 kernel 包成 vLLM attention backend，演示「填补空白」的动机。这是大工程，放最后，时间够才做。

**结论**：B/D/Section-3 全部在 HuggingFace + 自有 CUDA kernel 上完成，最后才碰 vLLM 引擎。

## 关键结构性洞见（决定了阶段顺序）

INT8 kernel 把 Q 读进 shared **fp32**（`q_sh`），K 以 int8 + per-token scale 存储。因此：

> **方案 B（SmoothQuant）和方案 D（Hadamard）不需要改任何 CUDA 代码**。两者都把变换折叠进 Q（fp32，host 侧），并存一个预变换后的量化 K。`Q'·K'ᵀ = Q·Kᵀ` 精确成立。验证方式：把变换后的 `Q'` 和重新量化的 `K'` 喂给现有的 v5/v6 kernel。

只有**非对称量化（3.3）**需要改 kernel（`S_q` 修正项）。所以先把便宜的精度收益做完，唯一的 CUDA 改动放后面。

## 分阶段计划

### Phase 0 — 评估地基（Section 3.2 指标 + 3.1a 离群数据）｜无需 GPU 的部分先行
- `tests/quant_metrics.py`：`cosine_sim(a,b)`、`rel_l2(a,b)=‖a−b‖/‖b‖`。
- 重写 `tests/test_paged_decode_attn.py` 里的 `rel_err` 块：用 cosine + rel-L2 取代「除以近零」的逐元素 max；收紧阈值。
- `tests/test_quant_outliers.py`：合成压力测试 —— Student-t KV + 对 K 固定通道注入离群值；断言误差**随离群幅度增长**（记录弱点，不做通过/失败表演）。
- *验收*：旧测试重写后通过；离群测试如预期显示对称 INT8 退化。

### Phase 1 — 离线 fake-quant 评估台（Section 3.1 基础设施）
- `benchmarks/dump_kv.py`：HF 跑 Qwen2.5-1.5B，dump **post-RoPE** K/V 到磁盘。
- `benchmarks/fakequant_eval.py`：`fake_quant_kv(mode=…)` 的 HF attention monkeypatch + 在 dump 的真实 K/V 上跑 Phase-0 指标，覆盖 `{fp16, int8_per_token, int8_per_tensor}`。
- *验收*：真实数据下当前对称方案的 cosine/rel-L2 —— 取代合成 3% 的诚实基线。

### Phase 2 — 方案 B：SmoothQuant（不改 kernel）
- `tests/quant_transforms.py::smoothquant_calibrate(K_stats, Q_stats, α)` → per-channel `s_d`；`apply(Q,K,s)` → `Q*s, K/s`。
- 作为新 mode 接入评估台；扫 α。
- *验收*：真实 K/V（及离群测试）上 SmoothQuant 优于纯对称；校验 `Q'·K'ᵀ` 与未平滑版在浮点容差内一致。

### Phase 3 — 方案 D：Hadamard 旋转（不改 kernel）
- `quant_transforms.py` 加 `hadamard_apply(Q,K)`（归一化 Hadamard，head_dim=128 是 2 的幂）。
- 新 mode；*验收*：`K@H` 的离群直方图比 `K` 更平；cosine/rel-L2 改善（离群数据上尤甚）；校验 `Q@H·(K@H)ᵀ == Q·Kᵀ`。

### Phase 4 — Section 3.3：非对称量化（唯一的 kernel 改动）
- 参考实现：`paged_decode_attn.py` 加 `quantize_per_token_asym`（min/max → scale + zero-point）。
- Kernel：`paged_attention_v5/v6` 加可选 `k_zeros`/`v_zeros` 缓冲；warp 归约算一次 `S_q = Σ_d q[d]`；分数变 `s_k·(dot − z_k·S_q)`。对称为默认（传 zeros）。建议就地扩展 v5/v6，靠 buffer 是否为空切换，而非新建 v7。
- *验收*：`tests/test_int8_asym.py` —— kernel 与非对称参考一致；偏分布 K 上非对称优于对称。

### Phase 5 — 端到端精度表（Section 3.1c）
- `fakequant_eval.py` 扩展出 **WikiText-2 PPL** + **GSM8K(200)**，纳入 B/D/asym。注意 Qwen `repetition_penalty=1.1` 贪心陷阱。
- *验收*：`report/` 一张表 —— `FP16 / FP8(sim) / INT8-per-token / per-tensor / +SmoothQuant / +Hadamard / +asym` × `PPL, GSM8K, rel-L2`。

### Phase 6 — vLLM（最后，可选）
- 保留 `vllm._C.paged_attention` 速度基线。时间够再原型 INT8 vLLM backend。

## 两个待定决策
- **非对称范围**：K-only（更便宜，K 才是离群源）vs K+V。建议先 K-only。
- **FP8 基线**：软件模拟 E4M3 dequant 用于对照表（A100 无完整 FP8 tensor core）。建议仅作 sim。

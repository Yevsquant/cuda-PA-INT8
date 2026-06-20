# Phase 4 结果：非对称 K 量化（zero-point + 点积修正项）

> 配套：kernel 改动 `paged_attention_v5/v6`（加可选 `k_zeros`），参考实现 `quantize_per_token_asym` / `build_k_cache_int8_asym` / `dequantize_kv(k_zeros=...)`，测试 `tests/test_int8_asym.py`。这是整个计划里**唯一改 CUDA**的一项，K-only。

## 直接回答审稿意见三的问题

审稿意见原话:「评估带 zero-point 的非对称量化……衡量精度增益是否值回多出的内层开销」。两边都量了:

- **开销极小**:correction 让 kernel 慢 **warp +0.7% / split-K +2.2%**(A2048×64seq×32head 实测)。
- **真实 K 上增益不大**:post-RoPE K 近似零中心(Phase 4 实测 skew proxy 0.04–0.35),非对称**单独几乎无效**(layer 0:0.1321→0.1313)。叠在 SmoothQuant 之上能再拿 **~11%**(layer 0:0.0313→0.0279),因为 SmoothQuant 压平通道后残差才轻微偏斜。

**结论**:非对称便宜(~1–2%),但对 post-RoPE K 是「锦上添花」而非主力——K 的病是**离群通道**(SmoothQuant 治),不是**偏斜**(非对称治)。作为可开关的 ablation 保留,默认关。

## 数学:为什么修正项几乎免费(已在 kernel 实现并验证)

Q 不量化(fp32),只有 K 走 INT8。非对称 dequant:`k_real = s_k·(k_int − z_k)`。于是

```
score = Σ_d q[d]·k_real[d]
      = s_k·( Σ_d q[d]·k_int[d]  −  z_k · Σ_d q[d] )
      = s_k·( dot_int           −  z_k · S_q       )
```

`S_q = Σ_d q[d]` **只跟 query 有关,每个 (seq,head) 算一次**(一次 warp 归约),所有 key 共用。每个 key 只多一次 `− z_k·S_q` 标量运算 + 一次 `k_zeros` 全局读。这解释了为什么实测只慢 ~1–2%。

## 工程实现(surgical,不破坏对称路径)

- kernel 加可选 `const float* k_zeros`:**为 null 时走原对称路径,字节级不变**;非 null 时算 `S_q` 并修正。`block_reduce_sum_warp` 广播 `S_q` 给全 block。
- 绑定层用 `c10::optional<torch::Tensor> k_zeros` 作尾参;Python runner 默认 `k_zeros=None`,所有既有调用(benchmark/对称测试)零改动。
- V 仍对称(K-only 决定)。V 的非对称类似但修正项是常数 `−z_v·s_v`(因 Σ_j p_j = 1),留作后续。

## 测试覆盖(全绿)

- CPU:affine 量化往返误差 <2%;偏斜分布(Exponential)上非对称 rel-L2 < 0.7× 对称。
- GPU 正确性:warp + split-K(含 600-token 多 partition)的 kernel 输出匹配非对称 dequant 参考(atol/rtol 2e-2)。
- **z=0 无操作守卫**:全零 zero-point 必须与对称 kernel **逐位相同**(correction = z_k·S_q = 0),验证 gated 分支是真 no-op。
- 回归:既有 262 个 INT8 kernel 测试全过,对称路径无损。

## 与前序阶段的合流

| 方法 | 治什么 | 真实 layer 0 rel-L2 | 改 kernel? | 成本 |
|---|---|---:|:--:|---|
| baseline 对称 per-token | — | 0.132 | — | — |
| **SmoothQuant** | 离群**通道** | **0.031** | 否 | ~0(融进 RoPE 写回) |
| Hadamard | 非系统离群 | 0.074 | 否 | 一次旋转 |
| + 非对称 K | 残差**偏斜** | 0.028(SQ+asym) | 是 | +0.7~2.2% |

主线已清晰:**SmoothQuant 打主力,非对称作为低成本 ablation 叠加再榨 ~11%**。

## 下一步(Phase 5)

把这些喂进端到端:WikiText-2 PPL + GSM8K,对照 `FP16 / INT8-sym / +SmoothQuant / +asym`,把 rel-L2 翻译成困惑度/正确率。

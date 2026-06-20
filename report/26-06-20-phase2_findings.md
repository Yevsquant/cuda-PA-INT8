# Phase 2 结果：SmoothQuant（方案 B）在真实 K/V 上的效果

> 配套：`tests/quant_transforms.py`（变换 + 单测）、`benchmarks/smoothquant_eval.py`（per-layer 标定 + α 扫描）。数据沿用 Phase 1 的 `report/kv_dump.pt`。

## 结论一句话

SmoothQuant **不改 kernel、对点积零损失**（`Q'·K'ᵀ = Q·Kᵀ`，实测 fp 残差 1.4e-5），却把 Phase 1 暴露的浅层离群灾难基本治好：**layer 0 的 decode rel-L2 从 0.132 砍到 0.031（-76%）**。收益与 K 离群比强相关——离群越重，治得越狠。

## 数据（decode rel-L2，越低越好）

| layer | K 离群比 | baseline (对称 INT8) | α=0.3 | α=0.5 | α=0.7 | α=0.85 | α=1.0 | 最佳 |
|------:|------:|------:|------:|------:|------:|------:|------:|:------|
| 0  | 71× | 0.1321 | 0.1132 | 0.0613 | 0.0452 | **0.0313** | 0.0317 | α=0.85 **-76%** |
| 14 | 4×  | 0.0203 | 0.0217 | 0.0198 | 0.0195 | 0.0203 | 0.0216 | α=0.7 -4% |
| 27 | 6×  | 0.0094 | 0.0082 | 0.0080 | **0.0068** | 0.0070 | 0.0073 | α=0.7 -28% |

> 标定：per-(kv_head, channel) 的 abs-max。K 取 kv-head 粒度；Q 在每个 GQA 组(6 个 q head)内取 max。α 是「把多少难度搬给 Q」的旋钮。

## 怎么读

- **layer 0 是甜点**：离群比 71×，baseline 0.132（Phase 1 的痛点）。SmoothQuant 把离群通道按 `s_d` 压平，per-token scale 不再被独大通道撑爆 → 0.031。难度搬到了 Q，而 Q 不量化(fp32)，所以**零代价**。
- **α 有最优值**：离群重的 layer 0 偏好大 α(0.85，几乎把 K 通道范围全摊到 Q)；离群轻的层 α≈0.7 即可，α 太大反而略伤(layer 14 在 α=1.0 比 baseline 还差)。**固定 α≈0.8 或 per-layer 标定**都合理。
- **GQA 正确性**：平滑因子必须 per-(kv_head, channel)，且一个 Q head 与它的 KV head 共用同一因子，否则 `Q'·K'ᵀ = Q·Kᵀ` 不成立。单测 + 这里的 exactness 检查(1.4e-5)都验证了。

## 与 Phase 1 的闭环

Phase 1 说「浅层离群是病灶，深层基本没事」。Phase 2 正好印证:**收益排序 = 离群比排序**(layer 0 ≫ 27 > 14)。把浅层 0.132 打到 0.031，已逼近深层/合成基线的 ~0.01 水平。

## 工程性质

- **不动 INT8 kernel**：`Q' = Q·s`(host 侧 fp32)、`K' = K/s` 后照常 per-token 量化，喂现有 v5/v6 kernel 即可。
- **运行时开销**：标定离线做一次得到 `s`；推理时 Q 乘 s、K 除 s 各一次 per-channel 乘法，可融进 RoPE 后的写回，几乎免费。

## 下一步指向

- 与 Phase 3(Hadamard 旋转)对比:两者都靠「不改 kernel 的 Q/K 变换」治离群。Hadamard 把离群**摊散**成高斯,SmoothQuant 把离群**搬给 Q**。在 layer 0 上比谁更狠、能否叠加。
- α 默认值:报告里用 α=0.85(outlier 层最优)或做 per-layer。

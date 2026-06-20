# Phase 3 结果：Hadamard 旋转（方案 D）vs SmoothQuant，及二者组合

> 配套：`tests/quant_transforms.py`（新增 `hadamard_matrix`/`hadamard_apply` + 单测）、`benchmarks/transform_eval.py`（baseline / SmoothQuant / Hadamard / 组合 四臂对比）。数据沿用 `report/kv_dump.pt`。所有臂的 V 都用同一套 per-token 量化，参考都是 raw-fp，所以差异**只来自 K 路径**。

## 结论一句话

对 post-RoPE K 这种**系统性的「按通道」离群**，**SmoothQuant 明显胜过 Hadamard**（layer 0：-76% vs -44%），而且**组合不叠加**（SQ+Had ≈ SQ，甚至略差）。原因是：离群锁定在固定通道上，知道这个基的方法(SmoothQuant 标定)就该赢过「基无关地摊平」的 Hadamard。

## 数据（decode rel-L2，越低越好；括号是相对 baseline 的变化）

| layer | K 离群比 | baseline | SmoothQuant(α=.85) | Hadamard | Smooth+Had |
|------:|------:|------:|------:|------:|------:|
| 0  | 71× | 0.1321 | **0.0313 (-76%)** | 0.0736 (-44%) | 0.0347 (-74%) |
| 14 | 4×  | 0.0203 | 0.0203 (-0%) | **0.0190 (-6%)** | 0.0196 (-3%) |
| 27 | 6×  | 0.0094 | 0.0070 (-26%) | 0.0070 (-25%) | **0.0069 (-26%)** |

Exactness：四臂的「变换后 fp 输出 vs 参考」最大 rel-L2 = 1.5e-5（纯 fp 噪声）→ `Q'·K'ᵀ = Q·Kᵀ` 对 Hadamard 和组合都成立，依旧零损、不改 kernel。

## 为什么 SmoothQuant 在 layer 0 赢 Hadamard

- **离群是「按通道」系统性的**：layer 0 有一个通道幅度 71× 中位通道。
- **Hadamard 把单通道尖峰摊到 128 个通道**，每个分到 ~峰值/√128 ≈ 峰值/11.3。71× 摊完仍剩 ~6× 抬升，per-token scale 只降了约 √d，没根除 → -44%。
- **SmoothQuant 直接把那个通道按 `s_d` 除掉**，难度搬给不量化的 fp Q → per-token scale 几乎不再被它撑爆 → -76%。

一句话：**Hadamard 适合「随机/每-token 漂移、且无法标定」的离群；SmoothQuant 适合「固定通道、可标定」的离群。** post-RoPE K 属于后者。

## 为什么组合不叠加

SmoothQuant 先在「离群对齐的原始通道基」里把系统性尖峰压平，残差已接近高斯；此时再 Hadamard 没有尖峰可摊，还可能让量化轻微变差（layer 0：0.035 vs 0.031）。所以对系统性离群，**二者不互补**。

## 浅层 vs 深层

- layer 0（重离群）：方法差异最大，SmoothQuant 独优。
- layer 14/27（轻离群）：三种变换都只小幅改善且互相接近，Hadamard 在 layer 14 略好——离群越「不系统」，Hadamard 的相对优势越显现。

## 对项目的结论

- **主线选 SmoothQuant**：post-RoPE K 的离群是系统性通道型，SmoothQuant 性价比最高（-76%、零 kernel 改动、α≈0.85）。
- **Hadamard 留作对照/亮点**：实现了、验证了、解释清楚了它何时更优(非系统离群)与何时不如(本场景)。这本身就是一个有说服力的分析点，不必硬上。
- **不做组合**：实测无增益。

## 下一步

Phase 4：非对称量化（zero-point + 点积修正项），唯一需要改 kernel 的一项。可叠在 SmoothQuant 之上看是否再降。

# Llama-3.2-1B A4 beta held-out 选择结果

## 结论

保留上游默认全局 A4 beta，不把本轮校准出的 `global`、`projection_type` 或 `module` 配置用于正式 test。逐模块配置是三个新方案中最优的，但 held-out PPL 仍从 **9.782117** 恶化到 **10.270711**（+0.488594，+4.99%）。

这是一个有价值的负结果：只降低 overload 并不保证语言模型质量更好。当前 First-beta DP 在校准样本上优化块级平方误差和覆盖约束，但该代理目标没有在独立 train 窗口上转化成更低的端到端 NLL。

## 公平协议

- 模型：`meta-llama/Llama-3.2-1B`，固定 revision；权重为同一 Hadamard + shared128 LDLQ W4 NQMC。
- beta 校准只使用 WikiText-2 train manifest 窗口 0--7；profile 选择只使用互不重叠的窗口 8--15。
- held-out 共 8 × 2048 = 16,384 token，token SHA-256 为 `6f1db4ae265dbfef92f9fc496fb8e170b9d1eeb6b7a91a68e881365b805126f2`。
- 四档配置使用同一 checkpoint、同一 token 顺序、同一 A4 fake-quant 路径；没有读取 WikiText-2 test 来选择 profile。
- 原始结果 SHA-256：`aaba10752edb39f58bf6e1143bea1b535e0d6372b9e56bae0382d000cbb340bf`。

## 结果

| beta profile | held-out PPL | 相对默认 PPL | overload | 相对平方误差 |
|---|---:|---:|---:|---:|
| upstream default | **9.782117** | 0.000000 | 5.435952% | **0.638403%** |
| global | 10.428573 | +0.646456 | **0.219551%** | 1.297745% |
| projection type | 10.557727 | +0.775610 | 0.353053% | 1.259561% |
| module | 10.270711 | +0.488594 | 0.917280% | 1.091281% |

新配置把 overload 从 5.44% 大幅降低到 0.22%--0.92%，但重建相对平方误差由 0.64% 升到 1.09%--1.30%，PPL 也同步恶化。原因不是“逐层配置复杂度不够”，因为最细粒度的 module 方案仍未超过默认值；更可能是当前候选选择目标、稳定 overload 包络和真实 codec 重建误差之间存在目标失配。

## 决策与后续

1. 当前正式 W4A4 结果继续使用上游默认 beta，不重新跑 test 来追逐数字。
2. 若继续研究 beta，应先在 train 内改进代理目标，例如按模块输出敏感度加权，或直接最小化小规模 held-out NLL；需要再划分新的 selection/validation 窗口，不能反复拟合本报告的窗口 8--15。
3. 所有 A4 数字仍是浮点重建的 fake-quant 质量模拟，不证明低比特激活存储、显存下降或 kernel 加速。

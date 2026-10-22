# Llama-3.2-1B：NestQuant W4 旋转 × block-LDLQ 消融

## 结论

在相同源码、模型、测试 token、量化参数、dither seed 和 WikiText-2 test 协议下，signed-Hadamard 权重旋转与 train64 block-LDLQ 可以组合。两个旋转种子的组合 PPL 分别为 **10.136666** 和 **10.121536**，都优于仅 LDLQ 的 10.276133，说明旋转在 LDLQ 之后仍提供了可重复的额外改善。

| 设置 | rotation seed | train64 LDLQ | PPL | 相对普通 W4 改善 |
|---|---:|---:|---:|---:|
| BF16 | — | 否 | **9.757045** | — |
| 普通 NestQuant W4 | — | 否 | 11.202577 | 0 |
| 仅 signed-Hadamard 旋转 | 123 | 否 | 11.004825 | 0.197752 |
| 仅 signed-Hadamard 旋转 | 124 | 否 | 11.147938 | 0.054639 |
| 仅 block-LDLQ | — | 是 | 10.276133 | 0.926444 |
| signed-Hadamard + block-LDLQ | 123 | 是 | **10.136666** | **1.065911** |
| signed-Hadamard + block-LDLQ | 124 | 是 | **10.121536** | **1.081041** |

两种子组合 PPL 均值为 **10.129101**，样本标准差为 **0.010698**。旋转在 LDLQ 之后额外降低 PPL `0.139467`（seed 123）和 `0.154597`（seed 124）。

## 模块贡献与交互

- 不使用旋转时，LDLQ 将 PPL 降低 `0.926444`。
- seed 123 中，旋转单独降低 `0.197752`，在 LDLQ 后额外降低 `0.139467`。
- seed 124 中，旋转单独降低 `0.054639`，在 LDLQ 后额外降低 `0.154597`。
- 两个种子都表现出部分收益重叠，而组合均优于任一单模块设置。

## 坐标系核验

对行向量旋转 `X -> X R` 和权重旋转 `W -> W R`，LDLQ 使用的协方差必须变为 `H_rot = R^T H R`。项目直接对 train64 Hessian 做双侧快速 Hadamard 变换，与重新对 `X R` 收集协方差数学等价；单元测试已用显式激活计算核对。

真实 `layers.0.self_attn.q_proj` smoke 中，仅旋转的 Hessian 代理损失为 `1.814437`，旋转加 LDLQ 为 `0.392790`，下降 `78.35%`。普通权重 MSE 增加到 `1.83` 倍，符合 LDLQ 优化激活加权误差而非逐权重 MSE 的目标。

## 可复核协议

- NestQuant 源码 SHA-256：`d499e68331017406b67bed05d2f2f6f1ce7b9f351455bb301db77e2620dc89c1`。
- 测试 token SHA-256：`54927bf19e6e64b4008eeb93370c370fb25db18cf68eac8015cfe094c464d567`；预测 token：`288,627`。
- train64 校准 token SHA-256：`7dcba79b76b0e185ab48d7e27697db01da6345122b6fd0f000a3d64ae9229db3`。
- 所有运行使用相同模型/数据 revision、q=14、dither seed 58、7 个 beta 候选中选 4 个、LDLQ 相对阻尼 0.01。

## 边界

本实验组合的是项目当前“每个 Linear 独立随机符号 Hadamard、量化后逆旋转回原权重坐标”的算法消融，与论文完整的全模型旋转融合仍有差距。组合结论只用两个预先测试的 rotation seed 做了稳健性核验，不构成大样本统计。当前也没有 A4、KV4、QA-LDLQ 或低比特 kernel；PPL 使用 BF16 Linear 执行，不能据此声称推理加速。

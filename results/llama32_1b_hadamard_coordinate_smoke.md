# Llama-3.2-1B 局部 Hadamard 坐标等价性 smoke

## 目的

验证论文 §4.3 的局部 Linear 坐标表达：离线保存 `W H`，前向在线计算 `X H`，在未量化时应满足 `(XH)(WH)^T=XW^T`。本实验只检查数值路径，不计算 PPL，也不证明量化质量或速度。

## 协议

- 模型：`meta-llama/Llama-3.2-1B`，revision `4e20de362430cd3b72f300e6b0f18e50e7166e08`；
- 原模型 dtype：BF16；旋转参考权重和 Hadamard/Linear 计算：FP32；输出转回 BF16；
- 输入：seed 58 生成的固定 16 token；
- `lm_head` 与输入 embedding tied，因此不转换；
- 转换不含 E8、beta、LDLQ、A4 或 KV4。

## 结果

| 转换范围 | 最大绝对 logits 误差 | 平均绝对误差 | 相对 L2 误差 | top-1 token 一致率 |
|---|---:|---:|---:|---:|
| 1 个 Linear | 0.28125 | 0.021346 | 0.9554% | 100% |
| 第一个 block 的 7 个 Linear | 0.28125 | 0.032452 | 1.0751% | 100% |
| 16 层共 112 个 Linear | 0.31250 | 0.032839 | 1.0964% | 100% |

FP64 单元测试在 `1e-12` 容差内通过前向、输入梯度和旋转权重梯度等价。真实 BF16 模型不是逐位等价：原 BF16 GEMM 与 FP32 Hadamard/GEMM 后再转 BF16使用不同舍入路径。误差从 7 个到 112 个 Linear 没有按层数成比例增长，且本次 16 个位置的 top-1 均一致，支持继续把该模块用于量化前的坐标语义核对。

## 边界

- 只有 16 个随机 token，不能替代 WikiText-2 PPL 或下游任务；
- top-1 一致不等于概率分布完全一致；
- FP32 参考权重增加内存和计算量，不是部署方案；
- 没有证明作者 `kron_h_ip` 的非 2 次幂路径；当前 Llama-3.2-1B 的目标 Linear 输入维度均为 2 的幂；
- 下一步应比较固定上游和本项目在同一小张量上的旋转、量化、逆旋转中间值，再决定是否接入正式权重量化路径。

机器可读摘要见 [同名 JSON](llama32_1b_hadamard_coordinate_smoke.json)，原始 manifest 保存在被 Git 忽略的 `results/raw/`。

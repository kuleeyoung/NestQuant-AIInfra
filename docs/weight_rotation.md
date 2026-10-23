# 权重旋转消融：实现范围与验证

## 目的

检验旋转坐标系是否改善现有 E8 W4 权重量化，并验证它与 block-LDLQ 是否可以叠加。当前仍是每个 Linear 独立的权重空间变换，尚未实现论文的完整旋转图或 QA-LDLQ。

论文 [§4.3](https://arxiv.org/html/2502.09720v3#S4.SS3) 对线性层的权重与激活使用匹配的 Hadamard 变换，并讨论非 2 次幂维度。本实现先支持 Llama-3.2-1B 线性层所需的 2 次幂输入维度，使用归一化 Sylvester Hadamard 与固定种子的随机符号矩阵。随机符号及其逐层生成顺序是本项目的消融定义，不宣称逐项对应作者实现。

## 计算

PyTorch 的线性层记为 `Y = X W^T`。对每层设 `R = D H`，其中 `D` 是 ±1 对角阵、`H H^T = I`。则：

```text
权重 W → W R → E8 / β DP / Opt-beta → Q(W R)
实际写回权重 = W + [Q(W R) - W R] R^T
输入 X → 原有浮点 Linear → 输出
```

实数精确算术下，写回权重等于 `Q(W R) R^T`，对应未量化输入的 `X R Q(W R)^T`。本版本把逆变换放在离线重建阶段，模型前向没有新增旋转算子。它不能评估在线激活旋转的开销，也不能用于 A4/KV4 的完整结论。

采用误差残差公式是数值稳定性选择：恒等量化器 `Q(Z)=Z` 时，残差严格为零，避免 FP32 正逆旋转的小误差在转回 BF16 后改变权重。

## 验证与实现

- `core/rotation.py`：蝶形算法，避免构造完整旋转矩阵；正变换为 `x D H`，逆变换为 `x H D`。
- `models/quantization.py`：独立 CPU rotation RNG，不消耗 dither RNG；按行分块旋转与逆变换；共享 embedding/lm_head 的参数关系保持不变。
- `tests/unit/core/test_rotation.py`：与显式 Hadamard 矩阵对照、逆变换、范数、线性输出、恒等量化器下 BF16 权重保持、固定 seed 和 tied weights。
- `scripts/check_rotation_equivalence.py`：真实 1B 模型 113 层的恒等量化器检查，用一个固定 32-token 输入比较 logits。此检查范围有限，不代替全量 PPL。
- `models/rotated_linear.py`：论文 §4.3 的局部参考路径，离线保存 `W H`，前向在线计算 `X H`；未量化时与原 `Linear` 严格等价。当前只支持 2 的幂输入维度，覆盖 Llama-3.2-1B 的 2048/8192 Linear 输入，不宣称已覆盖论文 Llama-3-8B 的 14336 维 Kronecker 路径。
- `convert_linears_to_hadamard_coordinates_`：先验证全部目标名称、维度和 tied embedding，再一次性替换模型投影；默认跳过与输入 embedding 共享存储的 `lm_head`。`scripts/check_llama_hadamard_coordinates.py` 用固定短 token 流比较转换前后 logits，只作为等价性 smoke。

旋转消融的评测入口为 `--methods nestquant_w4 --rotation-seed 123`；省略该参数是无旋转对照。与 `--ldlq-hessian-dir` 同时使用时，量化器按每个 Linear 的旋转计算 `H_rot = R^T H R`，再做 block-LDL 分解。该变换与重新对旋转激活 `X R` 收集协方差数学等价，并有显式激活单元测试。当前 NQRT 未保存旋转定义，因此旋转模式会拒绝导出/回载参数，防止把旋转坐标系码字当作原坐标系权重读取。既有无旋转 NQMC 文件不受影响。

完整 WikiText-2 test 的同源码消融已经覆盖无旋转与两个 rotation seed，结果见[旋转消融报告](../results/llama32_1b_w4_rotation_ablation.md)。两个 seed 都降低 PPL，但改善幅度不同；当前证据支持继续研究，不支持宣称对任意 seed 都稳定改善。

同源码消融已经完成：BF16 与普通 W4 的 PPL 分别为 9.757045 和 11.202577；seed 123/124 的仅旋转 PPL 为 11.004825/11.147938，加入 train64 block-LDLQ 后为 10.136666/10.121536，二者都优于仅 LDLQ 的 10.276133。两个组合结果均值 10.129101、样本标准差 0.010698，完整结果见[旋转 × LDLQ 两种子消融](../results/llama32_1b_w4_rotation_ldlq_ablation.md)。后续需要实现完整旋转结构及旋转元数据序列化；QA-LDLQ 对量化激活的修正随 A4 实验单独验证。

## 论文与作者代码的评测 split 差异

2026-10-20 核对：论文附录 G 写 train 校准、validation 评测；作者公开的
[`parallel/ppl.py`](https://github.com/cookiedoth/nestquant/blob/main/parallel/ppl.py)
调用 `is_testset=True`，
[`parallel/ppl_utils.py`](https://github.com/cookiedoth/nestquant/blob/main/parallel/ppl_utils.py)
将其映射为 `test`。本项目保留 test，与该公开代码的 split 一致。它不自动证明论文表格实际使用何种 split，完整复现仍应固定作者代码 commit 并逐项核对协议。

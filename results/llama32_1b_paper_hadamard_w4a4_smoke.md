# 论文式 Hadamard + shared128 LDLQ W4A4 首窗口 smoke

## 目的

验证 A4 是否能直接作用于论文式局部 `X H` 坐标，并与保存 `W H` 的 `HadamardRotatedLinear`、shared128 LDLQ NQMC 和完整 Llama-3.2-1B 前向兼容。该实验只使用 WikiText-2 test 第一个窗口的 2,047 个预测 token。

## 语义

固定作者代码的普通坐标模拟为：

```text
X -> H -> A4 -> H^-1 -> Linear(W)
```

本项目局部坐标路径为：

```text
X -> H -> A4 -> Linear(W H)
```

权重和激活使用相同正交 H 时，两条路径的点积数学等价。新增单元测试实测最大差异处于 FP32 舍入量级；局部路径省去量化后逆 Hadamard和 Linear 前再次 Hadamard的两个冗余变换。

## 结果

| 路径 | 首窗口 PPL |
|---|---:|
| Hadamard + shared128 LDLQ W4 | 7.462738 |
| Hadamard + shared128 LDLQ W4A4 | 7.860542 |
| A4 增量 | +0.397804（+5.33%） |

A4 统计：

- 坐标：`local_hadamard_input`；
- 模块：112；
- 激活 entry：671,088,640；
- 向量：229,376；
- 相对平方误差：0.00645637；
- overload：4,578,164 / 83,886,080 E8 blocks，比例 5.4576%；
- 分析码率：4.26094 bit/entry；
- 运行时：浮点重构，不是低比特激活 kernel。

5.46% overload 说明固定上游全局 A beta 在当前 Llama-3.2-1B 局部坐标分布上可能仍有优化空间。下一步可以先建立全量全局-beta基线，也可以先校准逐层 beta；不能根据首窗口数字预填完整 PPL。

## 验收状态

- 112 个投影全部成功挂载局部坐标 A4；
- 与同一个 NQMC 控制共享模型、token、checkpoint SHA-256；
- 没有 NaN、Inf、shape、dtype 或显存错误；
- 123 项项目测试通过；
- 正式完整 PPL 尚未运行。

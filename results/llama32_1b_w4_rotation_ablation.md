# Llama-3.2-1B NestQuant W4 权重旋转消融

## 结论

在完全相同的模型、WikiText-2 test token、dither seed、β 候选与源码下，两个 signed-Hadamard 权重旋转 seed 都优于无旋转版本。无旋转 PPL 为 **11.202577**；旋转后的平均 PPL 为 **11.076382**，平均下降 **0.126195**。两个 seed 的改善幅度不同，说明旋转选择会影响量化质量，目前还不足以宣称对任意 seed 都稳定有效。

## 固定协议

- 模型：`meta-llama/Llama-3.2-1B`，revision `4e20de362430cd3b72f300e6b0f18e50e7166e08`
- 数据：WikiText-2 raw test，revision `b08601e04326c79dfdd32d625aee71d232d685c3`
- 上下文长度：2048；完整非重叠块；预测 token 数：288,627
- token SHA-256：`54927bf19e6e64b4008eeb93370c370fb25db18cf68eac8015cfe094c464d567`
- NestQuant 源码 SHA-256：`6ea6106cddc588396b2b1a6ccdcc0405442b94b28b0dea9c672d934912e03e78`
- 量化：Linear 权重 W4 质量仿真，E8、q=14、7 个候选中 DP 选 4 个、CPU dither seed 58
- 执行：量化值反量化回 BF16 权重后计算；没有低比特 kernel

## 结果

| 设置 | rotation seed | PPL | 平均 NLL | 相对无旋转 PPL 变化 | 总耗时 |
|---|---:|---:|---:|---:|---:|
| 无旋转 | — | 11.202577 | 2.416144 | 0 | 213.3 s |
| signed-Hadamard | 123 | **11.004825** | **2.398334** | **-0.197752** | 187.0 s |
| signed-Hadamard | 124 | 11.147938 | 2.411255 | -0.054639 | 238.0 s |

两个旋转 seed 的 PPL 总体标准差为 0.071557。耗时包含模型加载、离线量化和 PPL，不作为推理速度结论；不同运行的缓存与机器状态会影响它。

## 结果边界

这项实验只回答“将 Linear 权重旋转到 signed-Hadamard 坐标系后，当前 E8 W4 量化误差是否影响 PPL”。为保持原模型接口，量化误差被逆旋转回原权重坐标，并以 BF16 Linear 执行。因此它不等同于论文完整的权重/激活匹配旋转，也没有覆盖 A4、KV4、LDLQ 或 QA-LDLQ。

当前 NQMC 格式没有保存旋转定义，本实验没有生成可回载的旋转低比特模型。它证明的是质量消融，不证明模型文件进一步缩小、显存下降或推理加速。

论文附录 G 写 train 校准、validation 评测；作者公开 `parallel/ppl.py` 与 `parallel/ppl_utils.py` 的路径使用 test。本项目这组实验明确使用 test。JSON 汇总见 [`llama32_1b_w4_rotation_ablation.json`](llama32_1b_w4_rotation_ablation.json)，原始 manifest 位于被 Git 忽略的 `results/raw/`。

## 下一步

下一阶段先在 WikiText-2 train 上收集线性层输入二阶统计并实现普通 LDLQ。W4 权重消融验证普通 LDLQ；面向量化激活的 QA-LDLQ 修正放到 A4 阶段单独验证，避免把多个因素同时加入实验。

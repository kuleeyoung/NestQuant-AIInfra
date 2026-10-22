# Llama-3.2-1B WikiText-2 首轮质量仿真

## 实测结果

| 方法 | WikiText-2 test PPL | 相对 BF16 的 PPL 变化 | 说明 |
|---|---:|---:|---|
| BF16 | 9.757045 | 基线 | 原始模型，BF16 权重/计算 |
| RTN W4 | 13.021485 | +33.46% | Linear 权重按行分组，group size 128，对称 RTN fake quant |
| NestQuant W4 | 11.149369 | +14.27% | Linear 权重行归一化、E8 分块，Appendix F beta 子集 DP + Opt-beta fake quant |

NestQuant 在本次设置下的 PPL 比 RTN W4 低约 14.38%。这是同一模型与输入协议下的初步质量对比；RTN 与 NestQuant 的有效码率、scale/beta 元数据开销尚未匹配，所以不能据此断言 NestQuant 在等码率条件下算法更优。

## 固定协议与可复核文件

- 模型：`meta-llama/Llama-3.2-1B` Base，revision `4e20de362430cd3b72f300e6b0f18e50e7166e08`。
- 数据：WikiText-2 raw test，revision `b08601e04326c79dfdd32d625aee71d232d685c3`。
- context length 2048，141 个完整窗口，共 288,627 个 next-token 目标；总 NLL 按预测 token 数聚合后计算 PPL。
- GPU：NVIDIA GeForce RTX 5070；PPL 前向阶段 BF16 / RTN / NestQuant 峰值 allocated 显存约 3.07 / 3.07 / 3.07 GB。量化阶段峰值没有单独记录；上述数字不是 INT4 存储或显存收益。
- 完整 JSON manifest：[llama32_1b_wikitext2_full_20261014.json](llama32_1b_wikitext2_full_20261014.json)，run ID `20261014T041113467349Z`，config SHA256 `4677e85a8321351dd5b7e53dadebad9d6d8cb6a9be0a9df2b19f74f8805d16c4`。
- 表格：[llama32_1b_wikitext2_full.csv](llama32_1b_wikitext2_full.csv)。原始运行文件仍保存在 `results/raw/`，但该目录被 Git 忽略。

## 必须披露的实现边界

本轮属于 **fake-quant 质量仿真**：量化后反量化回 BF16 浮点权重，再以 BF16 执行前向。未实现真实低比特 pack、INT4 kernel、推理加速或文件体积核算；也未实现 NestQuant 的旋转、LDLQ/QA-LDLQ。

NestQuant 的项目候选系数为 `[3.5, 4.5, 6.0, 14.5, 25.0, 40.0, 50.0]`，DP 选 4 个。论文附录 G 报告搜索全集到 40.0/q、默认选中列表到 25.0/q；本项目把 40 加入选中候选，并以超出论文报告搜索全集的 50 覆盖模型归一化权重块。固定 5 候选在 `model.layers.0.mlp.down_proj` 仍有 4 个块 overload；扩展候选后本轮全部 Linear 层均完成量化。该扩展和有限候选表意味着本实验不是严格复现论文完整 beta 网格。

本结果可以支持“完成了受限候选下的模型质量仿真并分析与 RTN 的差异”，不支持“完整复现 NestQuant”“真实压缩到 4 bit”或“推理加速”。下一步应先扩大/审计 beta 候选策略并做重复性检查，再评估论文完整旋转与 LDLQ/QA-LDLQ；之后才进入 pack/kernel 与性能对照。

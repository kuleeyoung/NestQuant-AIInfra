# 开发路线与阶段验收

本项目面向两类可核验结果：一是按论文协议复现量化质量结果；二是逐步建立 AI Infra 岗位关注的模型加载、显存、低比特存储和推理性能证据。每个阶段先通过验收，再扩大模型和计算开销。

| 阶段 | 做什么 | 为什么 | 验收条件 |
|---|---|---|---|
| 0. 环境与协议 | 固定论文版本、上游 commit、模型/tokenizer revision、数据 split、tokenization、seed、dtype、PPL 口径 | 不固定协议，结果无法公平比较或复核 | 配置可读；环境检查可复现；协议字段和来源都有记录 |
| 1. 数学核心 | E8 oracle、嵌套编码/解码、行归一化、固定 beta 量化和 First-beta 子集 DP | 先验证小规模数学，避免把大模型下载和 GPU 调试混进算法错误 | 单测覆盖格点成员、穷举对照、码字往返、零行、误差选择和 DP 子集对照 |
| 2. 合成矩阵实验 | 对照 FP、RTN 和 NestQuant 的矩阵乘积误差，保存配置与 JSON | 低成本验证编码、解码和乘积误差；码率需按论文公式核实后再加入 | 固定 seed 可重跑；JSON 可重建指标；toy 与论文 4096 规模分开标注 |
| 3. 量化算法 | 旋转、多 beta、DP 选择、LDLQ/QA-LDLQ，逐层建立消融 | 证明各模块的作用，而非只报一个最终分数 | 每个消融只改一个因素；误差和偏差可定位到层/模块 |
| 4. Llama-3.2-1B 质量评估 | BF16 基线、RTN W4、NestQuant W4，再扩展 W4KV4/W4A4KV4 与 QuaRot 对照 | 1B 是受当前显存约束的本地起点，先保证同协议公平比较 | 同一模型、tokenizer、数据、上下文和评测流程；记录 NLL/PPL、token 数、峰值显存和日志 |
| 5. 工程测量 | 做真实打包/解包，测模型文件大小、加载时间、峰值显存；再评估算子或 kernel | 量化质量仿真本身不能证明存储下降或推理加速 | 报告真实字节、元数据开销、端到端延迟/吞吐；同硬件和精度口径对照 |
| 6. GitHub 与简历 | 一键复现实验、CI 单测、结果追溯、限制与偏差说明、简历证据表 | 面试官能检查代码和结果边界，项目陈述才可信 | 干净环境可跑 toy 实验；每个指标对应配置、命令、日志和版本 |

当前机器已知为 RTX 5070 约 12 GiB 显存。模型权重访问、实际显存峰值和各算法的可运行性必须在本机验证；不把计划当作结果。项目第一阶段先做 CPU/GPU 都可执行的小张量参考实现。

## 当前状态

- 已创建独立项目结构和 Git 仓库。
- 已实现 E8 最近格点、支持减性 dither 的嵌套 Voronoi 编解码、给定 beta 候选集时的行归一化/分块量化参考版，以及论文 §4.3 的局部离线权重/在线输入 Hadamard 等价模块；架构按 core、models、evaluation、experiments 分层，104 项主项目单元测试通过。
- Appendix F DP 已在合成表上通过穷举对照；合成矩阵乘积 toy/多 seed 实验已输出 RTN-W4、8D 块 L∞ W4 和 NestQuant 质量仿真指标 JSON/CSV。Llama-3.2-1B Base 的 BF16、RTN W4、NestQuant W4 已在完整 WikiText-2 test 上完成首轮质量仿真，保存 JSON manifest、CSV 和结果说明；NestQuant 候选集包含超出论文上限的项目扩展值，详见结果边界。
- NQPK v1、NQRT v2 和 NQMC v1 已实现；Llama-3.2-1B 的 NQMC 导出、回载、文件字节数及完整 WikiText-2 PPL 已完成，见 [模型级结果](../results/llama32_1b_nqmc_w4_full.md)。低比特 kernel、端到端延迟和吞吐仍未测量。
- 已加入 [signed-Hadamard 权重旋转消融](weight_rotation.md)，通过 FP64 数学验证与真实模型恒等量化器检查；完整 WikiText-2 test 的两个旋转 seed 均优于无旋转版本。旋转与 block-LDLQ 已通过 `R^T H R` 合并；seed 123/124 的同源码组合 PPL 为 10.136666/10.121536，均优于对应仅旋转 11.004825/11.147938 与仅 LDLQ 10.276133；见[两种子消融](../results/llama32_1b_w4_rotation_ldlq_ablation.md)。旋转模式暂不导出 NQMC。
- Completed WikiText-2 train Hessian calibration and 8-column E8 block-LDLQ. The 16/64-chunk W4 PPL values are 10.323714/10.276133. QA-LDLQ has since been implemented and evaluated; see the current QA stage result below.
- Completed A4 and W4A4KV4 quality simulation: batch-4 W4/W4KV4/W4A4KV4 PPL is 10.276601/10.732014/11.578171. QA-LDLQ was implemented and the epsilon/Hessian normalization mismatch was subsequently audited; the bounded corrected run did not improve over ordinary LDLQ, so this route is closed without post-hoc wider tuning.
- train64 block-LDLQ 已完成 NQMC 导出和无 Hessian 独立回载：文件 659,281,364 B，有效码率 4.267043 bit/量化参数，三条执行路径 PPL 精确一致为 10.276132955046；见 [LDLQ NQMC 报告](../results/llama32_1b_w4_ldlq_nqmc.md)。

## Shared calibration protocol (2026-10-22)

The versioned 128-window WikiText-2 train calibration manifest is now implemented and consumed by both the NestQuant Hessian collector and the QuaRot compatibility runner. It records model/data revisions, tokenization, seed, every start index, every window hash, the full train-token hash, and the combined selected-token hash. The combined hash `358a77b4...90bb` and all 128 starts match the existing formal QuaRot runs. A real one-window Llama-3.2-1B Hessian smoke completed successfully before the full run. Prior train64 results remain valid historical results but are no longer used as the primary W4 comparison.

The full all-layer 128-window Hessian recalibration and WikiText-2 W4 evaluation are complete. NestQuant block-LDLQ W4 scored 10.277472 when quantizing all 113 Linear layers and 10.211760 when retaining `lm_head` in BF16. The latter aligns calibration tokens, test tokens, and `lm_head` scope with the QuaRot compatibility GPTQ W4 result 10.508346. The same saved W4 checkpoint now has the full W4/W4KV4/W4A4/W4A4KV4 matrix: 10.211760/10.668994/10.952032/11.504096. Export-reload and an independent load-only process both reproduce the W4 PPL exactly. See [the reviewed progression](../results/llama32_1b_shared128_w4a4kv4_progression.md).

## QA-LDLQ stage result (2026-10-22)

Implemented the public QA weight adjustment, 20-train-chunk activation second-moment collection, QA NQMC export, and calibration-free reload. With the local covariance-scale Hessian and relative damping 0.01, the upstream default `epsilon_constant=155` degraded W4 PPL from 10.276601 to 11.053594 and W4A4KV4 from 11.578171 to 12.386133. This negative result requires an epsilon/Hessian-scale ablation before any claim of reproducing the paper QA gain. See [QA-LDLQ implementation](qa_ldlq.md) and [result](../results/llama32_1b_qa_ldlq_ablation.md).

## QA-LDLQ scale audit completion (2026-10-22)

The public Hessian divides the token Gram matrix by sequence count, while the local collector divides by activation-row count. At context length 2048 this created a 2048x normalization mismatch. A bounded representative-layer sweep showed that C=155 caused up to 73.7% weight adjustment and 9.07x proxy loss; corrected candidates removed the outlier. Full C=634880 results were W4 10.283140 and W4A4KV4 11.584234 versus ordinary LDLQ 10.276601 and 11.578171. The large regression was explained and removed, but no quality gain remained, so the QA route is closed without post-hoc wider tuning.

## Unified reproduction workflow (2026-10-22)

The versioned [`reproduction.v1.json`](../configs/reproduction.v1.json) and `nestquant-reproduce` entry point now connect preflight, unit/toy smoke, the six-step shared128 Llama matrix, reviewed-result validation, and raw-evidence validation. Full execution requires CUDA plus an exact confirmation phrase; completed artifact sets are skipped for resumability. The workflow was exercised on WSL2/RTX 5070: preflight, 95 tests, toy smoke, full dry-run planning, and all four existing raw-manifest protocol checks passed. See [the workflow guide](reproduction_workflow.md).

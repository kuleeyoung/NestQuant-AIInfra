# NestQuant 论文、作者代码与本项目实现对应矩阵

## 审计对象

- 论文：`NestQuant: Nested Lattice Quantization for Matrix Products and LLMs`，本地 PDF SHA256 `e98cedd57ff74971ef109214804d931cbcdc64c77153e884c1e4245ad03ebe88`；
- 论文版本：项目协议登记为 arXiv `2502.09720v3`；
- 作者仓库：`cookiedoth/nestquant`；
- 固定作者 commit：`6b269f84264afad990a09248d8eeb45d5f6248ff`；
- 本项目模型：Llama-3.2-1B Base，并非论文主表中的 Llama-3-8B；
- 审计日期：2026-10-23。

页码使用 PDF 文件页码。论文正文印刷页码可能比 PDF 页码小一页。无法由论文或固定 commit 确认的内容明确标为“无法确认”，不根据期望结果补全。

## 总体结论

本项目已经实现并测试 E8 最近点、嵌套 Voronoi codec、L2 行归一化、multi-beta、First-beta 动态规划、block-LDLQ、QA-LDLQ 公式适配、权重/激活/KV 质量仿真和 NQMC 真实权重文件。当前最主要的论文一致性差距是：

1. 正式 LLM 权重 beta 使用项目扩展候选，不等于固定上游默认 `[4.0, 5.5, 7.0, 18.0]`，也不等于论文合成实验的完整 `0.5×{1,...,50}` 网格；
2. 论文 §4.3 的局部 `W H` 权重与在线 `X H` 输入路径已经接入 Llama-3.2-1B W4、LDLQ Hessian 坐标变换和 NQMC 回载；尚未与固定作者代码逐层逐张量核对，也未覆盖非 2 次幂 Kronecker 路径；
3. 当前 A4/KV4 模块覆盖论文所述位置，但尚未与新的局部 Hadamard W4 路径组合；与固定上游的 `kron_h_ip`、无 dither try-all 路径及逐层 beta 优化仍需逐项对齐；
4. QA-LDLQ 公式已实现并完成 Hessian 尺度审计，但在本地 Llama-3.2-1B 配置上没有复现论文增益；
5. NQMC 提供真实低比特权重存储，运行时仍反量化到 BF16；A/KV 没有 packed 常驻格式或低比特 kernel。

因此当前准确定位是“核心算法复现、Llama-3.2-1B 适配与可审计扩展实验”，还不能表述为论文全部端到端实现完成。

## 逐模块矩阵

| 模块 | 论文依据 | 固定作者代码 | 本项目代码 | 状态 | 剩余差距 |
|---|---|---|---|---|---|
| E8 最近点 | PDF p.7 Algorithm 1/2；Appendix C | `e8.py` | `nestquant/core/lattice.py` | 已实现并用穷举/格点性质测试 | 性能 kernel 未复现 |
| 嵌套 Voronoi codec | PDF p.7 §4.2，`Λ/qΛ` residue 与 overload | `coder.py::encode/decode` | `nestquant/core/codec.py` | 已实现；q=2 全 residue 往返 | 上游 CUDA 性能不在当前范围 |
| L2 行归一化与 8 维分块 | PDF p.7 Algorithm 3 | `coder.py::encode_matrix*` | `nestquant/core/quantizer.py` | 已实现 | 需继续区分不同实验的 dither 语义 |
| 减性 dither | 论文未在 LLM 每条路径完整展开 | `e8.py::generate_dither`；矩阵编码默认开启，`parallel/quant_utils.py` 的 LLM quantsim 明确关闭 | codec 与 checkpoint 保存共享 realization/CPU RNG 状态 | 部分对齐 | 论文表格具体路径是否使用 dither，无法仅由正文确认 |
| multi-beta First-beta | PDF p.20 Appendix F | 作者 README 称 DP 脚本“will be added”；固定 commit 无完整 DP 入口 | `nestquant/core/beta_search.py` | 项目独立实现并与穷举核对 | 不能宣称逐行复用了作者 DP 代码 |
| 合成实验 beta 网格 | PDF p.9 §5.1：`0.5×{1,...,50}` 中选 k 个 | 固定代码允许外部传入 betas | 合成实验支持该尺度映射 | 基本对齐 | 论文码率公式符号存在疑点，项目同时保留原式与 Shannon 解释 |
| LLM 权重 beta | PDF p.10：q=14、k=4；正文不足以确认每层值 | `parallel/w_quant.py` 默认 `[4.0,5.5,7.0,18.0]` | 正式候选含 `[3.5,4.5,6.0,14.5,25.0,40.0,50.0,100.0]`；100 只用于 QA outlier | 项目扩展 | 需区分 upstream default、paper grid 与 project extended |
| 权重—激活匹配旋转 | PDF p.7 §4.3：每个 Linear 使用匹配 H | `parallel/w_quant.py` 与 `parallel/quant_utils.py` 使用 `kron_h_ip` | `models/rotated_linear.py` 保存 `W H` 并在前向在线计算 `X H`；W4、LDLQ 坐标和 NQMC 元数据已接入 | Llama-3.2-1B W4 路径已实现 | 需跑完整 PPL、组合 A4，并与固定上游按层比较 |
| 非 2 次幂旋转 | PDF p.7 §4.3：`H1⊗H2`；p.21-22 Table 7 | `hadamard.py::kron_h_ip` 对固定模型尺寸提供硬编码 Hadamard | 当前核心旋转只支持 2 次幂；Llama-3.2-1B 的 2048/8192 输入维度均可覆盖 | 当前目标模型不阻塞 | 若扩展到论文 Llama-3-8B 的 14336 维 down projection，需实现并核对 Kronecker 路径 |
| Q/K 旋转与 K/V 量化 | PDF p.8-9 §4.3/Fig.4 | `parallel/model.py` post-RoPE 对 Q/K 做 head-dim Hadamard，量化 K/V | `evaluation/kv_cache.py` | 质量路径已实现 | 需做固定上游逐张量等价测试；当前 cache 存浮点重构值 |
| 激活量化 | PDF p.7-9：Linear 前旋转并量化 | `parallel/quant_utils.py` forward-pre-hook | `evaluation/activation_quantization.py` | 质量仿真已实现 | beta、旋转与列并行分类需逐层对齐上游 |
| A/K/V 默认 beta | 论文正文确认逐层 DP，正文未给 CLI 默认值 | `parallel/ppl.py`：A `[3.47,4.74,6.90,18.11]`，K `[3.50,4.58,6.47,17.06]`，V `[3.53,5.59,9.62,29.03]` | 版本化 KV beta 配置与项目 A beta 常量 | 部分对齐 | 配置需标注 upstream CLI default 或项目校准 |
| block-LDLQ | PDF p.8 §4.3、Appendix B | `coder.py::block_LDL`、`encode_matrix_LDLQ_*` | `core/ldlq.py`、分组 Hessian store | 已实现并完成 shared128 结果 | 归一化口径需持续记录 |
| QA-LDLQ | PDF p.8-9 §4.5，`W~=WH(H+J)^-1` | `parallel/quant_utils.py` 的 `eps` 调整；`parallel/err.py` | QA 权重调整、激活二阶矩与尺度审计 | 公式实现完成，结果为负 | 本地未复现论文增益 |
| 实际码率 | PDF p.9 合成公式；p.10 LLM 使用 zstd 压 beta indices | 公开路径保存量化模型 | NQPK/NQRT/NQMC 实测字节和 SHA256 | 权重存储已证明 | 未实现论文相同的 zstd beta 流与低比特执行布局 |
| 低比特点积 | PDF p.7 Algorithm 4 | 固定仓库有 CUDA/Hadamard 依赖，公开 Python 路径主要是 simulation | 未实现 INT4/E8 fused dot-product kernel | 未完成 | 不能声称吞吐、延迟或低比特常驻显存收益 |
| PPL split | PDF p.10 写 WikiText-2 validation | `parallel/ppl.py` 使用 test | 本项目使用 test | 已披露偏差 | 无法确认论文表格最终入口为何与正文不同 |
| zero-shot | PDF p.10 列 ARC-Easy/Challenge、HellaSwag、Winogrande | 固定 README 表示 zero-shot code 将来加入 | lm-eval PIQA/HellaSwag 路径已开发但正式运行暂缓 | 暂缓 | PIQA 不是论文任务，不能称论文 zero-shot 复现 |

## 旋转问题的当前状态与后续路线

项目已经完成 2 次幂维度下“离线旋转权重 + 在线旋转输入”的严格恒等测试，并把该结构接入 Llama-3.2-1B 的 112 个 Linear。首个 2,047-token smoke 已验证 W4 内存路径、流式 NQMC 导出、立即回载和独立进程回载数值一致；它不是正式质量结论。

后续按下面顺序推进：

1. 运行完整 288,627-token WikiText-2 PPL，确认局部 Hadamard W4 的正式质量；
2. 将该路径与 A4 组合，并明确在线 Hadamard 与激活量化的先后和尺度语义；
3. 对固定上游和本项目的量化前张量、旋转后张量、重构张量逐层比较；
4. 扩展到非 2 次幂模型时，实现并核对 `H1⊗H2` Kronecker 路径；
5. Q/K 的 post-RoPE 旋转、KV4 和低比特 kernel 保持为独立验证项。

这条路线先解决 NestQuant 自身的语义一致性，不把 QuaRot 的全模型变换结构强加给 NestQuant。

## 仍无法确认的信息

- 论文正文与固定公开代码在 WikiText-2 validation/test 上不一致，无法确认论文表格实际使用的私有运行入口；
- 固定作者仓库没有公开动态规划脚本，因此无法逐行核对 Appendix F 的作者实现；
- 论文正文不足以确认每层最终选中的 weight/A/K/V beta，公开 CLI 默认值不能自动等同论文表格配置；
- 论文低比特点积的完整生产 kernel、布局和真实性能条件无法由当前公开 Python 路径完整确认。

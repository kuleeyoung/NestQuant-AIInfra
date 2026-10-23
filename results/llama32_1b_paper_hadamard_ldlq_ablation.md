# Llama-3.2-1B 论文式 Hadamard 与 shared128 LDLQ 组合消融

## 结论

论文式局部 `W H`/在线 `X H` 与 shared128 block-LDLQ 的收益可以叠加。完整 WikiText-2 test 上，组合路径 PPL 为 **10.076729**：

- 相对同协议普通 W4 的 11.054059，下降 **0.977330**，相对下降 **8.841%**；
- 相对 shared128 LDLQ W4 的 10.211760，继续下降 **0.135031**，相对下降 **1.322%**；
- 相对 BF16 的差距从普通 W4 的 1.295947 缩小到 0.318617，消除 **75.414%** 的 PPL 差距。

Hadamard 单独带来 0.136964 的改善，shared128 LDLQ 单独带来 0.842299 的改善；两项简单相加为 0.979264，实测组合改善为 0.977330，交互项仅 -0.001933 PPL。这说明在当前协议中两种收益近似可加，而不是明显互相覆盖。

## 固定协议

- 模型：`meta-llama/Llama-3.2-1B` Base，revision `4e20de362430cd3b72f300e6b0f18e50e7166e08`；
- 测试：WikiText-2 test，141 个 2048-token 窗口，288,627 个预测 token；
- 测试 token SHA-256：`54927bf19e6e64b4008eeb93370c370fb25db18cf68eac8015cfe094c464d567`；
- 范围：112 个 Llama Linear 权重，tied `lm_head` 保持 BF16；
- NestQuant：E8、`q=14`、8 个项目候选中每层选择 4 个 beta、CPU dither seed 58；
- LDLQ：8 列 block-LDLQ、相对阻尼 0.01；
- 校准：WikiText-2 train 128 个窗口，共 262,144 token；
- 校准 token SHA-256：`358a77b403d372a11957df15dbf46ef3e6f54025d315eaf3514c40bd561d90bb`；
- Hadamard：112 个 Linear 保存 `W H`，前向在线计算 `X H`；LDLQ Hessian 同步变换到 Hadamard 输入坐标；
- A4、KV4、QA-LDLQ：关闭。

## 四因素结果

| 路径 | Hadamard | shared128 LDLQ | PPL | 相对普通 W4 改善 |
|---|:---:|:---:|---:|---:|
| BF16 | — | — | 9.758112 | — |
| 普通 NestQuant W4 | 否 | 否 | 11.054059 | — |
| 论文式 Hadamard W4 | 是 | 否 | 10.917094 | 0.136964 |
| shared128 LDLQ W4 | 否 | 是 | 10.211760 | 0.842299 |
| 论文式 Hadamard + shared128 LDLQ W4 | 是 | 是 | **10.076729** | **0.977330** |

四个量化路径保持相同测试 token、W4 范围、beta 候选和 seed。BF16 来自同一 WSL/CUDA 评测协议。组合运行的平均 NLL 为 2.310229，完整量化、流式导出、回载和评测耗时 838.7 秒；该耗时不是推理延迟。

## 自包含 NQMC

组合路径已经导出并立即回载：

- 文件：`llama32_1b_paper_hadamard_w4_shared128_ldlq.nqmc`；
- 文件大小：1,044,561,090 B；
- SHA-256：`f594a20cd2f2abfed88fc1bdeabe8efb7a71e4bef5509933a74ab327a841f821`；
- encoder：`block_ldlq`；
- Hadamard 格式：`local_hadamard_linear_v1`；
- 112 个量化张量；
- NQMC 记录 shared128 校准来源、Hessian 文件哈希、旋转模块集合、beta、范数和 dither 状态。

解码不需要再次加载约 5.12 GB Hessian；Hessian 只在 LDLQ 编码阶段使用。运行时仍按记录反量化到模型浮点 dtype 并执行 BF16 Linear。

## 结论边界

- 可以陈述局部 Hadamard 与 shared128 LDLQ 在当前 W4 协议下近似可加，并报告完整 PPL；
- 可以陈述真实 NQMC 文件大小、哈希、流式导出和立即回载；
- 不能把 838.7 秒解释为推理速度；
- 没有 INT4/E8 GEMM，不能声称低比特执行或 tokens/s 加速；
- A4/KV4 尚未接入该局部在线 Hadamard 路径；
- 结果属于 Llama-3.2-1B 本地复现，不等于论文 Llama-3-8B 主表。

## 原始证据

- 组合运行：`results/raw/llama32_1b_paper_hadamard_w4_shared128_ldlq_full.json`；
- 组合检查点：`results/raw/llama32_1b_paper_hadamard_w4_shared128_ldlq.nqmc`；
- 机器可读汇总：`results/llama32_1b_paper_hadamard_ldlq_ablation.json`。

# Llama-3.2-1B 论文式 Hadamard NestQuant W4 完整 PPL

## 结论

在固定模型、完整 WikiText-2 test token、量化范围、beta 候选、dither seed 和 BF16 `lm_head` 的条件下，论文 §4.3 风格的局部 `W H`/在线 `X H` 将 NestQuant W4 PPL 从 **11.054059** 降至 **10.917094**。绝对下降 **0.136964**，相对下降 **1.239%**；平均每 token NLL 从 2.402798 降至 2.390330。

这是当前实现中局部 Hadamard 旋转的正式完整质量结果。它表明旋转在无 LDLQ 的 W4 路径上有小幅正收益，但不能直接推出加入 LDLQ、A4 或 KV4 后仍有同样收益。

## 固定协议

- 模型：`meta-llama/Llama-3.2-1B` Base，revision `4e20de362430cd3b72f300e6b0f18e50e7166e08`；
- 数据：`Salesforce/wikitext` / `wikitext-2-raw-v1` test，revision `b08601e04326c79dfdd32d625aee71d232d685c3`；
- 上下文长度：2048；141 个完整窗口；288,627 个预测 token；
- token ID SHA-256：`54927bf19e6e64b4008eeb93370c370fb25db18cf68eac8015cfe094c464d567`；
- 量化：E8、`q=14`、每层从 8 个项目候选中用 DP 选 4 个 beta；CPU dither seed 58；
- beta 系数候选：`[3.5,4.5,6.0,14.5,25.0,40.0,50.0,100.0]`；
- 范围：112 个 Llama Linear 权重，tied `lm_head` 保持 BF16；
- 两组都不使用 LDLQ、QA-LDLQ、A4 或 KV4；
- 代码提交：`6173e6580e44a124151d04f376c2d3e240031c03`；
- Python 源码 SHA-256：`ec8146b0ff79c0d24818989d6ec0c956738ac8f71115504b33cea537ee6dca33`。

## 完整结果

| 路径 | PPL | 平均 NLL | 相对普通 W4 | 峰值 CUDA allocated |
|---|---:|---:|---:|---:|
| 普通 NestQuant W4 | 11.054059 | 2.402798 | — | 3,066,206,720 B |
| 论文式局部 Hadamard NestQuant W4 | **10.917094** | **2.390330** | **-0.136964 PPL（-1.239%）** | 3,065,830,912 B |

普通 W4 运行耗时 92.4 秒，包含模型加载、量化和完整 PPL。Hadamard W4 本次从已经验证的 NQMC 加载，67.6 秒为加载与完整 PPL 时间，因此两者耗时不能用于推理速度比较。

## 检查点证据

Hadamard 路径从 NQMC v1 独立回载后评测：

- 文件大小：1,044,559,427 B；
- SHA-256：`942109f1475defebf8e6eee2d8f74d5abefa525e292e7f3a662e470ebb7e7da3`；
- 112 个 quantized tensor；
- 147 个 tensor/alias 记录；
- 旋转格式：`local_hadamard_linear_v1`；
- 运行时按记录反量化到模型浮点 dtype，再执行 BF16 Linear。

首窗口内存模型、立即回载和独立进程回载曾取得完全一致的 8.064950 PPL；本次完整 PPL 进一步确认自包含检查点能够覆盖全部 141 个窗口。

## 能说明什么

- 本项目已经把论文式局部权重—激活匹配旋转接入真实 Llama-3.2-1B W4，而不只是小张量等价测试；
- 在当前无 LDLQ 的相同 W4 协议中，局部 Hadamard 带来 1.239% 的 PPL 相对改善；
- NQMC 可以保存旋转模块结构和低比特权重索引，并在新进程中恢复后得到完整质量结果。

## 不能说明什么

- 运行时权重仍反量化为浮点并执行 BF16 Linear，因此不能声称 INT4/E8 kernel 加速；
- 没有同时测量 Hadamard + shared128 LDLQ，不能把该收益外推到当前主表的 LDLQ W4；
- A4/KV4 还没有与该在线 Hadamard 路径组合；
- 项目的 beta 候选包含明确披露的扩展值，不等于论文未公开的逐层最终 beta；
- 这是 Llama-3.2-1B 的本地复现结果，不是论文 Llama-3-8B 主表数字。

## 原始证据

- 普通 W4：`results/raw/llama32_1b_standard_w4_no_lm_head_full_control.json`；
- Hadamard W4：`results/raw/llama32_1b_paper_hadamard_w4_full.json`；
- 可提交机器摘要：`results/llama32_1b_paper_hadamard_w4_full.json`。

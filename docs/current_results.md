# 当前正式结果与历史结果分层

本页是项目结果的唯一推荐入口。它不覆盖历史实验，而是说明当前应引用哪一组结果。历史文档保留当时的代码、协议和结论，不应用其中较旧的数字替代当前正式结果。

## 当前主协议

- 模型：`meta-llama/Llama-3.2-1B` Base，revision `4e20de362430cd3b72f300e6b0f18e50e7166e08`；
- 数据：`Salesforce/wikitext` / `wikitext-2-raw-v1` test，revision `b08601e04326c79dfdd32d625aee71d232d685c3`；
- 评测：context 2048，141 个完整窗口，288,627 个预测 token；
- 测试 token SHA256：`54927bf19e6e64b4008eeb93370c370fb25db18cf68eac8015cfe094c464d567`；
- 校准：固定的 128 个 WikiText-2 train 窗口，选中 token SHA256 `358a77b403d372a11957df15dbf46ef3e6f54025d315eaf3514c40bd561d90bb`；
- 公平范围：112 个 Llama `Linear` 权重使用 W4，`lm_head` 保持 BF16；
- 执行：W4 从低比特 NQMC 回载后反量化为 BF16 Linear；A4/KV4 为浮点重构质量仿真。

## 推荐引用结果

| 结论 | 当前值 | 推荐来源 | 证据状态 |
|---|---:|---|---|
| BF16 PPL | 9.758112 | [共享校准递进报告](../results/llama32_1b_shared128_w4a4kv4_progression.md)的同协议基线 | 正式质量基线 |
| NestQuant W4 PPL | 10.211760 | [共享校准递进报告](../results/llama32_1b_shared128_w4a4kv4_progression.md) | 正式质量结果；BF16 `lm_head` |
| NestQuant W4KV4 PPL | 10.668994 | 同上 | 正式质量仿真 |
| NestQuant W4A4 PPL | 10.952032 | 同上 | 正式质量仿真 |
| NestQuant W4A4KV4 PPL | 11.504096 | 同上 | 正式质量仿真 |
| 论文式 Hadamard W4 PPL | 10.917094 | [局部 Hadamard W4 完整报告](../results/llama32_1b_paper_hadamard_w4_full.md) | 同协议普通 W4 为 11.054059；无 LDLQ |
| QuaRot 兼容移植 GPTQ W4 PPL | 10.508346 | [QuaRot 对照](../results/llama32_1b_quarot_comparison.md) | 同校准/测试兼容移植，不是官方 Llama-3.2 结果 |
| QuaRot 兼容移植 W4A4KV4 PPL | 21.095586 | 同上 | fake-quant 兼容移植 |
| 共享校准 W4 NQMC | 1,044,557,140 B | [共享校准递进报告](../results/llama32_1b_shared128_w4a4kv4_progression.md) | 真实文件和独立回载；不证明低比特执行 |
| QA-LDLQ 修正尺度 W4/W4A4KV4 | 10.283140 / 11.584234 | [QA-LDLQ 报告](../results/llama32_1b_qa_ldlq_ablation.md) | 负结果；未优于普通 LDLQ |

BF16 的 9.758112 来自统一 WSL/CUDA 基线；早期 9.757045 和 QuaRot 兼容路径的 9.762884 属于不同运行路径。它们可以用于各自历史消融，但不能在当前主表中互相替换。

## 历史但仍有解释价值的结果

| 结果 | 用途 | 为什么不作为当前主结果 |
|---|---|---|
| 首轮 BF16/RTN/NestQuant W4 | 展示项目从 RTN 到 E8 的早期验证 | 尚未使用 shared128 LDLQ 和统一 `lm_head` 范围 |
| 普通 W4 NQMC 659,279,649 B | 验证第一版模型容器、码率和回载 | 权重算法与当前 shared128 block-LDLQ 不同 |
| train64 block-LDLQ W4 10.276133 | 展示校准规模消融与独立回载 | 当前公平协议使用 shared128 和 BF16 `lm_head` |
| 两个 seed 的逐 Linear 旋转消融 | 证明项目定义的旋转可与 LDLQ 叠加 | 不是论文所有运行时旋转语义的逐项复现 |
| QA-LDLQ `C=155` 退化结果 | 展示 Hessian 尺度错误的定位过程 | 已由尺度审计解释，不能作为最终 QA 配置 |

## 暂缓结果

PIQA/HellaSwag 完整评测已暂缓。相关脚本和严格汇总器保留，但没有正式分数，不能在 README、简历或报告中预填结论。

## 表述边界

- 可以陈述真实 NQMC 文件字节数、哈希、独立回载和 PPL 一致性；
- 可以陈述论文式 `W H`/`X H` 已完成 112 层、自包含 NQMC、独立回载和完整 288,627-token PPL；同协议无 LDLQ W4 中 PPL 从 11.054059 降至 10.917094；
- 可以陈述固定协议下的 PPL 与消融；
- A4/KV4 仍是质量仿真，不能声称低比特张量常驻显存；
- 没有 INT4 GEMM 或融合 kernel，不能声称推理加速；
- QuaRot 数字来自有明确补丁的兼容移植，不能写成官方发布的 Llama-3.2-1B 结果；
- 当前实现覆盖论文核心模块，但仍存在 beta 来源、旋转语义和运行时存储方面的差距，不能写成全文逐项复现完成。

# 当前正式结果与历史结果分层

本页是项目结果的唯一推荐入口。它不覆盖历史实验，而是说明当前应引用哪一组结果。历史文档保留当时的代码、协议和结论，不应用其中较旧的数字替代当前正式结果。

> Codec 版本说明：NQRT v3 `e8_upstream_half_coset_tie_v1` 的 W4/W4KV4/W4A4/W4A4KV4 全量矩阵已经完成。NQRT v2 legacy 结果继续保留用于历史复现，不再作为当前首选数字。详见 [v3 完整递进](../results/llama32_1b_paper_hadamard_w4a4kv4_nqrt3_progression.md)和[固定上游对齐报告](../results/fixed_upstream_codec_alignment.md)。

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
| 论文式 Hadamard + shared128 LDLQ W4 PPL | **10.076886** | [NQRT v3 上游语义端点](../results/llama32_1b_paper_hadamard_shared128_ldlq_nqrt3.md) | 当前首选 W4；112/112 记录为上游 E8 语义；真实 NQMC 回载 |
| 论文式 Hadamard + shared128 LDLQ W4A4 PPL | **10.689190** | [NQRT v3 完整递进](../results/llama32_1b_paper_hadamard_w4a4kv4_nqrt3_progression.md) | 固定上游全局 A beta；局部 `X H` 坐标 fake quant |
| 论文式 Hadamard + shared128 LDLQ W4KV4/W4A4KV4 PPL | **10.515783 / 11.232054** | [NQRT v3 完整递进](../results/llama32_1b_paper_hadamard_w4a4kv4_nqrt3_progression.md) | 同一 v3 NQMC；A4/KV4 为浮点重建质量仿真 |
| A4 beta held-out 选择 | 保留上游默认 | [独立 train held-out 报告](../results/llama32_1b_a4_beta_holdout.md) | 新 module 最优 PPL 10.270711，仍差于默认 9.782117；未读取 test 选配置 |
| QuaRot 兼容移植 GPTQ W4 PPL | 10.508346 | [QuaRot 对照](../results/llama32_1b_quarot_comparison.md) | 同校准/测试兼容移植，不是官方 Llama-3.2 结果 |
| QuaRot 兼容移植 W4A4KV4 PPL | 21.095586 | 同上 | fake-quant 兼容移植 |
| 共享校准 W4 NQMC | 1,044,557,140 B | [共享校准递进报告](../results/llama32_1b_shared128_w4a4kv4_progression.md) | 真实文件和独立回载；不证明低比特执行 |
| QA-LDLQ 修正尺度 W4/W4A4KV4 | 10.283140 / 11.584234 | [QA-LDLQ 报告](../results/llama32_1b_qa_ldlq_ablation.md) | 负结果；未优于普通 LDLQ |
| PIQA BF16/W4/W4A4KV4 `acc_norm` | 0.7481 / 0.7367 / 0.7078 | [下游评测报告](../results/llama32_1b_downstream_full.md) | 1,838 题；协议与逐样本身份校验通过 |
| HellaSwag BF16/W4/W4A4KV4 `acc_norm` | 0.6423 / 0.6282 / 0.5996 | [下游评测报告](../results/llama32_1b_downstream_full.md) | 10,042 题；协议与逐样本身份校验通过 |

BF16 的 9.758112 来自统一 WSL/CUDA 基线；早期 9.757045 和 QuaRot 兼容路径的 9.762884 属于不同运行路径。它们可以用于各自历史消融，但不能在当前主表中互相替换。

## 历史但仍有解释价值的结果

| 结果 | 用途 | 为什么不作为当前主结果 |
|---|---|---|
| 首轮 BF16/RTN/NestQuant W4 | 展示项目从 RTN 到 E8 的早期验证 | 尚未使用 shared128 LDLQ 和统一 `lm_head` 范围 |
| 普通 W4 NQMC 659,279,649 B | 验证第一版模型容器、码率和回载 | 权重算法与当前 shared128 block-LDLQ 不同 |
| train64 block-LDLQ W4 10.276133 | 展示校准规模消融与独立回载 | 当前公平协议使用 shared128 和 BF16 `lm_head` |
| 两个 seed 的逐 Linear 旋转消融 | 证明项目定义的旋转可与 LDLQ 叠加 | 不是论文所有运行时旋转语义的逐项复现 |
| QA-LDLQ `C=155` 退化结果 | 展示 Hessian 尺度错误的定位过程 | 已由尺度审计解释，不能作为最终 QA 配置 |

## 下游评测解释

PIQA/HellaSwag 完整评测已于 2026-10-24 完成。W4 相对 BF16 的 `acc_norm` 分别下降 1.1425/1.4141 个百分点；W4A4KV4 分别下降 4.0261/4.2721 个百分点。W4A4KV4 用时 15,633.48 秒，而 BF16/W4 分别为 257.46/264.34 秒；该差异反映当前 Python/PyTorch fake-quant 参考实现的重复量化、解码和 kernel launch 成本，不代表生产低比特 kernel 的性能。

当前三档结果只能说明 A4 与 KV4 联合启用后的增量退化，不能把全部差值归因于 A4 或 KV4 单项。若要分离来源，需要在相同逐样本协议下补 W4A4 与 W4KV4 消融。

## 表述边界

- 可以陈述真实 NQMC 文件字节数、哈希、独立回载和 PPL 一致性；
- 可以陈述论文式 `W H`/`X H` 已完成 112 层、自包含 NQMC、独立回载和完整 288,627-token PPL；同协议无 LDLQ W4 中 PPL 从 11.054059 降至 10.917094；
- 可以陈述上游语义 NQRT v3 的论文式 Hadamard + shared128 LDLQ 组合 PPL 为 10.076886；历史 legacy 四因素消融仍支持两项收益近似可加；
- 可以陈述 NQRT v3 组合检查点接入局部坐标 A4 后 PPL 为 10.689190；A4 relative squared error 为 0.6390%，overload 为 5.7126%；
- 可以陈述固定协议下的 PPL 与消融；
- 可以陈述 PIQA/HellaSwag 全量 zero-shot 下游质量和逐题正确性翻转；PIQA 不是论文 zero-shot 主表任务，不能写成论文主表复现；
- A4/KV4 仍是质量仿真，不能声称低比特张量常驻显存；
- 已有 packed NQRT 解码融合 BF16 Tensor Core dot 的真实 block 原型；100 次同协议测量中它相对标量 packed block 加速 1.57～1.73×，但仍慢于 BF16 常驻 block，不能声称 INT4 GEMM、端到端模型加速或生产吞吐提升；
- 长 prefill 的大 M tile 分派在 seq 32/64/128 上相对标量 packed block 加速 2.31×/3.80×/6.16×，同时仍比 BF16 常驻 block 慢 21.9%/26.7%/45.3%；可以陈述 packed runtime 优化收益，不能改写为 BF16 加速收益；
- QuaRot 数字来自有明确补丁的兼容移植，不能写成官方发布的 Llama-3.2-1B 结果；
- 当前实现覆盖论文核心模块，但仍存在 beta 来源、旋转语义和运行时存储方面的差距，不能写成全文逐项复现完成。

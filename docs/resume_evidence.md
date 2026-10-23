# 简历证据边界

## 当前可以陈述

- 独立建立 NestQuant 复现项目，已实现 E8 最近格点及嵌套 Voronoi 编码/解码参考模块。
- 实现了给定 beta 候选集的行归一化与 8 维分块量化参考流程。
- 实现 Appendix F First-beta β 子集动态规划，并用小规模穷举子集搜索核对最优误差。
- 用小规模独立穷举核对 E8 最近点距离；穷举验证 q=2 的 256 个码字可编码/解码往返，并测试零行和候选 beta 增多时最小重构误差不增。
- 已在 CPU 和 NVIDIA RTX 5070 CUDA 环境运行 E8 toy demo。
- 已在 Llama-3.2-1B Base、WikiText-2 raw test 上完成 BF16/RTN W4/NestQuant W4 首轮完整 fake-quant PPL 对比；PPL 分别为 9.7570 / 13.0215 / 11.1494，结果 manifest 与协议偏差说明已归档在 `results/`。
- 已实现 signed-Hadamard 权重旋转消融；在同源码、同 token 与同量化协议下，无旋转 PPL 为 11.2026，两个旋转 seed 分别为 11.0048 和 11.1479。该结果只证明质量仿真差异。
- 已实现论文式局部 `W H` 权重与在线 `X H` 输入路径、自包含 NQMC 和独立回载；完整 288,627-token 同协议对照中，普通 W4 PPL 为 11.054059，局部 Hadamard W4 为 10.917094，下降 1.239%。运行时仍为 BF16 Linear，不能写成低比特推理加速。
- 已实现校准激活二阶统计、8×8 block-LDL 分解、E8 块误差反馈和分组 Hessian 惰性加载；在相同源码与测试 token 下，16/64 个 WikiText-2 train 序列校准的 LDLQ 将 Llama-3.2-1B NestQuant W4 PPL 从 11.2026 分别降至 10.3237/10.2761，train-64 消除了 64.09% 的 BF16 PPL 差距。
- 已将 train64 block-LDLQ 的最终 code、beta index、行范数和 dither 状态写入 NQMC：实测文件 659,281,364 B、4.267043 bit/量化参数；无需 4.77 GiB Hessian 即可独立回载，完整 PPL 与内存量化精确一致为 10.276133。
- 已实现旋转坐标下的 `H_rot=R^T H R` 并完成同源码两种子消融：seed 123/124 的旋转+train64 block-LDLQ PPL 为 10.1367/10.1215，均优于对应纯旋转 11.0048/11.1479 和仅 LDLQ 10.2761，相对普通 W4 分别消除 73.74%/74.78% 的 BF16 差距。

上述 PPL 结果仅支持本项目当前量化范围和 beta 候选设置。NQMC 已提供真实低比特存储证据，但不等于完成 NestQuant 全文复现；不能据此声称等码率优越性、低比特常驻显存或推理加速。

## 达到对应证据后才能写

| 简历表述方向 | 必须先拿到的证据 |
|---|---|
| Reproduced core W4/W4KV4/W4A4KV4 modules | Project-scope implementation is complete for nested coding, multi-beta, rotation ablations, LDLQ, A4/KV4 quality simulation, and QA-LDLQ adaptation. Batch-4 W4/W4KV4/W4A4KV4 PPL is 10.276601/10.732014/11.578171. Default QA epsilon constant 155 was a documented negative result. Runtime A/KV tensors are still reconstructed floating point and the complete paper rotation graph and low-bit kernels are absent. |
| QuaRot same-task compatibility comparison | Ported the official QuaRot fake-quant path to Llama-3.2-1B/RTX 5070 with a pure-PyTorch Hadamard fallback, GQA-aware K-cache reshape, tied embedding/lm_head handling, and Transformers 4.46 GPTQ device placement. Under the same model revision, WikiText-2 test-token hash and 288,627-token NLL aggregator, rotated BF16/GPTQ W4/W4KV4/W4A4/W4A4KV4 PPL is 9.762884/10.508346/12.269130/15.850334/21.095586. This is a documented compatibility port and quality simulation, not the official CUDA kernel or an inference-speed result. |
| “模型困惑度改善/接近论文” | 同一模型 revision、tokenizer、评测 split、文本处理和 PPL 聚合；保存 run manifest、token hash 和日志 |
| “模型压缩到约 4.27 bit/量化参数 / NQMC 为 BF16 逻辑载荷的 26.67%” | **已具备**：实际文件、byte 级核算、SHA-256、独立回载与完整 PPL；表述时保留“BF16 分母是逻辑载荷而非另一个实测文件”的限定 |
| “显存下降 / 推理加速 X%” | 同机、同输入、同 batch/context、同质量口径的峰值显存和延迟/吞吐 benchmark；同步 GPU 并报告统计区间 |
| “服务部署能力” | 可调用的部署入口、并发/稳定性/错误处理和资源监控证据 |

任何数值都必须可追溯到配置、运行命令、日志和版本。结果不理想也保留，可作为取舍分析；不得用论文表格数字冒充本地结果。

## 后续可用的简历句式模板

> 独立实现并验证 NestQuant 的 [已完成模块]，在 [模型与数据协议] 上对比 [同范围基线]；将量化质量、实际存储开销与推理性能分开测量，获得 [本地测得指标与 trade-off]，完整保存配置、版本和复现实验脚本。

方括号内容必须在实验完成后填入，不根据目标或论文参考值预填。

## QA-LDLQ stage result (2026-10-22)

Implemented the public QA weight adjustment, 20-train-chunk activation second-moment collection, QA NQMC export, and calibration-free reload. With the local covariance-scale Hessian and relative damping 0.01, the upstream default `epsilon_constant=155` degraded W4 PPL from 10.276601 to 11.053594 and W4A4KV4 from 11.578171 to 12.386133. This negative result requires an epsilon/Hessian-scale ablation before any claim of reproducing the paper QA gain. See [QA-LDLQ implementation](qa_ldlq.md) and [result](../results/llama32_1b_qa_ldlq_ablation.md).

## QA-LDLQ scale audit completion (2026-10-22)

The public Hessian divides the token Gram matrix by sequence count, while the local collector divides by activation-row count. At context length 2048 this created a 2048x normalization mismatch. A bounded representative-layer sweep showed that C=155 caused up to 73.7% weight adjustment and 9.07x proxy loss; corrected candidates removed the outlier. Full C=634880 results were W4 10.283140 and W4A4KV4 11.584234 versus ordinary LDLQ 10.276601 and 11.578171. The large regression was explained and removed, but no quality gain remained, so the QA route is closed without post-hoc wider tuning.

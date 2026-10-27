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
- 已验证论文式 Hadamard 与 shared128 block-LDLQ 可以叠加：上游 E8 语义 NQRT v3 组合 W4 PPL 为 10.076886；1,044,566,690 B 的 NQMC 中 112/112 个量化记录携带明确 codec 语义，并完成全量回载评测。历史 legacy 消融 PPL 为 10.076729，继续用于说明 Hadamard 与 LDLQ 的近似可加性。
- 在同一 NQRT v3 checkpoint 上完成 W4/W4KV4/W4A4/W4A4KV4 全量矩阵，PPL 为 10.076886/10.515783/10.689190/11.232054；用 NLL 分解确认 A4 损失大于 KV4，联合交互项为 +0.006906。A4/KV4 仍是浮点重建质量仿真，不宣称低比特 kernel 加速。
- 实现 bounded-memory NQMC inspector，在不加载模型或反量化权重的情况下，对 1.04 GB checkpoint 完成 payload/file SHA、146 个唯一 record 和 112 个 NQRT v3 量化记录审计；实测量化 record 码率为 4.267428 bit/parameter，并明确整体 42.2623% BF16 逻辑载荷比例受 dense tied embedding 影响。
- 为 NQMC 增加 O(1) 按 tensor 名称/alias 定位与选择性 CUDA 解码；在独立进程中测得 2048×2048 q_proj 和 2048×8192 down_proj 的 record read/decode mean 为 5.87/71.56 ms 与 27.94/337.43 ms，并据峰值内存定位当前 float32 E8 解码中间张量为后续 fused runtime 的主要瓶颈。
- 实现 row-range packed bit 解包和 bit-exact 分块 BF16 解码；在真实 down_proj 上保持与 full-float32→BF16 逐元素一致，将单 tensor CUDA decode peak 从 850.01 MiB 降至 119.13 MiB（-85.99%），解码均值 337.43→338.25 ms。该结果只属于 standalone decode，不外推为端到端推理收益。
- 已实现校准激活二阶统计、8×8 block-LDL 分解、E8 块误差反馈和分组 Hessian 惰性加载；在相同源码与测试 token 下，16/64 个 WikiText-2 train 序列校准的 LDLQ 将 Llama-3.2-1B NestQuant W4 PPL 从 11.2026 分别降至 10.3237/10.2761，train-64 消除了 64.09% 的 BF16 PPL 差距。
- 已将 train64 block-LDLQ 的最终 code、beta index、行范数和 dither 状态写入 NQMC：实测文件 659,281,364 B、4.267043 bit/量化参数；无需 4.77 GiB Hessian 即可独立回载，完整 PPL 与内存量化精确一致为 10.276133。
- 已实现旋转坐标下的 `H_rot=R^T H R` 并完成同源码两种子消融：seed 123/124 的旋转+train64 block-LDLQ PPL 为 10.1367/10.1215，均优于对应纯旋转 11.0048/11.1479 和仅 LDLQ 10.2761，相对普通 W4 分别消除 73.74%/74.78% 的 BF16 差距。

- 已在固定 Llama-3.2-1B、`lm_eval==0.4.13`、zero-shot 协议下完成 PIQA 1,838 题和 HellaSwag 10,042 题全量评测，并核对模型 revision、数据指纹、任务定义哈希和逐样本身份。NQMC W4 相对 BF16 的 `acc_norm` 分别下降 1.14/1.41 个百分点，W4A4KV4 分别下降 4.03/4.27 个百分点。
- W4A4KV4 下游参考路径耗时 15,633.48 秒，BF16/W4 为 257.46/264.34 秒。该约 59 倍差异只可用于定位 Python/PyTorch fake-quant 开销和说明 fused kernel 的工程必要性，不能写成 NestQuant 低比特 kernel 性能。
- 已把 Llama-3.2-1B 的 112 个 Linear 接入 NQRT4 packed-resident runtime：kernel 内恢复 E8 block、转换为 BF16 tile 并使用 Tensor Core 累加，不物化完整 BF16 权重。完整模型 warm CUDA allocation 从 2,390.1 MB 降至 1,028.6 MB（-57.0%）；当前延迟仍慢于 BF16 resident，因此只能陈述显存—延迟权衡，不能陈述端到端加速。
- 已实现固定 batch 可复用 CUDA Graph decode runner。与 packed eager 相比，batch 1/16 的 decode-forward 中位延迟从 40.43/40.34 ms 降至 16.61/20.90 ms，即 2.43×/1.93×；32-token greedy 输出与 eager 一致。该数字不包含动态调度、采样、tokenizer 或网络开销。
- 已实现固定 capacity 的 ragged KV cache、continuous-batching 状态机、slot 回收/复用、host greedy sampling 与 admission microbatch。10 分钟真实 GPU soak 完成 17,264/17,264 个请求和 86,348 token，平均 143.79 token/s；368 次有限值检查通过，排空后请求注册表为 0，CUDA allocated 采样范围为 512 bytes。该结果是单进程 engine 稳定性证据，不是生产 SLA。
- 在 steady 50 req/s、16×3 burst 和 Poisson 20 req/s 双峰三类 open-loop 合成负载上，将 `8/4` admission 与即时 `1/0` 各重复 3 次。`8/4` 的 token throughput 分别提高 42.1%/46.2%/16.2%，TTFT P95 分别降低 47.5%/51.0%/33.8%；18 次 run、864 个请求全部完成。该结果不含 tokenizer/网络，不能写成 HTTP QPS。
- 对 `8/4` 策略完成 4/10/20/30/40/50 req/s 配对容量扫描，同一 repetition 的请求内容完全相同。30→50 offered req/s 时 token throughput 仅从 121.61 增至 129.10 token/s，而 TTFT P95 从 335.99 增至 746.94 ms，定位本配置约 25～30 req/s 的饱和拐点。该数字是有限 engine workload，不是生产稳定 QPS。
- 实现 112 个 Llama projection 的 packed/BF16 混合驻留规划器，将共享变换投影组成 64 个原子组，并按预测节省延迟/增量显存进行确定性选择。在 RTX 5070 的六点全模型测试中，512 MiB 预算相对全 packed 将 TTFT/TPOT 中位数降低 42.4%/45.8%，同时相对全 BF16 resident 少 36.3% CUDA allocated memory；方案已通过 ragged CUDA Graph 正确性验证。该结果只证明已测六个方案的 Pareto 权衡，不代表全局最优。

上述 PPL、下游和 runtime 结果仅支持各自记录的量化范围、beta 配置与评测协议。NQMC 与 packed runtime 已分别提供真实低比特存储和权重常驻执行证据，但不等于完成 NestQuant 全文复现；不能据此声称等码率优越性、A/KV 低比特常驻、超过 BF16 resident 的端到端加速或生产服务吞吐。

## 达到对应证据后才能写

| 简历表述方向 | 必须先拿到的证据 |
|---|---|
| Reproduced core W4/W4KV4/W4A4KV4 modules | Project-scope implementation is complete for nested coding, multi-beta, rotation ablations, LDLQ, A4/KV4 quality simulation, and QA-LDLQ adaptation. Batch-4 W4/W4KV4/W4A4KV4 PPL is 10.276601/10.732014/11.578171. Default QA epsilon constant 155 was a documented negative result. Runtime A/KV tensors are still reconstructed floating point and the complete paper rotation graph and low-bit kernels are absent. |
| QuaRot same-task compatibility comparison | Ported the official QuaRot fake-quant path to Llama-3.2-1B/RTX 5070 with a pure-PyTorch Hadamard fallback, GQA-aware K-cache reshape, tied embedding/lm_head handling, and Transformers 4.46 GPTQ device placement. Under the same model revision, WikiText-2 test-token hash and 288,627-token NLL aggregator, rotated BF16/GPTQ W4/W4KV4/W4A4/W4A4KV4 PPL is 9.762884/10.508346/12.269130/15.850334/21.095586. This is a documented compatibility port and quality simulation, not the official CUDA kernel or an inference-speed result. |
| “模型困惑度改善/接近论文” | 同一模型 revision、tokenizer、评测 split、文本处理和 PPL 聚合；保存 run manifest、token hash 和日志 |
| “模型压缩到约 4.27 bit/量化参数 / NQMC 为 BF16 逻辑载荷的 26.67%” | **已具备**：实际文件、byte 级核算、SHA-256、独立回载与完整 PPL；表述时保留“BF16 分母是逻辑载荷而非另一个实测文件”的限定 |
| “packed 权重显存下降 57%” | **已具备**：完整 1B、相同模型结构、warm allocation 与运行报告；必须同时披露当前比 BF16 resident 慢 |
| “CUDA Graph 相对 packed eager 加速 2.43×/1.93×” | **已具备**：固定 batch 1/16、相同 packed runtime、同步计时与输出一致性；不得外推为网络服务 QPS |
| “continuous batching 原型与稳定性” | **已具备**：ragged cache、slot 生命周期、10 分钟 GPU soak、17,264 请求完整回收；只能称 engine prototype |
| “服务负载下 admission 优化” | **已具备**：三类 open-loop workload、两策略、三重复、P50/P95/P99 与机器验收；不得写成 HTTP QPS 或生产 SLA |
| “定位服务容量拐点” | **已具备**：4～50 req/s 配对单因素扫描，吞吐平台与尾延迟/队列共同证据；必须限定模型、GPU、长度分布和有限请求数 |
| “混合驻留显存—延迟 Pareto” | **已具备**：六个全模型预算点、20 次重复、独立 validator、四类 SHA-256 和 Graph 正确性；必须说明选择器是贪心启发式、Pareto 仅限已测候选集合 |
| “服务部署能力” | 仍需可调用的网络入口、请求协议、错误处理、取消/超时和外部资源监控证据 |

任何数值都必须可追溯到配置、运行命令、日志和版本。结果不理想也保留，可作为取舍分析；不得用论文表格数字冒充本地结果。

## 后续可用的简历句式模板

> 独立实现并验证 NestQuant 的 [已完成模块]，在 [模型与数据协议] 上对比 [同范围基线]；将量化质量、实际存储开销与推理性能分开测量，获得 [本地测得指标与 trade-off]，完整保存配置、版本和复现实验脚本。

方括号内容必须在实验完成后填入，不根据目标或论文参考值预填。

## 下游评测完成（2026-10-24）

正式报告为 [`results/llama32_1b_downstream_full.md`](../results/llama32_1b_downstream_full.md)，机器可读小型汇总为 [`results/llama32_1b_downstream_full.json`](../results/llama32_1b_downstream_full.json)。原始逐样本 JSON 和日志保留在被 Git 忽略的 `results/raw/downstream_full_20261024/`。三档均使用相同模型 revision、任务定义和样本身份；当前没有 W4A4/W4KV4 单项下游结果，因此不能把联合退化归因给单一模块。

## QA-LDLQ stage result (2026-10-22)

Implemented the public QA weight adjustment, 20-train-chunk activation second-moment collection, QA NQMC export, and calibration-free reload. With the local covariance-scale Hessian and relative damping 0.01, the upstream default `epsilon_constant=155` degraded W4 PPL from 10.276601 to 11.053594 and W4A4KV4 from 11.578171 to 12.386133. This negative result requires an epsilon/Hessian-scale ablation before any claim of reproducing the paper QA gain. See [QA-LDLQ implementation](qa_ldlq.md) and [result](../results/llama32_1b_qa_ldlq_ablation.md).

## QA-LDLQ scale audit completion (2026-10-22)

The public Hessian divides the token Gram matrix by sequence count, while the local collector divides by activation-row count. At context length 2048 this created a 2048x normalization mismatch. A bounded representative-layer sweep showed that C=155 caused up to 73.7% weight adjustment and 9.07x proxy loss; corrected candidates removed the outlier. Full C=634880 results were W4 10.283140 and W4A4KV4 11.584234 versus ordinary LDLQ 10.276601 and 11.578171. The large regression was explained and removed, but no quality gain remained, so the QA route is closed without post-hoc wider tuning.

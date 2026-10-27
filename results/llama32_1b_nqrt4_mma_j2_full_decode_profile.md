# Llama-3.2-1B 完整模型单 token 解码性能剖析

## 实验范围

- 设备：NVIDIA GeForce RTX 5070，单卡；模型：`meta-llama/Llama-3.2-1B`；检查点：`results/raw/llama32_1b_paper_hadamard_w4_shared128_ldlq_nqrt4.nqmc`。
- 固定同步批处理，batch=1、16；输入长度 128；预热 3 次、采样 4 次单 token 解码；PyTorch profiler CPU/CUDA 活动。
- 脚本：`scripts/profile_full_nqrt_decode.py`。原始表：`results/raw/llama32_1b_nqrt4_mma_j2_profile_b1.json`、`results/raw/llama32_1b_nqrt4_mma_j2_profile_b16.json`、`results/raw/llama32_1b_bf16_profile_b16.json`；J2 时间线对应 `*_trace.json`。

## GPU 执行时间账单

下表为 profiler 中同名 GPU kernel 的累计时间除以 4 次解码步。`GPU kernel 合计`只计算底层 kernel，排除了重复包裹它们的 `aten::` 条目。它是 GPU 工作量，不是每步端到端耗时；profiler 会影响调度和耗时。

| 路径 | batch | GPU kernel 合计 | J2 partial kernel | J2 占比 | Split-K 归约 |
| --- | ---: | ---: | ---: | ---: | ---: |
| packed J2 | 1 | 22.37 ms/步 | 17.99 ms/步 | 80.43% | 0.116 ms/步 |
| packed J2 | 16 | 25.50 ms/步 | 20.31 ms/步 | 79.65% | 0.203 ms/步 |
| BF16 | 16 | 5.62 ms/步 | 不适用 | 不适用 | 不适用 |

J2 partial kernel 在每步调用 112 次，对应 16 层各 7 个量化线性投影。batch=1 的 Flash Attention kernel 累计约 0.13 ms/步；batch=16 约 0.30 ms/步。Hadamard 变换目前使用 PyTorch 基础算子执行；其大量 BF16 加法及拼接 kernel 也是可见开销，但 profiler 的同名加法或拼接可能包含模型其他操作，不能直接全部归因给 Hadamard。

## 时间线与端到端结果

4 步 J2 时间线分别记录约 12,632 次 kernel 启动。按 GPU kernel 区间合并，batch=1 的 kernel 忙时约 89.49 ms、首尾跨度约 295.06 ms；batch=16 分别约 102.00 ms、278.53 ms。这说明所采样执行中有大量 GPU kernel 之间的空隙；但 profiler 本身会放大 CPU 调度开销，30.3%/36.6% 的时间线忙碌比例不能当作无 profiler 的 GPU 利用率。

无 profiler 的较长 batch=16、decode=64 测量见 `results/llama32_1b_nqrt4_mma_j2_dynamic_batch.md`：J2 TPOT 44.674 ms，BF16 TPOT 13.063 ms，J2 聚合吞吐 358.149 tokens/s，BF16 1224.867 tokens/s。因此本次 profiler 的 kernel 时间不能与该 TPOT 直接相减推算精确的 CPU 开销。

## 决策

当前最有依据的下一项优化是降低 **J2 partial kernel** 每次投影的代价。它占所采样 J2 GPU kernel 时间约 80%，而 Split-K 归约不足 1%。优先在真实 q/gate/down 权重上分解该 kernel 的 E8 解码、无状态 dither、权重访存和 Tensor Core 指令成本，并做单变量性能/误差验证。另一条值得验证的路径是减少 Hadamard 的 PyTorch 小 kernel 数及 CPU 发射间隙；需要在无 profiler 的端到端测量中确认净收益。

这份剖析只能定位本机、此模型、此输入和当前实现的热点；它没有证明 E8 解码单独占据 80%，也没有证明 CUDA Graph 或连续批处理一定能带来多少加速。

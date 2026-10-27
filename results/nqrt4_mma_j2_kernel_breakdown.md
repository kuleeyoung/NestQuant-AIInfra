# NQRT4 J2 partial kernel 单变量耗时拆分

## 目的与方法

完整模型 profile 已表明 J2 partial kernel 占所记录 GPU kernel 时间约 80%。本实验用真实 Llama-3.2-1B 第 0 层 q、gate、down packed 权重，对该 kernel 构造三个诊断变体：

1. `full_decode`：与生产 J2 partial kernel 相同的哈希 dither、两次 E8 最近点搜索、缩放、BF16 打包和 Tensor Core MMA。
2. `packed_affine_proxy`：保留 packed code、beta ID、行范数读取、缩放、BF16 打包和 MMA，移除哈希及两次 E8 搜索。
3. `mma_floor`：保留激活读取、常量 B fragment、MMA 和 partial 写回。

每个变体预热 20 次；每组 100 次调用，重复 7 组，报告中位数。三个变体使用相同真实形状和生产 auto-split 配置。完整诊断变体经归约后与生产 kernel 的最大绝对差为 `0.95e-6` 至 `1.91e-6`，差异来自 PyTorch tree sum 与生产固定升序 FP32 归约的舍入顺序。

## M=1 单 token 解码

| 投影 | 形状 N×K | full | proxy | MMA floor | full-proxy 上界 | 占 full |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| q_proj | 2048×2048 | 82.05 us | 15.13 us | 9.52 us | 66.92 us | 81.6% |
| gate_proj | 8192×2048 | 306.38 us | 49.50 us | 25.53 us | 256.88 us | 83.9% |
| down_proj | 2048×8192 | 309.35 us | 57.73 us | 20.43 us | 251.62 us | 81.3% |

## M=16 同步批处理

| 投影 | 形状 N×K | full | proxy | MMA floor | full-proxy 上界 | 占 full |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| q_proj | 2048×2048 | 91.79 us | 30.77 us | 20.54 us | 61.02 us | 66.5% |
| gate_proj | 8192×2048 | 346.67 us | 109.54 us | 67.89 us | 237.13 us | 68.4% |
| down_proj | 2048×8192 | 352.18 us | 119.36 us | 69.94 us | 232.81 us | 66.1% |

## 两次 E8 搜索分别消融

新增 `skip_first_e8` 与 `skip_second_e8`：仍保留 hash、packed code 解码和另一轮 E8 搜索，每次只跳过一轮最近点搜索。下表是相对完整路径的耗时下降；它仍包含寄存器压力与指令调度变化，因此也是上界指标。

| M | 投影 | 跳过第一次 E8 | 跳过第二次 E8 |
| ---: | --- | ---: | ---: |
| 1 | q_proj | 25.0 us | 27.8 us |
| 1 | gate_proj | 96.1 us | 103.7 us |
| 1 | down_proj | 95.9 us | 102.3 us |
| 16 | q_proj | 22.0 us | 24.0 us |
| 16 | gate_proj | 86.9 us | 91.3 us |
| 16 | down_proj | 85.5 us | 89.9 us |

第二次 E8 搜索在六组测量中均略贵，但差距只有约 4% 至 11%。这说明只优化其中一轮不会消除主要瓶颈；两轮共享的 shuffle、奇偶校正、最大残差查找和距离归约才是主要代码优化目标。

## 结论与边界

当前 kernel 的第一优化对象是哈希 dither 加两次 E8 最近点搜索。`full-proxy` 是将这整段替换后的耗时差，包含指令依赖、寄存器压力和占用率变化，只能作为联合上界，不能解释为每个子步骤可独立相加的精确周期。`packed_affine_proxy` 改变了数学计算，不能用于质量评测或部署。

M=1 的联合上界比例高于 M=16，因为单 token 路径仍借用 m16 Tensor Core 指令，但只有第一行激活有效；批量增大后 MMA 本身占比上升。下一轮应在以下两条实现路线之间选择：

- 保持当前检查点格式：优化无状态 hash、E8 shuffle/reduction 和指令调度，不增加存储。
- 扩展检查点格式：缓存第一阶段 dither/E8 的紧凑辅助信息，以少量额外 bit/weight 换取解码速度；需要重新定义真实压缩率，并重新生成检查点。

原始数据：`results/raw/llama32_1b_nqrt4_mma_j2_breakdown_m1.json` 与 `results/raw/llama32_1b_nqrt4_mma_j2_breakdown_m16.json`。复现脚本：`scripts/benchmark_nqrt4_mma_j2_breakdown.py`。

## 已实施：保持归约顺序的 4-shuffle 优化

原 `nqrt4_sum8_torch_order_4lane` 让每个 lane 用 8 次 broadcast shuffle 收集全部坐标。新实现先以 XOR distance=2 计算 `(v0+v4)` 等配对，再以 distance=1 合并左右配对，因此只需 4 次 shuffle，并保持以下 FP32 加法树不变：

`((v0+v4)+(v2+v6))+((v1+v5)+(v3+v7))`

同轮 A/B 测量使用 `xor4` 与编译宏保留的 `broadcast8` 参考实现；两者的投影最大误差指标完全相同。

| M | 投影 | broadcast8 | xor4 | 延迟下降 |
| ---: | --- | ---: | ---: | ---: |
| 1 | q_proj | 86.42 us | 76.06 us | 12.0% |
| 1 | gate_proj | 311.90 us | 269.14 us | 13.7% |
| 1 | down_proj | 315.46 us | 273.15 us | 13.4% |
| 16 | q_proj | 98.87 us | 88.89 us | 10.1% |
| 16 | gate_proj | 352.92 us | 308.47 us | 12.6% |
| 16 | down_proj | 360.03 us | 315.62 us | 12.3% |

完整模型相同 decode=32 协议下，优化后 batch=1 TPOT 为 36.294 ms，batch=16 TPOT 为 35.645 ms、聚合吞吐为 448.87 tokens/s。此前动态 batch sweep 的对应结果分别为 45.674 ms 与 45.484 ms、351.77 tokens/s；即端到端 TPOT 分别下降约 20.5% 和 21.6%，batch=16 聚合吞吐提高约 27.6%。完整模型前后数据来自不同运行时段，可能受到 GPU 时钟影响；严格因果证据以同轮单层 A/B 表为准。

优化后原始数据：`results/raw/llama32_1b_nqrt4_mma_j2_checkpoint_shuffle4_m1.json`、`results/raw/llama32_1b_nqrt4_mma_j2_checkpoint_shuffle4_m16.json` 与 `results/raw/llama32_1b_nqrt4_mma_j2_shuffle4_full_b1_b16.json`。原始 broadcast 参考为相邻的 `checkpoint_broadcast8_m1.json` 和 `checkpoint_broadcast8_m16.json`。

## 已实施：2-bit 奇偶联合归约

整数格与半整数格各自产生一个奇偶位。原实现分别执行两轮 XOR shuffle，共 4 条 shuffle；新实现将两位打包进一个整数，用同一轮 XOR 归约，共 2 条 shuffle。按位 XOR 对两个 bit 相互独立，因此数学结果不变。

在保持 `xor4` 距离归约不变的同轮 A/B 中，`packed2` 相对 `separate4` 的单层延迟下降范围为 3.8% 至 17.2%。六组测试的最大误差逐项一致。完整模型测量受运行波动和 CPU 发射限制影响，没有观察到稳定的额外端到端收益，因此该优化只按单层 A/B 结果立论。

## 已实施：ballot 最大残差定位

原最大残差搜索在归约时同时 shuffle 浮点绝对值和坐标索引。新实现先只对绝对值做最大值归约，再用 warp ballot 生成“哪些 lane 达到最大值”的位掩码，用 `ffs` 选择最低 lane；每个 lane 已保留自己的局部最小坐标，因此仍与原来的“相等时取最小坐标”规则完全一致。

| M | 投影 | index shuffle | ballot | 延迟下降 |
| ---: | --- | ---: | ---: | ---: |
| 1 | q_proj | 69.93 us | 65.87 us | 5.8% |
| 1 | gate_proj | 245.69 us | 227.80 us | 7.3% |
| 1 | down_proj | 248.74 us | 230.14 us | 7.5% |
| 16 | q_proj | 84.07 us | 78.10 us | 7.1% |
| 16 | gate_proj | 295.48 us | 272.26 us | 7.9% |
| 16 | down_proj | 303.57 us | 279.25 us | 8.0% |

## 优化后的热点迁移

重新运行完整模型 profiler 后，J2 partial kernel 的 batch=1 GPU 时间由 17.99 ms/步降到 12.05 ms/步，下降 33.0%；batch=16 由 20.31 ms/步降到 14.58 ms/步，下降 28.2%。全部 GPU kernel 合计分别由 22.37 降到 16.06 ms/步、由 25.50 降到 19.40 ms/步。

| batch | 优化前 partial | 优化后 partial | 优化后 partial/GPU kernel |
| ---: | ---: | ---: | ---: |
| 1 | 17.99 ms | 12.05 ms | 75.05% |
| 16 | 20.31 ms | 14.58 ms | 75.14% |

E8 partial 仍是最大的单一 GPU 热点，但完整模型 TPOT 已不再随单层 kernel 的小幅优化稳定下降，说明 CPU 发射间隙和大量 Hadamard/PyTorch 小 kernel 已成为共同瓶颈。继续只压缩 E8 指令的端到端边际收益会低于单层收益；下一步应优先验证 CUDA Graph 或 Hadamard 融合能否减少发射开销。

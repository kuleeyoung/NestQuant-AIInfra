# Llama-3.2-1B NQRT4 服务级 engine benchmark

## 结论

在单张 NVIDIA GeForce RTX 5070 上，对 packed NQRT4、BF16 累加、capacity 8 的
host-driven continuous-batching engine 运行三种 open-loop 合成负载。每种负载分别比较
本机选定的 `min_prefill_batch=8, max_wait_steps=4` 与即时准入 `1/0`，各重复 3 次。
下表中的延迟是“三次实验各自 P95 的均值”，吞吐是三次实验吞吐的均值。

| workload | policy | request/s | token/s | TTFT P95 ms | TPOT P95 ms | E2E P95 ms | completion |
|---|---|---:|---:|---:|---:|---:|---:|
| steady 50 req/s, uniform | 8/4 | 23.82 | 117.94 | 895.87 | 119.84 | 1,029.81 | 100% |
| steady 50 req/s, uniform | 1/0 | 16.77 | 82.98 | 1,704.84 | 121.78 | 1,972.45 | 100% |
| 16-request bursts × 3, uniform | 8/4 | 26.42 | 132.05 | 665.28 | 92.45 | 775.09 | 100% |
| 16-request bursts × 3, uniform | 1/0 | 18.07 | 90.31 | 1,357.49 | 121.09 | 1,592.35 | 100% |
| Poisson 20 req/s, bimodal | 8/4 | 18.07 | 81.85 | 406.56 | 86.57 | 651.28 | 100% |
| Poisson 20 req/s, bimodal | 1/0 | 15.63 | 70.46 | 614.09 | 147.85 | 1,165.78 | 100% |

完整的 P50/P95/P99 延迟如下。每个数值仍是三次正式 run 对应分位数的算术平均，并非把三次
run 的请求样本混合后重新计算，因此保留了 repetition 作为独立实验单位。

| workload | policy | TTFT P50/P95/P99 ms | TPOT P50/P95/P99 ms | E2E P50/P95/P99 ms |
|---|---|---:|---:|---:|
| steady | 8/4 | 547.54/895.87/983.61 | 23.09/119.84/121.28 | 644.78/1,029.81/1,105.17 |
| steady | 1/0 | 832.52/1,704.84/1,796.71 | 107.62/121.78/122.76 | 1,143.31/1,972.45/2,006.10 |
| burst | 8/4 | 357.81/665.28/688.77 | 23.06/92.45/121.91 | 452.87/775.09/806.14 |
| burst | 1/0 | 643.28/1,357.49/1,475.21 | 84.35/121.09/121.72 | 990.81/1,592.35/1,640.31 |
| Poisson bimodal | 8/4 | 219.34/406.56/478.79 | 23.03/86.57/147.48 | 287.58/651.28/735.76 |
| Poisson bimodal | 1/0 | 327.68/614.09/742.06 | 82.83/147.85/148.38 | 546.87/1,165.78/1,357.32 |

相对 `1/0`，`8/4` 的改善为：

| workload | token throughput | TTFT P95 | TPOT P95 | E2E P95 |
|---|---:|---:|---:|---:|
| steady 50 req/s | +42.1% | -47.5% | -1.6% | -47.8% |
| burst 16 × 3 | +46.2% | -51.0% | -23.7% | -51.3% |
| Poisson bimodal | +16.2% | -33.8% | -41.4% | -44.1% |

三种 workload 的 18 次正式 run 均完成全部 48 个请求（总计 864 个），失败数、拒绝数和
最终 scheduler 注册表残留数均为 0，即观测失败率为 0%。正式 run 的最慢单个 engine step
为 149.51 ms；没有再出现诊断阶段的
秒级冷形状停顿。

三类负载的平均 CUDA peak allocated 分别约为 1,142/1,142/1,174 MiB；所有正式 run 的
peak reserved 均为 3,296 MiB。reserved 包含 PyTorch allocator 在全宽度预热阶段保留的
工作区，不能解释为每个请求真正同时使用了 3.2 GiB 张量。

## Workload 定义

- `steady_50rps_uniform`：每 20 ms 到达一个请求；prompt 4～32 token，输出 2～8 token；
- `burst_16x3_uniform`：每 500 ms 突发 16 个请求，共 48 个；长度范围同上；
- `poisson_20rps_bimodal`：平均间隔 50 ms；70% 为 prompt 4～16/output 2～4，
  30% 为 prompt 32～48/output 6～12。

50 req/s 是 offered load（施加到系统的到达速率），高于本实现的可持续 request throughput，
因此 TTFT 包含合理的排队时间。所有请求最终排空，所以 100% completion 不表示系统能够无限期
承受 50 req/s 而不增长队列。

## 计时与预热口径

TTFT 从计划到达时刻计到第一个输出 token 完成，包含引擎忙碌造成的提交滞后和排队时间；
TPOT 汇总同一请求相邻输出 token 的完成时间差；E2E 从计划到达到请求最后一个 token 完成。

第一次试跑发现某些首次出现的 query width 会产生约 2.2 秒的一次性初始化。正式结果因此明确
定义为 steady-state benchmark：在计时前预热配置覆盖的每个 prompt width（4～48）。原始
诊断保留在 `results/raw/service_benchmark_mixed_diagnostic.json`，没有删除异常 run 或从正式
重复中挑选最好结果。预热后所有正式 step 均不超过约 150 ms。

## 复现

```bash
python scripts/benchmark_service_workloads.py \
  results/raw/llama32_1b_paper_hadamard_w4_shared128_ldlq_nqrt4.nqmc \
  --config configs/service_benchmark.v1.json \
  --offline \
  --output results/raw/service_benchmark_v1.json
```

配置见 `configs/service_benchmark.v1.json`，逐 run、最慢 step、跨重复统计和环境版本见
`results/raw/service_benchmark_v1.json`。
独立验收结果写入 `results/raw/service_benchmark_v1_validation.json`；验收器检查 18-run
矩阵、完成率、失败/残留状态、百分位次序和全 prompt-width 预热覆盖。
可提交的机器摘要为 [`llama32_1b_nqrt4_service_benchmark.json`](llama32_1b_nqrt4_service_benchmark.json)，
其中记录 raw/config SHA-256，但不包含体积较大的逐 step 明细。

```bash
python scripts/verify_service_evidence.py \
  results/raw/service_benchmark_v1.json \
  --config configs/service_benchmark.v1.json \
  --summary results/llama32_1b_nqrt4_service_benchmark.json
```

## 证据边界

这是无 tokenizer、无网络 I/O 的单进程 engine benchmark，token ID 和到达过程为固定 seed
合成数据。它支持“`8/4` 在这三种本机负载下优于即时准入”的结论，不能写成 HTTP 服务 QPS、
生产 SLA、跨 GPU 通用最优策略或相对 BF16 resident 的加速。换 GPU、模型、capacity、长度分布
或到达过程后必须重新调参。

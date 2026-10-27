# Llama-3.2-1B NQRT4 Host-driven Continuous Batching

## 正确性

容量为 2 的真实 packed engine 同时处理 3 个请求，prompt 长度为 7/4/3，生成长度为 3/2/2。
第三个请求等待空槽并复用已经清理的物理 slot。三条请求生成的全部 token 与各自独立 greedy
full-forward 完全一致。

另外，CPU mock backend 上的固定 seed soak 覆盖 200 个请求、4 个 slot、每请求 1～6 个输出
token；队列完整排空，所有请求均进入 `FINISHED`，未发现跨请求 token 污染。

## Closed-loop 性能

设备：NVIDIA GeForce RTX 5070。容量 8，24 个请求在计时前全部入队；prompt 长度 4～16，
输出长度 2～6。先运行 8 请求 warmup，再统计正式 workload。

| 指标 | 结果 |
|---|---:|
| 生成 token | 101 |
| continuous elapsed | 1.0469 s |
| continuous throughput | 96.47 token/s |
| static-wave elapsed | 0.5149 s |
| static-wave throughput | 196.16 token/s |
| continuous / static-wave | 0.492× |
| decode-only step P50/P95 | 22.93 / 23.61 ms |
| admission step P50/P95 | 94.14 / 105.88 ms |
| logical TTFT P50/P95/P99 | 390.47 / 954.06 / 954.06 ms |
| logical E2E P50/P95/P99 | 688.84 / 1023.27 / 1046.89 ms |
| CUDA allocated/reserved | 1,130,983,936 / 1,287,651,328 bytes |

## 解释

这是一个有价值的负结果。当前策略在任何 slot 释放后立即接入等待请求，因此 15 个 engine step
中有 10 个 admission step；每次都执行一个较小的 packed prompt prefill，然后再执行已有请求的
decode。静态 batch-wave 只需要 3 个较大的 prefill，因而在这个“所有请求预先排队”的吞吐型
workload 中明显更快。

结果说明 per-request position、slot 生命周期和 CUDA Graph decode 已经正确，但调度策略尚未在
TTFT 与吞吐之间取得平衡。下一阶段应实现 admission microbatch：允许空 slot 短暂等待，把多个
新 prompt 合并为一次 prefill，同时设置最大等待时间保护 TTFT。

## C1 admission microbatch

已实现 `min_prefill_batch`、`max_wait_steps` 和 idle fast path。准入窗口从首次出现空 slot 时
开始计算；GPU 没有运行请求时直接启动，不等待凑批。

### Closed-loop 策略扫描

| min batch / max wait | 吞吐 token/s | TTFT P50/P95 ms | admission steps |
|---|---:|---:|---:|
| 1 / 0 | 103.29 | 351.88 / 893.10 | 10 |
| 2 / 1 | 134.16 | 258.14 / 668.76 | 8 |
| 4 / 1 | 150.97 | 274.88 / 582.10 | 6 |
| 4 / 2 | 156.45 | 197.14 / 560.15 | 5 |
| 6 / 2 | 155.50 | 199.15 / 563.76 | 5 |
| 8 / 4 | **215.76** | 218.70 / **361.50** | 3 |

### Open-loop 固定间隔到达

| 到达间隔 | 策略 | 吞吐 token/s | TTFT P95 ms |
|---:|---|---:|---:|
| 20 ms（50 req/s） | 1/0 | 32.15 | 2593.82 |
| 20 ms | 4/2 | 122.12 | 281.09 |
| 20 ms | 8/4 | **142.33** | **224.97** |
| 100 ms（10 req/s） | 1/0 | 26.82 | 2550.77 |
| 100 ms | 4/2 | 28.42 | 2309.06 |
| 100 ms | 8/4 | **40.80** | **187.05** |
| 250 ms（4 req/s） | 1/0 | 17.14 | 72.43 |
| 250 ms | 4/2 | **17.15** | 60.56 |
| 250 ms | 8/4 | 17.15 | **59.46** |

因此 RTX 5070、Llama-3.2-1B、capacity 8 的当前默认选择 `8/4`，并保留 idle fast path。
该默认值是硬件和 workload 相关配置，模型、GPU、容量或到达分布改变后必须重新扫描。

## 证据边界

TTFT 是所有请求在计时前入队的 closed-loop 排队延迟，不是固定到达率下的在线 TTFT。当前没有
tokenizer、网络、随机采样、SLA、open-loop arrival process 或长时间 GPU soak。不能把本结果
描述为生产服务 QPS，也不能声称 continuous batching 已经带来吞吐提升。

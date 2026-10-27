# Llama-3.2-1B NQRT4 服务容量扫描

## 结论

在 RTX 5070、capacity 8、`8/4` admission、packed NQRT4/BF16 accumulate 的固定环境中，
用 4/10/20/30/40/50 req/s 六档 fixed-rate open-loop workload 扫描饱和曲线。每档 48 个
请求、重复 3 次；同一 repetition 在六档速率中复用完全相同的 prompt/output 长度和 token ID，
只改变 arrival interval。

| offered rate | completed req/s | token/s | TTFT P50 ms | TTFT P95 ms | TTFT P99 ms | TPOT P95 ms | E2E P95 ms | queue depth P95 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | 4.04 | 20.02 | 79.77 | 93.57 | 94.84 | 21.73 | 234.61 | 1.00 |
| 10 | 9.68 | 47.92 | 166.53 | 264.14 | 277.67 | 108.89 | 453.54 | 4.00 |
| 20 | 17.93 | 88.76 | 197.20 | 292.64 | 302.52 | 109.69 | 468.16 | 8.00 |
| 30 | 24.57 | 121.61 | 232.14 | 335.99 | 371.97 | 110.04 | 543.22 | 11.67 |
| 40 | 25.97 | 128.63 | 368.54 | 578.13 | 623.42 | 108.99 | 734.47 | 19.20 |
| 50 | 26.07 | 129.10 | 453.89 | 746.94 | 821.57 | 109.39 | 869.77 | 24.48 |

吞吐在 30 req/s 后开始进入平台：offered rate 从 30 增加到 50 req/s（+66.7%），实测 token
throughput 只从 121.61 增加到 129.10 token/s（+6.2%），而 TTFT P95 从 335.99 增加到
746.94 ms（+122.3%），queue-depth P95 从 11.67 增加到 24.48。当前合成长度分布下的
饱和拐点约在 25～30 req/s，稳态吞吐上限约为 129 token/s。

TPOT P95 在 10～50 req/s 基本保持约 109～110 ms；高负载的主要退化来自 admission 前排队，
而不是已运行请求的单步 decode 时间持续恶化。4 req/s 下 TPOT 较低，是因为经常只有少量活跃
slot，不能与饱和区直接解释为同一批大小下的 kernel 加速。

## 公平性修正

第一版开发扫描为每档到达率使用不同 seed，导致平均输出长度不同，混淆了 token throughput。
正式配置启用 `couple_profile_seeds=true`：同一 repetition 的六档负载使用相同请求内容，构成
one-factor ablation。旧开发扫描被新的 paired JSON 覆盖，不作为正式证据。

## 复现与验收

```bash
python scripts/benchmark_service_workloads.py \
  results/raw/llama32_1b_paper_hadamard_w4_shared128_ldlq_nqrt4.nqmc \
  --config configs/service_capacity_sweep.v1.json \
  --offline \
  --output results/raw/service_capacity_sweep_v1.json

python scripts/validate_service_benchmark.py \
  results/raw/service_capacity_sweep_v1.json \
  --output results/raw/service_capacity_sweep_v1_validation.json
```

18 次 run、864 个请求全部完成，验收器检查 profile/policy/repetition 矩阵、完成率、终态回收、
百分位顺序和 prompt-width 预热覆盖。
可提交机器摘要见 [`llama32_1b_nqrt4_service_capacity.json`](llama32_1b_nqrt4_service_capacity.json)，
并保存 raw/config SHA-256 供回查。

```bash
python scripts/verify_service_evidence.py \
  results/raw/service_capacity_sweep_v1.json \
  --config configs/service_capacity_sweep.v1.json \
  --summary results/llama32_1b_nqrt4_service_capacity.json
```

## 证据边界

这是固定 seed、有限请求数、无 tokenizer/网络 I/O 的单进程 engine 容量扫描。“约 25～30
req/s 拐点”和“约 129 token/s 平台”只适用于本报告的模型、GPU、capacity、请求长度分布和
greedy sampling。它不是无限时长稳定到达率、HTTP QPS 或生产 SLA。

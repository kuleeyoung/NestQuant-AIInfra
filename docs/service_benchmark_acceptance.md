# 服务级 benchmark 验收矩阵

更新时间：2026-10-27

本页把路线 A 的要求映射到当前工作树中的直接证据。`passed` 只表示下表限定的单机、单进程、
合成 token-ID engine benchmark 已完成，不表示网络服务或生产 SLA 已完成。

| 要求 | 状态 | 直接证据 | 验收说明 |
|---|---|---|---|
| 固定速率负载 | passed | `configs/service_benchmark.v1.json`、`configs/service_capacity_sweep.v1.json` | 正式策略对照含 steady 50 req/s；容量扫描含 4/10/20/30/40/50 req/s |
| 突发负载 | passed | `configs/service_benchmark.v1.json`、`results/llama32_1b_nqrt4_service_benchmark.md` | 16 请求一组、500 ms 间隔，共 3 组 |
| 混合 prompt/output | passed | `poisson_20rps_bimodal` 配置及正式报告 | 70% 短请求、30% 长请求，prompt 与 output 长度同时变化 |
| TTFT P50/P95/P99 | passed | `results/raw/service_benchmark_v1.json`、正式服务报告 | 每个 run 保留三档分位数，报告给出三次 run 对应分位数的均值 |
| TPOT P50/P95/P99 | passed | 同上 | 基于同一请求相邻输出 token 完成时间差 |
| E2E P50/P95/P99 | passed | 同上 | 从计划到达到最后一个输出 token 完成 |
| 请求/Token 吞吐 | passed | 正式服务报告及容量扫描报告 | 同时记录 completed req/s 和 generated token/s |
| 完成率/失败率 | passed | 两份 `*_validation.json`、正式机器摘要 | 两组正式矩阵各 864/864 完成，failed/rejected/retained 均为 0 |
| GPU 显存 | passed | raw JSON 的 peak allocated/reserved、正式服务报告 | 正式服务 run 报告峰值；reserved 明确包含预热后 allocator 保留区 |
| 稳态预热口径 | passed | `warmup_prompt_lengths` 与验收器 | 计时前覆盖配置中的 4～48 prompt width；冷形状诊断单独保留 |
| 重复实验 | passed | raw JSON 与验收器 | 每个 profile×policy 重复 3 次，run ID 和 repetition 必须唯一完整 |
| 容量拐点 | passed | `results/llama32_1b_nqrt4_service_capacity.md` | 配对单因素扫描定位约 25～30 req/s、约 129 token/s 平台 |
| 长时间稳定性 | passed（限定范围） | `results/llama32_1b_nqrt4_continuous_batch_soak.md` | 600 秒、17,264 请求、86,348 token、终态注册表为 0 |
| 证据可追溯 | passed | 两份正式摘要及 `verify_service_evidence.py` | raw/config SHA-256 和摘要逐字段重建均通过 |
| 可发布安装包 | passed | `scripts/smoke_publishable_tree.sh` | source wheel、sdist→wheel、独立安装、console、CUDA/C++ package data 通过 |

## 最终机器验收

```text
service benchmark: 18 runs, 864/864 completed, failed=0, rejected=0, retained=0
capacity sweep:    18 runs, 864/864 completed, failed=0, rejected=0, retained=0
pytest:            246 passed, 25 subtests passed
repository audit:  366 publishable files, 0 findings
```

## 明确未验收的范围

- tokenizer 与真实文本请求；
- HTTP/JSON 或 gRPC 网络入口；
- 多进程或多 GPU 调度；
- 请求取消、超时、限流和租户隔离；
- 无限时长稳定到达率和生产 SLA；
- 3B/8B 跨模型规模复现；
- packed A/KV 和直接 INT4 Tensor Core GEMM。

这些未完成项不能从本轮 engine benchmark 外推。下一阶段是否进入混合权重缓存、网络服务或
GitHub/面试交付，见 [`next_route_decisions.md`](next_route_decisions.md)。

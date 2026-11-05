# W4A4KV4 在线推理服务

## 已实现边界

`OnlineInferenceService` 使用一个专用工作线程独占
`ContinuousBatchInferenceEngine`、模型与 CUDA KV Cache。HTTP 连接线程只能发送提交或
取消命令，不能直接修改调度器或页面池。当前支持：

- 有界在途请求数与 HTTP 429 背压；
- waiting/running 请求取消，以及取消后的即时 slot/KV 清理；
- 每请求 deadline 超时与 HTTP 504；
- 同步 JSON 与逐行 NDJSON token 流；
- 流式客户端断开后的取消；
- `/health` 和 `/metrics`；
- Ctrl+C/SIGTERM 正常停机，未完成请求进入 cancelled 终态。

这是单进程、单 GPU 的演示服务，不是生产级多副本网关。它没有鉴权、TLS、跨进程
负载均衡、持久化请求或 Prometheus 文本协议，也未声明生产 SLA。

## 启动真实 1B packed 服务

在 WSL Ubuntu 的项目根目录和已经激活的虚拟环境中执行：

```bash
python scripts/serve_w4a4kv4.py \
  results/raw/llama32_1b_paper_hadamard_w4_shared128_ldlq_nqrt4.nqmc \
  --offline \
  --host 127.0.0.1 \
  --port 8000 \
  --capacity 2 \
  --max-inflight 32 \
  --max-cache-len 2048
```

在线服务默认对已经通过 2,048-context、1% teacher-forced PPL 非劣性门的精确
`M=1,K=8192,N=2048` down-projection 启用 split-8。该策略不保证与旧 kernel 生成完全相同的
token；需要复现旧输出或快速回滚时，在启动命令中增加 `--disable-m1-down-splitk`。底层通用
Linear API 和 benchmark 不继承这一服务默认值，其他形状也不会自动启用。

默认 `127.0.0.1` 只允许本机访问。除非已经配置防火墙、鉴权和反向代理，否则不要改成
`0.0.0.0` 暴露到局域网或公网。

## 请求示例

同步生成：

```bash
curl -X POST http://127.0.0.1:8000/v1/generate \
  -H 'Content-Type: application/json' \
  -d '{"request_id":"demo-1","prompt":"The capital of France is","max_new_tokens":8}'
```

流式生成：

```bash
curl -N -X POST http://127.0.0.1:8000/v1/generate/stream \
  -H 'Content-Type: application/json' \
  -d '{"request_id":"demo-stream","prompt":"Once upon a time","max_new_tokens":16}'
```

取消、健康和指标：

```bash
curl -X DELETE http://127.0.0.1:8000/v1/requests/demo-stream
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/metrics
```

## 本机真实 smoke

2026-11-02 使用 Llama-3.2-1B NQRT4 checkpoint、packed W4A4 CUDA MMA、NQKV4、
capacity 2 和 page size 16 启动服务。`/health` 返回 `ok`；一个 2-token 请求返回
`finished`，指标为 `submitted=1`、`completed=1`、`inflight=0`；Ctrl+C 后进程以 0
退出。该结果只证明端到端可运行，不是吞吐、质量或网络 SLA 评测。

## 下一性能验收

下一项工作是固定本机协议的 HTTP open-loop benchmark：分别测非流式和 NDJSON 流式
接口的 TTFT/TPOT/E2E、429 比例、取消释放时间与 10 分钟资源稳定性，并与现有无网络
token-ID engine 数据分开报告。

首轮 12 请求 open-loop smoke 已完成：固定 100 ms 到达间隔、每请求生成 3 token，
12/12 完成、0 失败、0 过载；客户端观测 token throughput 20.24 token/s，TTFT
P50/P95 为 429.58/550.31 ms，TPOT P50/P95 为 58.34/66.01 ms，E2E P50/P95 为
546.86/665.79 ms。详见
[HTTP open-loop smoke](../results/http_service_smoke_v1.md)。该规模仍不足以形成容量或
生产 SLA 结论，下一步需扩大请求数并执行 10 分钟网络 soak。

随后完成 2/4/8/16/32 offered request/s 五档容量扫描，每档 24 个请求、每请求 3
token。2/4/8 req/s 均为 24/24 完成、0 过载；16 req/s 为 19/24 完成、5 个 429；
32 req/s 为 13/24 完成、11 个 429。最高无损档为 8 req/s；峰值完成吞吐为
8.46 request/s、25.39 token/s。TTFT P95 从 8 req/s 的 301.53 ms 增至 16 req/s
的 771.16 ms，表明容量拐点位于 8～16 offered request/s。详见
[HTTP capacity sweep](../results/http_capacity_sweep_v1.md)。

10 分钟网络 soak 也已完成：8 offered req/s、4,800 个计划请求中 4,792 个完成，8 个
显式 429，0 失败/超时；完成吞吐 7.984 request/s、23.953 token/s。60 秒后的 CPU
RSS 范围仅 323,584 bytes，CUDA allocated/reserved 范围均为 0；排空后 inflight、
registered、waiting、running 和 cold-page allocated 均为 0。详见
[HTTP 600-second soak](../results/http_service_soak_600s_v1.md)。该短 prompt 未填满
16-token NQKV 页面，因此证明的是服务与 hot-tail 生命周期稳定性，不是跨页 page-pool soak。

该边界随后由独立的 120 秒跨页 soak 补齐：固定 prompt 经同 revision tokenizer 实测为
29 token，每个请求必定跨过 16-token 页边界。240/240 请求完成、0 失败/超时/429；
worker 观测到 `nqkv_allocated_pages_peak=32`，与 16 层 × 2 个并发请求各持有一页完全
一致，排空后 allocated=0、free=192。CUDA allocated/reserved 在 60 秒后的范围仍为
0，CPU RSS 范围为 69,632 bytes。详见
[cross-page HTTP soak](../results/http_cross_page_soak_120s_v1.md)。

同一 29-token prompt 的 1/2/4/8 req/s 跨页容量扫描显示：1/2 req/s 均 24/24
完成且 0 过载；4 req/s 为 21/24 完成、12.5% 429；8 req/s 为 15/24 完成、
37.5% 429。该进程内的最高无损档为 2 req/s，首个过载档为 4 req/s，峰值完成
吞吐为 2.63 req/s、7.89 token/s。详见
[cross-page capacity sweep](../results/http_cross_page_capacity_sweep_v1.md)。该次独立
进程的 TPOT P95 为 174～203 ms，明显慢于上一跨页 soak 的 60.90 ms；因此只提升
同进程容量拐点，不把绝对延迟当成跨进程稳定基线。

为检查上述跨进程波动，随后固定相同 29-token prompt、2 req/s、24 请求和两次显式
warmup，重新启动 3 个全新服务进程。三轮均为 24/24 完成、0 失败/过载；TPOT P95
为 68.60/64.24/64.55 ms，CV 为 3.02%，TTFT P95 的 CV 为 4.17%。测量阶段三轮均
处于 P1，SM 时钟均值为 2885.7～2888.0 MHz，功耗均值为 87.9～88.7 W。详见
[fresh-process reproducibility](../results/http_fresh_process_repro_v1.md)。因此受控预热
协议下的约 65 ms TPOT 可以跨进程复现；旧进程的 174～203 ms 异常没有复现，可能
因素包括未显式预热或当时未记录的竞争负载，但现有证据不能指定唯一根因。

在此基础上，容量协议进一步固定为两次显式 warmup、12 个 client workers、24 个
29-token 跨页请求。2/4/6 req/s 均 24/24 完成且无 429；8/10/12 req/s 分别出现
1/3/5 个 429。最高无损档为 6 req/s，首个显式背压档为 8 req/s，峰值完成吞吐为
6.36 req/s、19.09 token/s。此前使用 8 个 client workers 测 10/12 req/s 时，客户端
线程池自身排队，使到达服务端的并发不超过 `max_inflight=8`，从而掩盖了服务端背压；
因此该组结果不进入正式容量表。详见
[controlled cross-page capacity sweep](../results/http_cross_page_capacity_repro_v2.md)。

Nsight Systems 对 2→4 req/s 的受控 A/B 显示：完成吞吐提高 1.92x，TPOT P95 仅提高
1.02x，但 TTFT P95 提高 2.51x；GPU 平均利用率从 47.1% 提高到 70.3%。每请求
CUDA Graph launch 和 CUDA kernel launch 分别减少到 0.83x 与 0.84x，说明连续批处理
把更多请求摊入更少的模型执行。当前 WSL/Nsight 组合未产生 GPU kernel timeline，
所以证据支持“排队/准入等待增加、批处理摊销改善”，但还不能定位某个 kernel 的
duration 或 stall。详见 [Nsight offered-load A/B](../results/http_nsys_cross_page_2v4_ab_v1.md)。

随后在 worker 所有权边界内增加了单调时钟阶段埋点。2 req/s 时 command queue 与
scheduler waiting 合计仅占服务端 TTFT 均值 0.2%；4/6 req/s 时占比升至 45.1%/43.7%。
4 req/s 的 command queue、scheduler waiting 和 prefill P95 分别为
146.69/126.97/170.50 ms，而客户端与服务端 TTFT P95 仅差 2.32 ms，因此主要增长来自
worker忙碌与准入排队，不是 HTTP/socket。详见
[server-side TTFT phase decomposition](../results/http_phase_timing_2_4_6rps_v1.md)。

基于该归因，比较了默认策略 `min_prefill_batch=2,max_wait_steps=2` 与立即准入 `1,0`，
每种策略运行 3 个全新进程。立即准入虽把 scheduler waiting 均值中位数降低 56.1%，
却使 command queue 增加 63.2%、TPOT P95 增加 123.2%、E2E P95 增加 33.5%，吞吐
下降 3.1%；三轮 TPOT 和吞吐均退化。`min_prefill_batch=2,max_wait_steps=1` 的单轮探针
也显著退化，因此保留默认 `2,2`。详见
[admission-policy A/B](../results/http_admission_policy_4rps_ab_v1.md)。

固定执行容量从 2 扩到 4 的单轮方向探针也被否决：4 req/s 下吞吐从 3.90 降至
3.08 req/s，TTFT/TPOT P95 从 439.20/65.53 ms 增至 1640.80/331.84 ms，prefill
均值从 165.51 增至 395.13 ms，显存峰值约增加 199 MiB。当前 runner 要求输入 batch
严格等于 cache capacity，`active_mask` 只保护 KV 语义，不能跳过非活跃行的 112 个
packed Linear 计算，因此 capacity 4 固定支付 M4 成本。详见
[fixed-capacity 2 versus 4 probe](../results/http_capacity2_vs4_probe_v1.md)。

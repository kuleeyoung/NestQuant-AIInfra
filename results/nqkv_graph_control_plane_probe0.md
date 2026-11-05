# NQKV 双 CUDA Graph 控制面 Probe 0

日期：2026-11-02

## 结论

在真实 Llama-3.2-1B、capacity 2、page size 16、packed W4A4 CUDA backend 和
NQKV4 双 Graph 路径上完成 30 轮同步分段测量。普通 decode 的控制面中位耗时为
0.623 ms，占分段总时间 0.85%；页边界控制面中位耗时为 3.498 ms，占 4.46%，其中
页面预留为 2.717 ms。边界每 16 token 发生一次，摊销后全部已测控制面成本的理论上限
约为 0.8 ms/token，约占当前约 74 ms decode 的 1.1%。

因此触发预先约定的停止条件：**当前不实现深度 2 跨层 reserve window，也不在 CUDA
Graph 内开发通用 device-side page allocator。** 保留现有双 Graph，并优先优化占 99%
左右的模型/Graph 数据面。若未来 kernel 将 decode 降到 20 ms 以下，或更大 capacity
使控制面比例超过 5%，再重新评估预留窗口。

## 协议

- checkpoint：`llama32_1b_paper_hadamard_w4_shared128_ldlq_nqrt4.nqmc`
- 模型：`meta-llama/Llama-3.2-1B`
- compute backend：`packed_w4a4_cuda_mma`
- KV backend：NQKV v2 packed KV4
- capacity：2
- page size：16
- max cache length：32
- repetitions：30
- normal probe：先 prefill 14 token，再 decode 第 15 token
- boundary probe：随后 decode 第 16 token，触发 hot-page pack
- 输出：`results/raw/nqkv_graph_control_plane_probe0.json`

每个阶段前后调用 `torch.cuda.synchronize()`，使异步 GPU 工作归属于实际发起它的阶段。
因此这些数字用于瓶颈归因，不能替代默认异步服务路径的端到端 TPOT。

## 结果

| 路径 | 阶段 | 中位数 | P95 |
|---|---|---:|---:|
| normal | validation | 0.365 ms | 0.879 ms |
| normal | boundary detection | 0.166 ms | 0.534 ms |
| normal | metadata prepare | 0.024 ms | 0.057 ms |
| normal | Graph replay | 72.755 ms | 74.059 ms |
| normal | control plane total | **0.623 ms** | 1.316 ms |
| normal | profiled total | **73.458 ms** | 74.872 ms |
| boundary | validation | 0.305 ms | 0.588 ms |
| boundary | boundary detection | 0.293 ms | 0.573 ms |
| boundary | page reservation | **2.717 ms** | 4.396 ms |
| boundary | metadata prepare | 0.083 ms | 0.226 ms |
| boundary | Graph replay | 74.736 ms | 76.091 ms |
| boundary | page commit | 0.061 ms | 0.154 ms |
| boundary | control plane total | **3.498 ms** | 5.362 ms |
| boundary | profiled total | **78.479 ms** | 80.497 ms |

第一轮 normal boundary-detection 出现 35.06 ms 的一次性异常值，但 30 轮中位数为
0.166 ms、P95 为 0.534 ms。结论使用中位数和 P95，不使用最小值或单次最好结果。

## 摊销估算

page size 16 时，每 16 个 decode token 才发生一次边界：

```text
normal control median                         = 0.623 ms/token
boundary extra control                       = 3.498 - 0.623 = 2.875 ms
boundary extra amortized over 16 tokens       = 2.875 / 16 = 0.180 ms/token
visible control-plane upper bound             = 0.623 + 0.180 = 0.803 ms/token
share of approximately 74 ms decode           = about 1.1%
```

这是“把所有已测控制面成本完全消除”的乐观上界。实际实现仍需保留输入检查、active-mask
更新、请求取消和页所有权维护，因此真实收益会更低。

## 工程决定

1. 不进入完整 GPU page-ring/reserve-window 开发。
2. 不将 vLLM/SGLang/TensorRT-LLM allocator 整体引入项目。
3. 保留当前 host reservation + boundary Graph，因为其正确性和性能已经闭环。
4. 保留调研文档中的 reserve-window 设计，作为未来 kernel 更快或 batch 更大时的备选。
5. 下一轮性能优化应针对 Graph replay 内的 packed W4A4 projection、NQKV attention 或
   launch/同步结构，而不是当前页池控制面。

## 证据边界

- 结果来自单张本机 GPU 和 capacity 2，不外推到 H100、多 GPU 或高并发服务。
- 分段间显式同步会改变默认异步执行行为；报告只用于占比与优化决策。
- 本结果否决的是“当前硬件和当前 kernel 下立即开发复杂 reserve window”，不是证明
  GPU 预留页技术在所有规模下都无收益。

# NQKV 双 CUDA Graph 数据面 Kernel Profile

日期：2026-11-02

## 结论

真实 Llama-3.2-1B、capacity 2、packed W4A4 CUDA + NQKV4 双 Graph 的 GPU 时间
主要由 `nqrt4_w4a4_mma_small_m_kernel` 占用。普通与边界 Graph 共调用该 kernel
224 次，累计 133.505 ms；两张 Graph 的 GPU 总时间约 150.385 ms，W4A4 projection
约占 88.8%。NQKV cold-page Attention 累计 1.802 ms，hot-tail Attention 0.143 ms，
边界 E8 encode 1.147 ms，均不是当前主瓶颈。

按每层固定的 Q/K/V/O/gate/up/down 调用顺序验证并拆分后，`down_proj` 两张 Graph
累计 52.661 ms，占 W4A4 kernel 时间约 39.4%，是单一最大的 projection family。
下一步应对 M2、K=8192、N=2048 的 down projection 做 split-K 隔离 probe，而不是继续
优化页池、KV Attention 或 page encode。

## 工具与协议

- 模型：`meta-llama/Llama-3.2-1B`
- checkpoint：NQRT4 shared128 LDLQ
- compute backend：`packed_w4a4_cuda_mma`
- KV backend：NQKV v2 packed KV4
- capacity：2
- page size：16
- warmup：3 次完整 normal + boundary cycle
- 捕获：1 次 normal Graph + 1 次 boundary Graph
- PyTorch/Kineto：展开 Graph 内 CUDA kernel，导出 Chrome trace
- Nsight Systems：CUDA API 捕获成功，但当前 WSL 环境未导出 Graph kernel 表；不使用
  其空结果做结论

机器可读结果：

- `results/raw/nqkv_graph_kernel_probe0.json`
- `results/raw/nqkv_graph_kernel_probe0_torch_summary.json`
- `results/raw/nqkv_graph_w4a4_shape_breakdown.json`

Chrome trace 保存在 `tmp/`，不属于发布产物。

## 总体 Kernel 构成

| Kernel/范围 | 调用次数 | 累计 GPU 时间 | 相对两 Graph GPU 时间 |
|---|---:|---:|---:|
| 两张 Graphed callable | 2 | 150.385 ms | 100% |
| packed W4A4 small-M | 224 | 133.505 ms | **88.8%** |
| 通用 elementwise add family | 2,944 | 3.605 ms | 2.4% |
| NQKV cold-page attention | 32 | 1.802 ms | 1.2% |
| lm_head CUTLASS BF16 GEMM | 2 | 1.744 ms | 1.2% |
| NQKV boundary E8 encode | 192 | 1.147 ms | 0.8% |
| NQKV hot-tail attention | 32 | 0.143 ms | 0.1% |
| NQKV page reduce | 32 | 0.032 ms | <0.1% |

上述百分比按 `Graphed` GPU 总时间计算。record-function/NVTX range 自身会包含子 kernel，
不能与 kernel 时间再次相加。

## W4A4 Projection 拆分

Graph ID 2 和 5 分别对应本次捕获的两张 full-model Graph。每张 Graph 均有 16 层、
每层 7 个 projection，共 112 次 W4A4 调用。汇总器验证每层 grid 顺序严格重复：

```text
q       k      v      o       gate    up      down
128,    32,    32,    128,    512,    512,    128
```

当前 kernel 每个 grid block 负责 16 个输出列，因此：

- grid 32 → N=512：K/V；
- grid 128 → N=2048：Q/O/down；
- grid 512 → N=8192：gate/up。

| Projection | Graph 2 总计 | Graph 5 总计 | 两图总计 | 典型单层中位 |
|---|---:|---:|---:|---:|
| q_proj | 5.771 ms | 6.230 ms | 12.001 ms | 0.351 ms |
| k_proj | 6.804 ms | 5.850 ms | 12.654 ms | 0.350 ms |
| v_proj | 6.419 ms | 6.486 ms | 12.905 ms | 0.351 ms |
| o_proj | 5.861 ms | 6.104 ms | 11.966 ms | 0.350 ms |
| gate_proj | 7.322 ms | 7.422 ms | 14.744 ms | 0.423 ms |
| up_proj | 8.638 ms | 7.937 ms | 16.576 ms | 0.423 ms |
| down_proj | **25.142 ms** | **27.519 ms** | **52.661 ms** | **1.53～1.69 ms** |

两张 Graph 的 W4A4 family 占比：

```text
down_proj                 52.661 / 133.505 = 39.4%
gate_proj + up_proj       31.319 / 133.505 = 23.5%
Q/K/V/O                   49.525 / 133.505 = 37.1%
```

## 原因解释

当前 small-M kernel 每个 CTA 只有一个 warp，每个 warp 负责 16 个输出通道，并串行遍历
全部 `K/16` tile：

- Q/K/V/O 与 gate/up 的主要输入 K 为 2048，需要 128 个 K tile；
- down projection 输入 K 为 8192，需要 512 个 K tile。

down kernel 单层时间约为普通 Attention projection 的 4～5 倍，与 K 维扩大 4 倍一致。
这说明问题主要是 K 维串行工作量，而不是 NQKV page 或页边界调度。

## 下一 Probe：M2 down split-K

建议先实现隔离版 `run_small_m_splitk_probe`，仅覆盖：

```text
M = 2
N = 2048
K = 8192
split-K = 2 / 4 / 8
```

每个输出 tile 使用多个 warp 分别处理一段 K，并将 FP32 partial sums 写入固定 workspace，
再由第二个 kernel 归约。第一版不接生产 dispatch，不使用 atomic accumulation，以便保持
确定的归约顺序和可比较误差。

验收标准：

1. 与当前 small-M kernel 的最大绝对误差和余弦相似度明确记录；
2. 100 次 warm timing 的中位数至少快 1.25x；
3. workspace 在 Graph capture 前固定分配；
4. split/reduce 两个 kernel 均可被 CUDA Graph 捕获；
5. 若最快配置不足 1.15x，停止 split-K，不接完整模型；
6. 若达到 1.25x，再只替换 down_proj 做完整 1B A/B。

## 证据边界

- 本次只捕获一对 warm Graph，kernel 占比用于定位，不作为稳定服务吞吐结果；
- projection family 通过固定 grid 顺序验证，但未标识单独层名；
- PyTorch profiler 会增加 CPU 侧开销，GPU kernel duration 仍可用于同一次 trace 内归因；
- 当前 CUDA W4A4 路径已有累加顺序导致的生成分叉，split-K 还会改变归约顺序，因此必须
  继续保持显式 opt-in，并单独报告质量边界。

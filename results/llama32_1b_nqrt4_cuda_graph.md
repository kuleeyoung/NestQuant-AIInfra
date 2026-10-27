# Llama-3.2-1B NQRT4 CUDA Graph 可行性与正确性验证

## 目标

完整模型 profile 表明，J2 partial kernel 优化后，端到端解码仍存在大量 CPU kernel launch 间隙。本实验验证固定 batch 的单 token 解码能否被 CUDA Graph 捕获，并验证 token 与 KV Cache position 更新后仍与 eager 执行一致。

环境为单张 NVIDIA GeForce RTX 5070、Llama-3.2-1B、输入长度 128。检查点为 `llama32_1b_paper_hadamard_w4_shared128_ldlq_nqrt4.nqmc`。每组预热后执行 30 次 replay，重复 5 组并报告 wall-clock 中位数。

## 调试中发现的两个边界问题

### Transformers StaticCache 状态

Transformers 5.20.2 的 `StaticLayer.update` 不使用外部 `cache_position`，而是内部递增 `cumulative_length`。重复固定位置会越界，也不适合直接捕获。本探针保留官方 StaticCache 的预填充过程，在 decode capture 阶段将 update 替换为固定地址的 `index_copy_`，位置由一个持久 CUDA tensor 提供。

### 自定义 J2 kernel 使用了错误的 CUDA stream

生产 J2 扩展原先使用不带 stream 参数的 `<<<grid, block>>>`，因此 kernel 被提交到 CUDA 默认流。PyTorch Graph 在专用 capture stream 上录制，导致 J2 kernel 未被捕获：表面 replay 约 4 ms，但 logit 最大误差达到 15.875，batch=16 还出现 NaN。该结果无效，已明确弃用。

所有 J2 partial 和 Split-K reduction launch 现已改为 `at::cuda::getCurrentCUDAStream()`。修复后无 Cache J2 Graph、固定位置 Cache Graph及动态 position Graph 均与 eager 完全一致。构建测试另增加非默认 stream 验证，防止回归。

## 有效结果

使用 `torch.cuda.make_graphed_callables` 管理静态输入、Graph 私有内存池和输出缓冲区。J2 结果如下：

| batch | eager | CUDA Graph | 加速比 | Graph 聚合吞吐 |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 38.810 ms/步 | 16.978 ms/步 | 2.286× | 58.90 tokens/s |
| 16 | 40.124 ms/步 | 25.273 ms/步 | 1.588× | 633.10 tokens/s |

正确性检查：

- capture 与 eager 最大 logit 绝对误差：0；
- 第一次 replay 与 capture 最大误差：0；
- 第二次 replay 与第一次 replay 最大误差：0；
- Graph 后再次执行 eager 与 Graph 前 eager 最大误差：0；
- 将 token 和 position 作为输入缓冲区，连续验证 position 130、131、132、133，逐步 Graph 对 eager 最大误差均为 0。

BF16 对照也通过固定位置 Graph 精确验证：batch=1 从 10.838 ms 降到 4.624 ms（2.344×），batch=16 从 10.530 ms 降到 6.697 ms（1.572×）。这些 BF16 数字用于证明模型与 StaticCache 捕获路径有效，不是 J2 与 BF16 的最终公平性能比较。

## 结论与边界

CUDA Graph 明确消除了当前 J2 路径的大量 CPU 发射开销，A1 可行性成立。修复 current-stream 后的 16.98/25.27 ms 才是有效性能结果；修复前约 4 ms 的数字不得使用。

当前脚本已经验证动态 token 内容和连续 cache position，但仍是独立基准入口：argmax 在 Graph 外执行，最大 cache 长度固定为测试范围，尚未接入 Transformers `generate()` 或多请求调度器。下一阶段需要封装可复用的 graph runner，为各 batch size 管理静态输入、KV Cache、capture 和 replay 生命周期，再接入完整生成循环。

复现脚本：`scripts/benchmark_full_nqrt_cuda_graph.py`。有效原始数据：`results/raw/llama32_1b_nqrt4_cuda_graph_dynamic_b1_b16.json`。

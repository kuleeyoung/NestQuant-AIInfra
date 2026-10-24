# Llama-3.2-1B Lazy NQMC 真实 Forward 分阶段剖析

## 目的

上一阶段测得 256 MiB LRU 的真实 lazy forward 平均约 23.96 秒，而 warm BF16 基线为 52.61 ms。为了避免在错误方向上继续优化，本实验把 lazy projection 拆成五个可测阶段：

```text
record_read
→ record_parse
→ unpack_dequant
→ input_transform / Hadamard
→ GEMM + output cast
```

只有显式 profiling 模式会在 GPU 阶段前后执行 `torch.cuda.synchronize`；默认 runtime 不增加这些同步点。正式计时前先执行一轮包含全部 112 个 projection 的 warm-up，然后清空事件、LRU 和统计，排除首次 kernel 初始化。

## 总体结果

真实 `LlamaForCausalLM.forward` 用时 22.071 秒，各阶段为：

| 阶段 | 次数 | 总时间 | Forward 占比 |
|---|---:|---:|---:|
| NQMC record read | 112 | 1.554 s | 7.04% |
| NQRT record parse | 112 | 0.121 s | 0.55% |
| unpack + dequantize BF16 | 112 | 20.224 s | 91.63% |
| 输入转换 / online Hadamard | 112 | 0.061 s | 0.28% |
| BF16 GEMM + output cast | 112 | 0.016 s | 0.074% |
| 其他模型开销 | — | 0.095 s | 0.43% |

五个显式阶段合计 21.977 秒，与 forward wall time 的差值为 94.65 ms。差值覆盖 attention、RoPE、RMSNorm、SwiGLU、residual、embedding、lm_head 和 Python 模型调度等未在 lazy Linear 内部计时的操作。

## 关键判断

当前瓶颈不是 GEMM，也不是 Hadamard，而是 packed record 到 BF16 weight 的 unpack/dequant：

```text
unpack_dequant / forward = 91.63%
```

文件 read 与 record parse 合计只占 7.59%。因此，即使设计出理想预取器，能够把所有文件读取和 CPU parse 完全隐藏，理论上也只能触及本次测量约 7.6% 的 wall time，无法解决相对 BF16 基线约 456 倍的差距。

这改变了后续优先级：

```text
低优先级：只做异步文件 I/O、线程预取、调整 LRU
高优先级：减少 Torch operator 数量、compiled unpack/dequant、fused decode + GEMM
```

## 不同投影的解码成本

| Projection | 16 层合计 | 单层均值 |
|---|---:|---:|
| `q_proj` | 1.376 s | 86.01 ms |
| `k_proj` | 0.266 s | 16.63 ms |
| `v_proj` | 0.272 s | 16.98 ms |
| `o_proj` | 1.265 s | 79.08 ms |
| `gate_proj` | 6.052 s | 378.26 ms |
| `up_proj` | 6.071 s | 379.41 ms |
| `down_proj` | 4.922 s | 307.61 ms |

三个 MLP projection 合计 17.045 秒，占全部 unpack/dequant 的 84.28%。这与权重规模吻合：Llama-3.2-1B 的 intermediate size 为 8192，`gate_proj`、`up_proj` 和 `down_proj` 显著大于 attention projection。

因此 kernel 原型不应先从最小的 k/v projection 开始追求端到端收益。更合理的顺序是：

1. 用 q_proj 的 2048×2048 形状建立正确性和基础 kernel；
2. 尽快覆盖 gate/up 的 8192×2048 以及 down 的 2048×8192；
3. 优先优化 MLP shape，因为它们决定了大部分真实 forward 解码时间。

## 为什么 GEMM 时间看起来只有 16 ms

这里的输入只有 8 tokens，且 RTX 5070 对小 batch BF16 GEMM 很快；每个 Linear 的平均 GEMM 加输出 cast 约 0.146 ms。当前实现却在每次 GEMM 前重建整个 BF16 权重矩阵，所以权重解码成本远高于实际乘法。

这正是 weight-only 推理 kernel 要解决的问题：不能先完整生成 BF16 权重，再调用普通 GEMM。需要按 tile 解包、反量化并立即参与乘加，减少全矩阵物化和多次 kernel launch。

## 方法边界

为了让 GPU 阶段时间可相加，profiling 模式故意在每个阶段执行同步。这会破坏正常异步流水，因此 22.071 秒应作为诊断 wall time，而不是 serving latency。它与未逐阶段同步的 23.70–24.23 秒处于相同量级，主瓶颈结论稳定。

本实验仍然执行 BF16 materialization 和普通 PyTorch GEMM，不能表述为 packed low-bit kernel benchmark。

## 下一步

下一步应停止扩展 Python LRU 策略，开始做一个范围严格的 compiled decode 原型：

1. 先为单个 q_proj record 固定 NQRT v3 输入布局；
2. 用 `torch.utils.cpp_extension` 或 Triton 实现 packed code/beta index 解包；
3. 与当前 `dequantize_chunked` 做逐 bit BF16 对照；
4. 测 kernel launch 数、decode latency 和临时显存；
5. 再扩展到 gate/up/down 三种大 MLP shape；
6. compiled decode 证明有效后，再决定是否与 GEMM 融合。

原始机器数据位于 `results/raw/llama32_1b_lazy_model_phases.json`。

# Llama-3.2-1B NQRT4 packed/BF16 混合驻留 Pareto 报告

## 结论

在单张 NVIDIA GeForce RTX 5070 12 GB 上，项目完成了 Llama-3.2-1B 全模型
NQRT4 packed 权重与 BF16 权重的混合驻留实现，并实测了 6 个显存预算点。所有点均通过
独立校验，且在“CUDA 已分配显存、TTFT、TPOT”三个越小越好的指标上，构成本次有限
候选集合的 Pareto 前沿。

512 MiB 额外驻留预算是一个有代表性的折中点：相对全 packed 路径，CUDA 已分配显存
从 1028.6 MiB 增至 1521.7 MiB，但 TTFT 中位数由 59.74 ms 降至 34.41 ms
（降低 42.4%），TPOT 中位数由 40.42 ms 降至 21.91 ms（降低 45.8%）；同时仍比
全 BF16 resident 的 2390.1 MiB 少 36.3% 已分配显存。

## 实现与选择策略

- 将 112 个 Llama projection 划分为 64 个原子组。共享输入变换的 `q/k/v` 与
  `gate/up` 必须一起切换，避免重复变换和不一致的数据通路；`o_proj`、`down_proj`
  单独成组。
- 对 token 数 1 和 16 分别测量每个 projection 的 packed 与 BF16 GPU 耗时，并按
  0.8/0.2 加权。为降低单层计时噪声，同一 projection family 使用 16 层中位数作为
  预测收益。
- 按“预测节省毫秒 / 增量 MiB”排序，只选择预测收益为正且能放入预算的原子组。
- 每个预算重新加载完整模型；被选中的 projection 保持 BF16 resident，其余 projection
  从同一个 NQRT4 checkpoint 转换为 packed runtime。所有预算使用相同 prompt 和 token。

该选择器是确定性的贪心启发式，目标是得到可解释、可复现的工程方案；它没有证明背包
问题的全局最优解，也没有穷举 64 个原子组的全部组合。

## 固定协议

| 项目 | 设置 |
|---|---|
| 模型 | `meta-llama/Llama-3.2-1B` Base |
| 模型 revision | `4e20de362430cd3b72f300e6b0f18e50e7166e08` |
| checkpoint | NQRT v4、paper Hadamard、shared128 block-LDLQ W4 |
| GPU | NVIDIA GeForce RTX 5070 |
| PyTorch | `2.14.1+cu130` |
| batch / prompt / decode | 1 / 32 token / 16 token |
| warmup / 重复次数 | 3 / 20 |
| 预算 | 0、256、512、768、1024、1400 MiB |
| Pareto 指标 | allocated memory、TTFT median、TPOT median，均最小化 |

TTFT（Time To First Token）是从请求开始到首个输出 token 的时间；TPOT（Time Per
Output Token）是首 token 之后每生成一个 token 的平均时间。

## 全模型实测结果

| 预算 MiB | BF16 / packed projections | CUDA allocated MiB | packed stream MiB | Prefill median ms | TTFT median / P95 ms | TPOT median / P95 ms | token/s/req | 相对 BF16 logit L2 误差 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0 / 112 | 1028.6 | 494.4 | 60.24 | 59.74 / 62.22 | 40.42 / 44.63 | 24.46 | 10.48% |
| 256 | 64 / 48 | 1263.3 | 409.1 | 46.98 | 45.55 / 47.60 | 27.17 / 31.14 | 36.17 | 8.95% |
| 512 | 75 / 37 | 1521.7 | 315.5 | 35.04 | 34.41 / 36.29 | 21.91 / 25.07 | 45.06 | 7.89% |
| 768 | 86 / 26 | 1780.0 | 221.8 | 26.52 | 26.52 / 28.13 | 17.89 / 20.09 | 56.36 | 5.06% |
| 1024 | 96 / 16 | 2014.6 | 136.5 | 20.69 | 20.72 / 24.56 | 15.13 / 16.95 | 66.21 | 4.04% |
| 1400 | 112 / 0 | 2390.1 | 0.0 | 11.11 | 11.27 / 17.15 | 10.45 / 12.48 | 93.10 | 0.00% |

相对全 packed 端点：

| 预算 MiB | TTFT 降低 | TPOT 降低 | 相对全 BF16 仍节省 allocated memory |
|---:|---:|---:|---:|
| 256 | 23.7% | 32.8% | 47.1% |
| 512 | 42.4% | 45.8% | 36.3% |
| 768 | 55.6% | 55.8% | 25.5% |
| 1024 | 65.3% | 62.6% | 15.7% |

## CUDA Graph 集成验证

512 MiB 方案（75 个 BF16、37 个 packed projection）已实际捕获并 replay 到
`NQRTRaggedGraphDecodeRunner`。batch-2 请求在不同 cache position `17/16` 以及旧 slot
释放后 `17/3` 的场景中，Graph 与 ragged eager 的 logits 最大绝对误差均为 0，argmax
完全一致。这证明混合模型可进入现有 ragged CUDA Graph decode 路径；该项只验证正确性，
不作为 Graph 加速数字。

## 可复现命令

```bash
python scripts/profile_hybrid_residency.py \
  --config configs/hybrid_residency.v1.json \
  --checkpoint results/raw/llama32_1b_paper_hadamard_w4_shared128_ldlq_nqrt4.nqmc \
  --output results/raw/llama32_1b_nqrt4_hybrid_profile_v1.json

python scripts/benchmark_hybrid_residency.py \
  --profile results/raw/llama32_1b_nqrt4_hybrid_profile_v1.json \
  --config configs/hybrid_residency.v1.json \
  --checkpoint results/raw/llama32_1b_paper_hadamard_w4_shared128_ldlq_nqrt4.nqmc \
  --output results/raw/llama32_1b_nqrt4_hybrid_pareto_v1.json

python scripts/validate_hybrid_pareto.py \
  results/raw/llama32_1b_nqrt4_hybrid_pareto_v1.json \
  --profile results/raw/llama32_1b_nqrt4_hybrid_profile_v1.json \
  --config configs/hybrid_residency.v1.json \
  --checkpoint results/raw/llama32_1b_paper_hadamard_w4_shared128_ldlq_nqrt4.nqmc \
  --output results/raw/llama32_1b_nqrt4_hybrid_pareto_v1_validation.json

python scripts/promote_hybrid_pareto.py \
  results/raw/llama32_1b_nqrt4_hybrid_pareto_v1.json \
  --profile results/raw/llama32_1b_nqrt4_hybrid_profile_v1.json \
  --config configs/hybrid_residency.v1.json \
  --checkpoint results/raw/llama32_1b_paper_hadamard_w4_shared128_ldlq_nqrt4.nqmc \
  --output results/llama32_1b_nqrt4_hybrid_pareto.json
```

## 证据与结论边界

- 正式、可提交摘要为 `results/llama32_1b_nqrt4_hybrid_pareto.json`，其中记录原始报告、
  profiler、配置和 checkpoint 的 SHA-256；原始大文件保留在被 Git 忽略的 `results/raw/`。
- 本报告只支持单卡、batch=1、prompt=32、decode=16、20 次重复的有限样本结论，不外推为
  生产服务 SLA、网络 QPS 或其他模型结果。
- 6 个点是本次已测方案内的 Pareto 前沿，不声称是所有混合策略的全局 Pareto 前沿。
- 相对 BF16 logit L2 误差随 BF16 驻留增加而下降是本次固定输入上的输出诊断，不能替代
  WikiText-2 PPL 或下游任务质量评测。
- 全 packed 仍慢于全 BF16；本路线的价值是把这一限制转化为可配置的显存—延迟权衡，
  不是宣称 packed kernel 已超过供应商 BF16 GEMM。

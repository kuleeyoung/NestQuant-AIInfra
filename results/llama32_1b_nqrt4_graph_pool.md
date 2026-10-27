# Llama-3.2-1B NQRT4 多 batch CUDA Graph Pool 验证

## 结论

`NQRTGraphDecodePool` 已预建 batch 1/2/4/8 runner，并按活跃请求数选择最小可容纳
容量。active 1/2/3/4/5/7/8 在 RTX 5070 上全部通过：padding 后的有效 logits 与同容量
packed eager 最大绝对误差为 0，argmax 全部一致。

| active | Graph 容量 | padding slots | 首次捕获 allocated 增量 |
|---:|---:|---:|---:|
| 1 | 1 | 0 | 32.25 MiB |
| 2 | 2 | 0 | 32.49 MiB |
| 3 | 4 | 1 | 32.98 MiB |
| 4 | 4 | 0 | 已捕获 |
| 5 | 8 | 3 | 33.96 MiB |
| 7 | 8 | 1 | 已捕获 |
| 8 | 8 | 0 | 已捕获 |

四张图的首次捕获 allocated 增量合计 138.07 MB（131.67 MiB）。全部驻留后，进程记录的
CUDA allocated/reserved 为 1,285,811,200 / 1,436,549,120 bytes。原始机器可读报告位于
被 Git 忽略的 `results/raw/nqrt_graph_pool_validation.json`。

## 复现

```bash
python scripts/benchmark_nqrt_graph_pool.py \
  results/raw/llama32_1b_paper_hadamard_w4_shared128_ldlq_nqrt4.nqmc \
  --offline \
  --output results/raw/nqrt_graph_pool_validation.json
```

## 证据边界

当前池只支持同一轮回放中的请求具有相同 prompt length 和 cache position。padding slot
与有效请求在 batch 维相互独立，所以本实验可以证明容量路由和空槽不会污染有效行；它尚不能
证明不同生成进度的请求可以安全共存，也不构成 continuous batching 或在线服务证据。

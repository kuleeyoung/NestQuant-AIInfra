# NQRT v3 分块 BF16 选择性解码

## 结论

NQRT 选择性解码现在支持 `output_dtype` 和 `rows_per_chunk`。每个 chunk 使用原始 float32 codec 数学重建后立即写入预分配 BF16 输出，避免在 GPU 上同时物化完整 float32 权重和全量解码中间张量。真实 `q_proj` 与 `down_proj` 均已验证与“完整 float32 解码后转 BF16”逐元素一致。

`rows_per_chunk=128` 时：

| Tensor | 原始 decode | 分块 BF16 decode | 原始 CUDA peak | 分块 CUDA peak | 降幅 |
|---|---:|---:|---:|---:|---:|
| 2048×2048 `q_proj` | 71.56 ms | 102.51 ms | 236.51 MiB | 53.78 MiB | **77.26%** |
| 2048×8192 `down_proj` | 337.43 ms | 338.25 ms | 850.01 MiB | 119.13 MiB | **85.99%** |

对最大的 `down_proj`，分块 BF16 几乎不增加延迟，却将 CUDA allocated peak 降低约 731 MiB。`q_proj` 的峰值同样显著下降，但小矩阵上的循环开销使延迟增加 43.24%。因此 chunk size 或是否分块应按 tensor shape/可用显存自适应，而不是所有层固定一个策略。

## Bit-exact dither 重放

初版实现尝试把一个 CPU generator stream 拆成多次小 shape `torch.rand` 调用。小型测试通过，但真实 `down_proj` 的 67,108,864 个 BF16 输出中有 8 个元素不同。原因是不能假设 PyTorch CPU RNG 对“大 shape 一次调用”和“小 shape 多次调用”给出可拼接的完全相同序列。

最终实现保留原始完整随机 draw shape，在 CPU 一次性重建 dither，再按行把 dither chunk 送入 GPU。修正后：

- `q_proj` mismatch：0；
- `down_proj` mismatch：0。

这保证了历史 checkpoint 的 generator-state 语义不因优化而漂移。代价是 CPU 端 dither 仍需整形重放，因此 RSS 没有像 CUDA peak 一样显著下降：q_proj RSS 增加 2.96%，down_proj 仅下降 2.25%。下一步若要降低主机峰值，需要定义可随机访问、可分块跳转的 dither RNG 格式，而不能静默改变现有 NQRT v3 的随机流。

## Down projection chunk sweep

| Rows/chunk | Decode mean | RSS peak increment | CUDA peak allocated |
|---:|---:|---:|---:|
| 64 | 354.22 ms | 1,184.50 MiB | **91.56 MiB** |
| 128 | 338.25 ms | 1,197.05 MiB | 119.13 MiB |
| 256 | 314.38 ms | **1,161.00 MiB** | 174.25 MiB |
| 512 | **301.80 ms** | 1,200.46 MiB | 284.50 MiB |
| 2048 | 320.42 ms | 1,236.75 MiB | 882.01 MiB |

64 rows 最省 GPU 显存，512 rows 在本次短 benchmark 中最快，128 rows 提供偏向显存的折中。由于每档只有 3 次正式重复，这些延迟用于方向判断，不应宣称为稳定生产 SLA。

## API

```python
weight = reader.decode_tensor(
    "model.layers.0.mlp.down_proj.weight",
    device="cuda",
    output_dtype=torch.bfloat16,
    rows_per_chunk=128,
)
```

底层 `PackedNestQuantIndices.unpack_rows(start, end)` 只解包所需行的 residue 与 beta ID。`PackedNestQuantTensor.dequantize_chunked()` 为目标 tensor 预分配输出，并逐 chunk 恢复权重。

## 证据边界

该结果证明单 tensor 解码的设备峰值内存可以显著下降，并保持 BF16 bit-exact。它没有执行 GEMM、模型 forward 或 token generation，也没有实现 packed 权重常驻、lazy layer 调度或 fused kernel。因此不能把 CUDA decode peak 的改善写成端到端模型显存或推理速度提升。

机器可读摘要见 [`llama32_1b_nqrt3_chunked_bf16_decode.json`](llama32_1b_nqrt3_chunked_bf16_decode.json)。

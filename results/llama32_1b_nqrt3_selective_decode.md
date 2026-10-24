# Llama-3.2-1B NQMC selective decode benchmark

## 结论

`NQMCFileReader` 现在支持按 state-dict 名称 O(1) 定位 record、解析 tied alias、只读取一个 tensor，以及选择性解码到指定设备。真实 1.04 GB checkpoint 上，单 record 的 seek/read 明显快于当前 E8 float32 解码；后续优化重点应放在解码中间张量、输出 dtype 和 fused/on-device 实现，而不是文件索引。

| Tensor | Shape | NQRT record | float32 输出 | Read mean | Decode mean | RSS peak + | CUDA allocated peak |
|---|---:|---:|---:|---:|---:|---:|---:|
| layer 0 `q_proj` | 2048×2048 | 2,241,886 B | 16,777,216 B | 5.87 ms | 71.56 ms | 785.19 MiB | 236.51 MiB |
| layer 0 `down_proj` | 2048×8192 | 8,926,555 B | 67,108,864 B | 27.94 ms | 337.43 ms | 1,224.54 MiB | 850.01 MiB |

## 公平测量控制

- GPU：NVIDIA GeForce RTX 5070；
- PyTorch：`2.14.1+cu130`，CUDA 13.0；
- 每个 tensor 在独立进程测量，避免前一个 tensor 的 allocator cache 污染后一个；
- CUDA context 在 RSS baseline 之前初始化；
- 1 次 warmup，3 次计时，表中报告 mean；
- RSS 由 5 ms 采样线程捕获；
- timed run 跳过整个 1.04 GB payload checksum，避免把 O(file size) 扫描混入 O(record size) 延迟；checkpoint 已在独立 inspector 阶段完整校验；
- 每个 raw benchmark manifest 保存 SHA-256，机器摘要可反查原始输出。

## 工程解释

`q_proj` record 只有其 float32 输出的约 13.36%，`down_proj` record 约为 13.30%。按名称选择性读取因此可以避免为访问单层而扫描或反量化整个模型。但当前 `PackedNestQuantTensor.dequantize()` 会构造 residue、beta、dither 和 E8 reconstruction 的中间张量，峰值内存远大于最终输出。

本 checkpoint 的 NQRT `output_dtype` 是 float32；完整模型加载随后再复制/转换到目标 BF16 参数。这提供了明确优化方向：

1. 支持直接输出目标 dtype，避免完整 float32 权重常驻；
2. 按 row/block 分块解码，限制临时张量峰值；
3. 把 unpack、dither reconstruction、E8 representative 恢复和 dtype cast 融合到设备端；
4. 最终将解码与 GEMM 融合，避免显式物化完整权重。

## 证据边界

本实验没有执行 GEMM、模型 forward 或 token generation。它只证明单 tensor 随机访问和选择性解码接口，并给出当前参考解码器的延迟/内存基线；不能据此宣称 lazy-layer 端到端加速或 packed 低比特推理已经实现。

机器可读摘要见 [`llama32_1b_nqrt3_selective_decode.json`](llama32_1b_nqrt3_selective_decode.json)。原始 benchmark JSON 位于 `results/raw/`。

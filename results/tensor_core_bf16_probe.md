# BF16 Tensor Core probe

在 RTX 5070（CUDA capability 12.0）上使用 PyTorch BF16 矩阵乘法验证 Tensor Core 路径可用。测试矩阵为 `[512, 4096] × [4096, 512]`，warmup 10 次，测量 50 次。

```text
dtype: bfloat16
accumulator: framework-selected (FP32 accumulation path)
average: 0.09937 ms
```

这只是 Tensor Core 基线，不代表 packed fused GEMM 已经实现。下一步需要把 packed code 解码后的 BF16 weight tile 接入 WMMA/MMA tile，并与 FP32 参考结果比较误差。

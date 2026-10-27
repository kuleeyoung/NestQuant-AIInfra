# Direct Tensor Core layout prototype

当前选择 B2：packed E8 code 直接解码到 Tensor Core fragment 所需的 BF16 tile，跳过 shared-memory transpose。

## 第一版约束

- tile: `M=16, N=16, K=16`；
- activation: BF16，行主序 `[M, K]`；
- weight tile: BF16，逻辑上为 `[N, K]`；
- accumulator: FP32；
- `K` 必须是 16 的整数倍；
- 每个 E8 block 提供连续 8 个 feature，两个 E8 block 组成一个 `K=16` 片段。

## 解码映射

```text
packed byte payload
  -> code index (按 code_bit_width 解包)
  -> 两个 E8 block 的 coset 坐标
  -> dithered NQRT residual
  -> beta * row_norm / sqrt(feature_count)
  -> BF16 fragment
  -> MMA K=16
```

## 验收标准

1. 直接 fragment layout 与 Python/Triton 反量化结果逐元素对齐；
2. FP32 accumulator 输出与 BF16 PyTorch GEMM 的误差可量化；
3. `M=N=16, K=16` 原型通过后，再扩展到 `K=128/256`；
4. 只有在数值通过后才比较 direct layout 与 shared-memory layout 的性能。

当前尚未宣称 B2 已完成；本文档记录的是实现约束和验证边界。

# CUDA E8 接入边界

## 当前结论

独立 CUDA E8 kernel 已经通过 Python 参考实现和边界样例验证，但它不能直接替换当前完整 packed projection 中的 E8 运算。

当前生产路径位于 `nestquant/runtime/triton_tiled_decode.py` 和 `triton_fused_linear.py` 的 Triton fused kernel 内。单个 kernel 同时完成：

1. stateless dither 生成；
2. E8 最近点投影；
3. 两个 coset 的距离选择；
4. NQRT 残差重建；
5. beta 查表和反量化；
6. 与输入激活的矩阵乘法。

如果只调用独立 CUDA E8 函数，必须把数据从 Triton kernel 写回 global memory，再由另一个 kernel 读取，增加一次 kernel launch 和中间张量写回，无法保证端到端收益。

## 下一步路线

- **CUDA fused decode**：将 dither、E8、beta lookup、反量化和 GEMM 一起迁移到 CUDA C++，目标是获得完整的 CUDA backend。
- **Triton 主路径 + CUDA reference backend**：保留当前 Triton fused path，把 CUDA E8 用于离线校验、边界测试和独立微基准，不宣称端到端加速。

当前证据支持独立 CUDA E8 的局部加速，但尚未支持完整 packed projection 的端到端加速结论。

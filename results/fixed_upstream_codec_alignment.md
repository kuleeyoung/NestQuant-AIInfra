# 固定上游 E8 边界语义与逐阶段数值对齐

## 结论

固定作者 commit `6b269f84264afad990a09248d8eeb45d5f6248ff` 的真实 `coder.py`、`e8.py` 和 `hadamard.py` 已与本项目直接对跑。修正 Voronoi 边界 tie-break 后，8/16/32 维、seed 58/59 共六组输入全部通过：

- 归一化、Hadamard、E8 重构、逆变换和 Linear 输出逐阶段对齐；
- residue code 与 beta index 全部逐元素一致；
- 最大 Linear 输出绝对误差 `2.861023e-6`；
- 全部阶段最大相对 L2 误差 `2.691726e-7`。

这证明的是 2 次幂、单进程、float32、无 dither、`try_all=True` 路径的数值一致性，不覆盖 FairScale `re_glue`、非 2 次幂 Kronecker、LDLQ 或 CUDA kernel。

## 定位到的问题

作者 `encode_e8_fast` 在整数 E8 coset 与半整数 coset 距离完全相等时，使用严格 `<`，因此选择半整数 coset。本项目此前使用 `<=`，在同一边界选择整数 coset：

```text
fixed upstream: equal distance -> half-integer coset
legacy project: equal distance -> integer coset
```

普通随机输入很少恰好处于边界，但 nested decoder 会对 residue representative 除以 `q`，此时点经常精确落在 Voronoi 边界。修复前，两套 encoder 产生的 residue code 可以完全相同，decoder 却选择不同的同余类代表；这会进一步改变 try-all 的重建误差、beta index 和 Linear 输出。因此它不是可忽略的 `1e-7` Hadamard 舍入差异，而是需要版本化的 codec 语义。

## 格式兼容

- 新量化与 NQRT v3：`e8_upstream_half_coset_tie_v1`；
- 旧 NQRT v1/v2：自动标记并使用 `e8_legacy_integer_coset_tie_v1`；
- 未知 codec semantics 会被拒绝，不允许静默猜测；
- NQMC 容器版本保持 v1，但内部 NQRT 记录携带明确语义。

真实旧检查点 `f594a20c...` 已重新回载。首个 2048-token 窗口 PPL 仍为 `7.462738`，共 2,047 个预测 token，与历史结果一致，证明兼容路径不是只在玩具张量上成立。

## 证据与边界

- 对齐原始 JSON SHA-256：`606a9e02aa82cae6d227769f18f970b4247424bb22d40d0f69251bf38a10b9a1`；
- legacy checkpoint smoke SHA-256：`9ab45c362faca6bdc39f8050132bea426007929af817e54550a88dd6ca007c25`；
- 作者源码文件 SHA-256 全部保存在原始对齐 JSON；
- 作者 CUDA Hadamard 扩展没有参与本次 CPU 对齐。上游模块因检测到 CUDA 而提前 import 扩展，harness 注入了一个“被调用即报错”的占位符，实际 Hadamard 仍执行作者原始 SciPy CPU 分支。

历史完整 PPL 报告由 NQRT v2/legacy 语义生成，现有兼容层可以继续复现，但不能自动称为严格上游 codec 语义结果。最佳 W4 checkpoint 已用 NQRT v3 重建并完成 288,627-token 正式评测：上游语义 PPL 为 `10.076886`，legacy PPL 为 `10.076729`，差值仅 `0.000158`。详见 [NQRT v3 模型端点报告](llama32_1b_paper_hadamard_shared128_ldlq_nqrt3.md)。A4/KV4 四档矩阵仍待在新 checkpoint 上重跑。

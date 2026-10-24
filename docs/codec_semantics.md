# E8 codec 语义版本

NestQuant residue code 本身不足以唯一确定 Voronoi 边界上的重构结果；decoder 还必须固定 E8 两个 coset 等距时的 tie-break。项目定义两个明确的语义 ID：

| ID | tie-break | 使用范围 |
|---|---|---|
| `e8_legacy_integer_coset_tie_v1` | 等距时选整数 coset | 历史 NQRT v1/v2 与已有结果 |
| `e8_upstream_half_coset_tie_v1` | 等距时选半整数 coset | 固定作者 commit 对齐后的新量化与 NQRT v3 |

NQRT v3 在每个量化张量的 metadata 中保存 `codec_semantics`。旧 v1/v2 没有该字段，读取器必须按 legacy 语义解释；不能用当前默认 tie-break 解码旧 code。详细定位、真实上游对跑和兼容 smoke 见[对齐报告](../results/fixed_upstream_codec_alignment.md)。

这次版本升级不改变 NQPK residue/beta index 的位打包方式，也不改变外层 NQMC v1 索引结构。改变的是 NQRT 从 residue 恢复 E8 representative 时的数学语义。

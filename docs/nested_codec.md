# E8 嵌套 Voronoi 编解码

当前默认 E8 边界语义与固定作者 commit 一致：整数与半整数 coset 等距时选择半整数 coset。新 NQRT v3 将该语义记录为 `e8_upstream_half_coset_tie_v1`；历史 v1/v2 使用兼容的 `e8_legacy_integer_coset_tie_v1`。相同 residue 在边界处可能解码为不同 representative，因此两种语义不能静默混用，详见[版本说明](codec_semantics.md)和[上游对齐结果](../results/fixed_upstream_codec_alignment.md)。

对应论文 §4.1 第 7 页的 Definition 4.1、Algorithm 1 和 Algorithm 2。代码实现位置为 [nestquant/codec.py](../nestquant/codec.py)，E8 最近点 oracle 为 [nestquant/lattice.py](../nestquant/lattice.py)。

## 编码的直觉

普通格点量化得到最近 E8 格点 `p = Q_E8(x)`。E8 由生成矩阵 `G` 表示，所以 `p = Gv`，其中 `v` 是整数坐标。编码不直接存完整坐标 `v`，而只保存每个坐标除以嵌套比例 `q` 的余数：

```text
p = Q_E8(x)
v = round(G^{-1} p)       # 行向量约定下代码对应 p @ G^{-T}
code = v mod q
```

余数 `code` 标识 `E8 / qE8` 中的一个陪集。对 8 维块，码字数是 `q^8`，每个坐标的理想信息率为 `log2(q)` bit；这只是码字率，不包含 beta 元数据、容器对齐和文件格式开销。`q=14` 的理想率约为 3.807 bit/坐标，不能直接称为已实现 4-bit 文件。

## 解码与最小能量代表

解码器先用余数坐标找到该陪集中的一个点，再将它移动到缩放 Voronoi 区域内：

```text
r = G code
x_hat = r - q Q_E8(r / q)
```

这样每个余数向量对应一个确定的最小能量代表。若原始最近点 `p` 本身位于 `q V_E8`，解码可以还原 `p`；否则发生 overload，解码返回同陪集的另一个代表。`encode_nested` 返回的布尔值记录 overload 状态。

## 实现约定与边界

- 生成矩阵数值取自固定作者版本中的 E8 基；这是可读 PyTorch 参考实现，不是作者的快速 CUDA 解码内核。
- 本实现使用 `nearest_e8` 作为 oracle，并验证格点坐标映射到整数；适合数学参考和小规模实验，不代表高性能 kernel。
- 解码输入必须是整数余数且每个坐标满足 `0 <= code < q`；越界输入明确报错，不静默取模。
- 当前测试枚举 q=2 的全部 256 种码字，验证再编码回到同一余数、无 overload，重构点属于 E8 且处于 `q V_E8`。
- `nestquant/quantizer.py` 已按 Algorithm 3 的结构实现矩阵行归一化、8 维分块、给定 beta 列表逐一量化并按块选最小平方误差，再保存行范数用于反归一化。
- `nestquant/beta_search.py` 已实现 Appendix F 的 First-beta 候选子集动态规划；它通过 `build_candidate_tables` 计算各向量/β的误差与 overload，并由 `select_beta_subset` 选出固定数量的 β。
- DP 用于挑选 beta 集合；量化器随后仍可对该集合采用 Opt-beta，即按块选重构误差最小的 beta。两者的目标不同，代码和文档将它们分别记录。
- beta index 在量化计算中是 int64，但 NQPK/NQRT/NQMC 已将其按实际位宽打包；论文网格完整搜索尚未接入。
- 编解码核心已接入模型级 W4、权重旋转、E8 block-LDLQ、A4/KV4 和 QA-LDLQ 适配。A4/KV4 仍为浮点重构质量仿真，QA-LDLQ 在尺度审计后未优于普通 LDLQ；论文 §4.3 的权重/激活旋转语义尚未与固定上游逐项对齐，也没有低比特推理 kernel，因此不能称完整 NestQuant 复现。

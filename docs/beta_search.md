# Appendix F：First-beta 动态规划

本文实现对应 NestQuant 论文 Appendix F 第 19–21 页 Algorithm 6。核心代码在 `nestquant/beta_search.py`；这里严格区分“选 β 集合”和“使用已选集合量化样本”。

## 两种编码策略

对向量 `v` 和候选缩放码本 `β_i C`，论文描述两种策略：

- **First-β**：按 β 从小到大，选第一个不发生 overload 的码本。
- **Opt-β**：试遍所有 β，选重构 MSE 最小的码本。

论文的动态规划用 First-β 策略来挑选一个大小为 `k` 的 β 子集。得到子集后，Algorithm 3 对每个 8 维块仍然比较重构误差并选最小者，属于 Opt-β。First-β 是子集选择目标，不代表最终量化必须使用 First-β 分配。

## 状态和递推

先对每个样本 `p`、每个 β 候选 `i` 计算：

- `mse[p, i]`：该候选的重构平方误差；
- `overloaded[p, i]`：最近 E8 格点是否超出该缩放码本的 Voronoi 区域。

令 `dp[i, j]` 表示以候选 `i` 为最大 β、已经选 `j` 个 β 时，First-β 已覆盖样本的最小误差和。若前一个选中 β 是 `s`，则新由 `i` 覆盖的样本满足 `overloaded[:, s] & ~overloaded[:, i]`，转移代价是这些样本在 `i` 下的 MSE：

```text
dp[i, j] = min_s(dp[s, j-1] + sum_p(overflow[p,s] and not overflow[p,i]) * mse[p,i])
```

使用 sentinel `s=0` 表示所有样本此前都未覆盖。最终候选必须使所有校准样本都不再 overload。实现会检查 β 严格递增、overload 随 β 增大不重新出现，并拒绝无法覆盖全体样本的候选集。

## 本项目实现边界

- `build_candidate_tables` 用 E8 codec 直接构造 MSE 和 overload 表；输入向量须已经完成 Algorithm 3 的行归一化。
- `select_beta_subset` 只接收表格并执行 Algorithm 6，便于对 DP 与组合穷举单独对照。
- 单元测试用小候选集枚举全部大小为 `k` 的组合，核对动态规划给出的子集和目标误差。
- 论文给出了 β 范围和间距变化范围，但未列出完整候选网格。新项目尚未冻结正式模型搜索网格，也未运行模型校准；不能把测试网格说成论文原始网格。
- 当前 synthetic table 上的 DP 验证不是模型结果，也不证明整套 NestQuant 已复现。

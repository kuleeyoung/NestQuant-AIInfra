# A4 beta 校准协议

## 目的

论文式局部 Hadamard + shared128 LDLQ W4A4 使用固定上游全局 beta 时，完整 WikiText-2 test PPL 为 10.688878，激活 overload 为 5.4411%。本阶段只使用 WikiText-2 train 校准 beta，禁止根据 test PPL 选择候选网格或配置。

## 数据与采样

- 使用版本化 shared128 train manifest 的前 N 个窗口；
- 加载同一个论文式 Hadamard + shared128 LDLQ W4 NQMC，使深层激活来自实际 W4 前向；
- 对 112 个 Q/K/V/O、gate/up/down projection 采集量化坐标输入；
- `HadamardRotatedLinear` 直接采集内部 `X H`，普通 Linear 采集临时 Hadamard 结果；
- 每个模块、每个窗口均匀抽取有界数量的 Algorithm-3 行归一化 E8 block，不保存全部激活。

校准产物记录模型/数据 revision、校准 manifest 及 token 哈希、W4 检查点哈希、候选 beta、样本哈希、候选 overload 和 First-beta 误差。

## 三种配置

同一批样本同时生成：

1. `global`：112 个模块共享一组 4 beta；
2. `projection_type`：q/k/v/o/gate/up/down 七类各共享一组；
3. `module`：112 个模块分别选择一组。

选择目标为 Appendix F First-beta 总平方误差。overload 是约束和审计指标，不单独作为优化目标。所有候选 beta 必须严格递增，最大的候选必须覆盖全部校准样本。

本项目的 codec 在细格最近点投影后判定 overload；真实激活上，细格点随 beta 跳变可能产生“较小 beta 已 fit、较大 beta 暂时重新 overload”的非单调表，而 Appendix-F DP 要求嵌套覆盖。校准器因此使用保守的后缀 OR 单调包络：候选只有在自身及所有更大 beta 都 fit 时才视为稳定覆盖。产物同时保存原始/稳定 overload 计数和被包络改变的 cell 数，不能把修正后的计数冒充原始 codec 行为。

## 入口

校准：

```bash
python scripts/calibrate_llama_activation_betas.py \
  --checkpoint results/raw/llama32_1b_paper_hadamard_w4_shared128_ldlq.nqmc \
  --calibration-chunks 8 --sample-limit-per-module 4096 --offline \
  --output results/raw/calibration/a4_betas_train8.json
```

小规模独立评测时，在原 W4A4 命令中增加：

```bash
--activation-beta-config results/raw/calibration/a4_betas_train8.json \
--activation-beta-profile global
```

在正式 test 前，必须先用未参与校准的 train 窗口选 profile：

```bash
python scripts/evaluate_activation_beta_profiles.py --offline
```

默认协议固定用 manifest 的窗口 0--7 校准、窗口 8--15 验证，并比较 `upstream_default`、`global`、`projection_type`、`module`。脚本会拒绝窗口重叠，并保存 manifest、checkpoint、beta 配置和验证 token 的 SHA-256，以及各方案的 PPL、overload 和相对平方误差。该结果只用于选择 profile，不是最终测试集指标，也不代表低比特激活 kernel 性能。

本轮 held-out 结果见[报告](../results/llama32_1b_a4_beta_holdout.md)：上游默认 PPL 9.782117，校准后的 global / projection-type / module 分别为 10.428573 / 10.557727 / 10.270711。因此保留上游默认配置。该负结果说明降低 overload 的校准代理目标没有转化成更低的端到端 NLL。

`global` 可替换为 `projection_type` 或 `module`。评测 manifest 保存配置绝对路径、SHA-256、profile 和校准 token 哈希。

## 结论边界

- beta 只由 train 数据选择，test 只用于最终一次正式报告；
- 小规模筛选应使用独立 validation token，不能反复使用 test 前几个窗口；
- A4 仍为浮点重构质量仿真，不证明低比特激活存储或 kernel 加速；
- 如果逐模块配置不优于按类型配置，应优先保留更简单的按类型方案；
- 如果三种配置都无改善，需要审计 dither、归一化、量化位置和 QA-LDLQ，而不是扩大网格追逐 test 数字。

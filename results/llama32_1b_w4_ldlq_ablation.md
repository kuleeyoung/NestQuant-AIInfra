# Llama-3.2-1B NestQuant W4 block-LDLQ 消融

## 结论

在模型、数据、测试 token、量化参数、dither seed 与源码哈希完全相同的条件下，加入 8 列 E8 block-LDLQ 后，完整 WikiText-2 test PPL 从 **11.202577** 降至 **10.323714**，绝对下降 **0.878863**，相对下降 **7.85%**。相对 BF16 的 PPL 差距从 **1.445532** 缩小到 **0.566669**，本轮 LDLQ 消除了 **60.80%** 的差距。

这说明当前 block-LDLQ 实现在固定的 16 序列校准条件下显著改善 W4 质量。16 序列结果仍需要更大校准集验证稳定性，不能直接当作论文全部结果的复现。

## 固定协议

- 模型：`meta-llama/Llama-3.2-1B`，revision `4e20de362430cd3b72f300e6b0f18e50e7166e08`
- 测试集：`Salesforce/wikitext/wikitext-2-raw-v1`，revision `b08601e04326c79dfdd32d625aee71d232d685c3`
- 上下文：2048；完整非重叠块；预测 token：288,627
- 测试 token SHA-256：`54927bf19e6e64b4008eeb93370c370fb25db18cf68eac8015cfe094c464d567`
- 项目 Python 源码 SHA-256：`74051211205397bed4a913fae76fdb614cb0af8e1bd73d28a42851372d973776`
- 量化：Linear 权重质量仿真；E8、q=14、每层从 7 个候选中用 DP 选 4 个 beta；CPU dither seed 58
- LDLQ 校准：WikiText-2 train 的 16 个完整序列（32,768 token），相对阻尼 0.01
- 精确选中校准 token SHA-256：`8f310000f176cee30737b9ed4f307af7ae0cd72d13a81e3718184a530be39f3a`
- 完整 train 语料 token SHA-256：`d605904bdc2bf7af8402b7704e9a3845fb291a598284ce3c0c6f8ccd8df8f526`

### 校准文件清单

| Hessian 文件 | 唯一模块统计数 | 字节数 | SHA-256 |
|---|---:|---:|---|
| `hessian_layers00_03_train16.pt` | 16 | 1,275,074,859 | `7791c77e555862226d86a380e5e60e9d650a5cb65dcec5d723f2fe77bc4525cc` |
| `hessian_layers04_07_train16.pt` | 16 | 1,275,074,859 | `7709e664396a55c7509af576f911e791e09b43e8761aad09564516cf06831c01` |
| `hessian_layers08_11_train16.pt` | 16 | 1,275,074,859 | `56d2c697efc9364e8c00bbd46fdfacf882154425c3d2b1e3635a2da3fed98891` |
| `hessian_layers12_15_train16.pt` | 16 | 1,275,074,859 | `9b6ceb40522e1f084bf08c84b215505f1c77dd03f71058b918b183d625f2f96f` |
| `hessian_lm_head_train16.pt` | 1 | 16,779,033 | `182890ecb5479610c99f91f48cca1241fa73f9114a7de757695b5bc972702a10` |

合计 5,117,078,469 字节。大型矩阵不提交 Git；此表中的名称、大小和摘要足以核对本地文件是否是产生该结果的校准数据。

## 完整 PPL 结果

| 方法 | PPL | 平均 NLL | 相对 BF16 的 PPL 增量 | 整个方法耗时 |
|---|---:|---:|---:|---:|
| BF16 | **9.757045** | 2.277990 | 0 | 17.5 s |
| NestQuant W4 | 11.202577 | 2.416144 | +1.445532 | 95.9 s |
| NestQuant W4 + block-LDLQ（train 16） | **10.323714** | **2.334444** | +0.566669 | 625.8 s |

耗时包含模型加载、离线量化和 PPL 评测。它用于说明本机实验成本，不能解释为推理延迟或吞吐。LDLQ 版本当前使用 Python/PyTorch 参考递推，耗时明显高于普通 W4。

## 实现与验证证据

- `nestquant/core/ldlq.py` 实现带 8×8 单位对角块的 block-LDL 分解，以及从右向左的 8 列误差反馈。
- `nestquant/models/calibration.py` 收集 Linear 输入的 `E[x x^T]`；q/k/v 共享输入统计，gate/up 共享输入统计。
- `nestquant/models/hessian_store.py` 按四个 Transformer 层惰性加载约 4.77 GiB 的 FP32 Hessian，避免全部统计同时进入内存或显存。
- 单层 q_proj smoke 中，Hessian 加权代理损失下降 90.41%；down_proj 下降 86.91%。两层的普通 MSE 均上升，符合 LDLQ 优化激活加权误差而非逐权重 MSE 的目标。
- 67 项单元测试覆盖 E8、嵌套码字、beta DP、打包、PPL 口径、旋转协方差、block-LDL 分解、LDLQ 反馈、组合路径、Hessian 惰性加载及 LDLQ NQMC 往返。

## 结果边界

这组实验是 **W4 权重量化质量仿真**：量化结果写回 BF16 权重并执行 BF16 Linear。它没有测量 LDLQ 版本的真实模型文件大小、运行时权重显存、推理延迟或吞吐，也没有实现 A4、KV4、QA-LDLQ 或论文完整旋转图。因此可以陈述“实现并验证了 E8 block-LDLQ，在固定协议下将 PPL 从 11.2026 降至 10.3237”，不能陈述“实现了 W4A4KV4”“获得 INT4 推理加速”或“完整复现论文”。

## 原始证据

- 同版本 BF16/普通 W4 manifest：`results/raw/llama32_1b_current_source_baselines_full_20261021.json`
- LDLQ manifest：`results/raw/llama32_1b_ldlq_train16_full_20261021.json`
- 机器可读汇总：`results/llama32_1b_w4_ldlq_ablation.json`
- 大型 Hessian 与逐层原始记录位于被 Git 忽略的 `results/raw/calibration/`，提交到 GitHub 时保留 manifest 与哈希，不提交约 4.77 GiB 原始矩阵。

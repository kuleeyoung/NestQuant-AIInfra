# Llama-3.2-1B：NestQuant W4 打包回载与 WikiText-2 全量 PPL

2026-10-14 本机实测。模型为 `meta-llama/Llama-3.2-1B` Base，固定 revision `4e20de362430cd3b72f300e6b0f18e50e7166e08`；数据为 `Salesforce/wikitext` 的 `wikitext-2-raw-v1` test split，固定 revision `b08601e04326c79dfdd32d625aee71d232d685c3`。上下文长度 2048，共 141 个完整 chunk、288,627 个预测 token。token ID SHA-256：`54927bf19e6e64b4008eeb93370c370fb25db18cf68eac8015cfe094c464d567`。三项结果都经过同一 tokenizer 与评测口径核对；BF16/RTN 和 NestQuant 在不同 run 中取得。

| 方法 | 权重处理及评测方式 | 全量 PPL | 相对 BF16 增量 |
|---|---|---:|---:|
| BF16 | 原始 BF16 权重 | 9.757045 | — |
| RTN W4 | 每行、每 128 值一组的对称 RTN fake quant；BF16 执行 | 13.021485 | +3.264440 |
| 项目版 NestQuant W4 | E8、每层 β 子集 DP、逐块最优 β；写入 NQMC 后回载到 BF16 模型评测 | 11.202577 | +1.445532 |

本项目 NestQuant W4 的 PPL 比此 RTN W4 基线低 1.818908。此前未导出版本的 PPL 为 11.149369；该版本使用了不同的 dither RNG 生成路径，**不可用此前数值代替本次打包回载结果**。单 chunk 回载 smoke 的 PPL 为 8.303907，只用于确认导出和回载链路，不属于全量结果。

## 实际文件与码率

| 项目 | 实测或计算值 |
|---|---:|
| NQMC 文件大小 | 659,279,649 B（约 628.74 MiB） |
| 113 个量化张量的记录 | 659,123,050 B |
| 量化权重数 | 1,235,746,816 |
| 包括每张量元数据的有效码率 | 4.267043 bit/量化权重 |
| 33 个稠密张量载荷 | 135,168 B |
| tied alias | 1 个；输入 embedding 与 `lm_head` 共用权重 |
| NQMC 文件 / 同一份唯一状态值的 BF16 原始载荷 | 26.6739% |

BF16 分母为同一批唯一状态值按 BF16 存储的**计算载荷** 2,471,628,800 B，未实际导出对照 BF16 文件，因此 26.6739% 是文件对原始载荷之比，不是两个实际模型文件的实测比值。RTN 的 4.25 bit/权重是按 4-bit code 加每组 FP32 scale 计算的估计；目前没有 RTN 打包模型文件。两者码率接近，但不应宣称已完成严格等码率文件与推理对比。

NQMC SHA-256：`6e3f666b50d3eeef9a1a950c1e24a0439f39d9398ddddee0d97ad5b61e44d65d`。大体积模型文件、完整逐层计时、运行 manifest 位于本机 `results/raw/`，不提交 Git。逐层量化合计约 163.2 秒，其中 `lm_head` 约 63.8 秒；这组时间属于量化/导出流程，**不是推理加速数据**。

## 与论文复现目标的差距

- 本次实现有 E8 嵌套量化、β 子集 DP 与逐块选择；模型 β 候选为 `[3.5, 4.5, 6.0, 14.5, 25.0, 40.0, 50.0]`。论文附录 G 报告搜索全集到 40.0/q、默认选中列表到 25.0/q；本项目把 40.0 加入选中候选，50.0 则超出论文报告的搜索全集。
- 该次普通 W4 NQMC 导出未使用旋转或 LDLQ；项目后续已完成二者及组合质量消融，但仍未实现 QA-LDLQ、W4A4KV4 或论文完整旋转图。项目整体仍不能称完整论文复现。
- NQMC 确实保存 4-bit residue 和 2-bit/块的 β 索引；评测时先解码到浮点权重，模型在 BF16 权重上执行。没有低比特矩阵乘 kernel、端到端延迟或吞吐提升的证据。

## 可复核材料

- 原始 manifest：`results/raw/llama32_1b_bf16_rtn_w4_current_full_ppl.json`、`results/raw/llama32_1b_nestquant_w4_nqmc_full_ppl.json`、`results/raw/llama32_1b_nestquant_w4_optimized_smoke_v2.json`。
- 字节核算：`results/raw/llama32_1b_nestquant_w4_nqmc_accounting.json`；其脚本为 `scripts/summarize_nqmc.py`。
- 可提交的小型汇总：[llama32_1b_nqmc_w4_full.json](llama32_1b_nqmc_w4_full.json)，由 `scripts/promote_nqmc_result.py` 在核对完成状态、模型与数据版本、token 哈希和文件 SHA-256 后生成。
- 当前测试入口：`python -m unittest discover -s tests -q`，67 项通过。该历史 manifest 运行时仓库尚无 Git commit，`git_commit=null`，因此 GitHub 上可复现性还需建立提交版本与运行版本的对应关系。

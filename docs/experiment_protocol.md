# 首轮实验协议

本文件定义第一轮论文复现与面试可展示实验的比较边界。没有实际运行的结果保持 `not_run`，论文数据只作参考，不能写入本地实测表。

## 固定对象

| 项目 | 首轮约定 |
|---|---|
| 论文 | NestQuant，arXiv:2502.09720v3，2025-07-26 版本 |
| 作者实现参考 | `cookiedoth/nestquant`，固定 commit `6b269f84264afad990a09248d8eeb45d5f6248ff` |
| 主模型 | `meta-llama/Llama-3.2-1B` Base，不使用 Instruct；需用户账号有权访问 gated 权重 |
| 模型 revision | `4e20de362430cd3b72f300e6b0f18e50e7166e08` |
| 数据 | `Salesforce/wikitext` 的 `wikitext-2-raw-v1` |
| 数据 revision | `b08601e04326c79dfdd32d625aee71d232d685c3` |
| 主指标 | `PPL = exp(total next-token NLL / total predicted tokens)` |
| 序列长度 | 2048；只评估完整、连续、非重叠 chunks；不完整尾段丢弃 |
| 计算 | 先固定 BF16 模型基线，量化质量计算用 float32 logits；模型和算法实际 dtype 逐项记录 |
| 随机种子 | 58；重复性种子 59、60 作为后续检查，不用于挑选验证集结果 |

模型 revision、数据版本和上游 commit 来自旧学习项目已冻结的来源审计；新项目正式跑模型前，应将原始 PDF SHA256、数据文件 SHA256、tokenizer 实测 token hash 和环境版本写入单独的 run manifest。

## 逐步实验矩阵

| 阶段 | 方法/设置 | 目的 |
|---|---|---|
| A | BF16 | 锁定浮点基线和 tokenization 正确性 |
| B | RTN W4 | 最基本 weight-only 量化参照 |
| C | NestQuant W4 | 首先检验论文核心权重量化；与 RTN 使用完全相同模型和评测输入 |
| D | QuaRot W4A4KV4 | 已完成 Llama-3.2-1B 兼容移植：旋转 BF16/GPTQ W4/W4KV4/W4A4/W4A4KV4 PPL 为 9.762884/10.508346/12.269130/15.850334/21.095586；属于纯 PyTorch fake-quant 质量对照，不是官方 kernel 复现 |
| E | NestQuant W4KV4 | 在 NestQuant W4 上单独增加 KV Cache 量化 |
| F | NestQuant W4A4KV4 | 再单独增加激活量化；清楚披露仍未量化的算子 |
| G | 消融 | 逐项关闭旋转、多 beta/DP、LDLQ、QA 噪声适配；一次只变一个因素 |
| H | 部署测量 | 在数值质量完成后实现真实 pack/unpack，再测体积、显存、加载与推理表现 |

NestQuant W4 对 RTN W4 是首要同范围比较。W4A4KV4 对比 W4-only 方法时，必须标明量化范围不同，不能直接据此归因算法优劣。QuaRot 需按它实际支持的模块和配置记录，不能声称未验证过的设置已经等价。

首轮模型脚本将 RTN W4 定义为 Linear 权重按输出行、group size 128 的对称 fake quant；NestQuant W4 为 Linear 权重行归一化、8维 E8 块、q=14、在项目固定候选 `[3.5, 4.5, 6.0, 14.5, 25.0, 40.0, 50.0, 100.0]` 中用 Appendix F DP 选 4 个，再按 Opt-beta 量化。论文附录 G 报告搜索全集到 40.0/q，并从中选出默认列表 `[3.5, 4.5, 6.0, 14.5, 25.0]/q`；本项目把 40.0 加入选中候选，并用超出该报告搜索全集的 50.0 覆盖模型归一化权重块。候选子集仍非论文完整 beta 网格搜索。量化以 512 个输出行为一批处理。两者均以 BF16 浮点权重保存反量化值并继续执行 BF16 计算。NestQuant 已实现 signed-Hadamard 权重旋转和 E8 block-LDLQ 的独立与组合消融，也已完成 post-RoPE 匹配 Q/K 旋转及逐 attention-head KV4 的质量仿真。仍不含论文完整全模型旋转图、A4、QA-LDLQ 或 KV4 低比特 kernel，不能称为完整论文实现。

## 文本与 token 口径

1. 使用指定 split 和固定 tokenizer revision，不套用 chat template。
2. 按数据集原始顺序拼接 text 行，保留空行，以两个换行符连接；全局添加一次 BOS/EOS（如 tokenizer 定义了对应 token）。
3. 以 2048 token 切成连续、不重叠完整 chunks；每段预测 2047 个 next-token 目标，丢弃最后不完整片段。
4. 每个 run 保存原始文本 SHA256、token IDs SHA256、chunk 数和预测 token 数。
5. 使用未平滑的 next-token cross entropy 累加 NLL，再按总预测 token 数计算 PPL；不对 chunk PPL 做简单算术平均。

原文对若干 tokenizer、BOS/EOS 和尾段细节没有完整展开；上表中项目补充约定应在报告中作为实现约定，而非伪称论文逐字规定。

## 必须保存的结果

每次运行都生成唯一 `run_id` 和 JSON manifest，至少包含：commit、配置哈希、模型/tokenizer/data revision、数据文件哈希、seed、Python/PyTorch/CUDA/GPU、量化范围、q/k/beta、校准数据来源、chunk/token 数、NLL/PPL、峰值显存 allocated/reserved、耗时、原始日志位置、协议偏差和 `result_status`。

质量仿真、真实存储和真实推理使用不同阶段状态。未实现 pack 或低比特 kernel 时，不填写低比特文件体积收益或推理加速比。

## 合成矩阵乘积阶段补充

论文 §5.1（第 9 页）使用 A∈R^(4096×4096)、B∈R^(4096×4096)，A、B 元素独立服从 N(0,1)，指标为 `||AB^T - Ahat Bhat^T||_F^2 / (n*m)`。本项目默认的小矩阵仅为 smoke run；正式比较必须报告矩阵尺寸、seed、q、beta 子集和原始矩阵乘积误差。

论文该节给出的 beta 系数候选集合为 `0.5 * {1,...,50}`。作者参考实现传入编码器前使用 `beta = beta_0 / q`（`coder.py` 的 `encode_matrix_try_all` 与 `encode_matrix_first_beta`），所以实验配置必须同时记录论文尺度 beta_0 和 codec 尺度 beta_0/q。本项目按这个映射实现。作者参考编码默认开启减性 E8 dither（`e8.py` 的 `generate_dither`，`coder.py` 的矩阵编码函数）；本项目已实现固定 seed 的共享 dither realization，跨候选 beta 复用以便做可比搜索，这与上游 try-all 每个候选重新生成 dither 的行为仍有差异。论文第 9 页打印的码率表达式为 `log2(q) + (1/8) * sum_i p_i log2(p_i)`；由于 `sum p log2 p` 非正，项目同时记录此原文公式值与采用 Shannon 熵侧信息成本时的 `log2(q) - (1/8) * sum_i p_i log2(p_i)`，不将二者混为实际打包码率。当前 p_i 来自仿真中的 beta 使用频率，未实现熵编码，也未测实际字节数。
# Reproducible LLM PPL experiment

The default model, dataset revisions, quantization settings, and evaluation dtype are maintained in `configs/project.json` under `llm_ppl_experiment`. The evaluator reads those defaults at startup; command-line arguments override only the options exposed by the CLI. Each JSON run manifest stores the effective config and its SHA-256, the project-config SHA-256, Python/package versions, hardware, and Git commit/working-tree status. A null Git commit means the checkout had no commit history when the run started.

Example:

```powershell
python scripts/evaluate_llama_wikitext_ppl.py --limit-chunks 2 --methods bf16
```

Use `--config path\to\project.json` to select another config. Keep the generated raw manifest with the reviewed CSV/Markdown summary when reporting results; never edit a run manifest to make it appear to come from another source state.



Current QA protocol note: the candidate grid includes the disclosed project extension `100/q` because one QA-adjusted block in `model.layers.1.mlp.down_proj` remained overloaded at `50/q`. Only that layer selected 100. QA-LDLQ is implemented; default epsilon constant 155 degraded local full PPL, so the result is recorded as a negative adaptation ablation rather than a reproduced paper gain.

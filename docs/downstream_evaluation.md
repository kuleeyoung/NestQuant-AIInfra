# PIQA 与 HellaSwag 下游评测协议

## 目标

路线三用PIQA和HellaSwag回答一个独立于WikiText-2 PPL的问题：量化强度增加后，模型的多项选择任务准确率和候选排序是否同步退化。

首轮只比较三档：

| 方法 | 作用 |
|---|---|
| BF16 | 浮点能力基线 |
| NQMC W4 | 只使用共享校准权重量化 |
| W4A4KV4 | 当前最激进的权重、激活和KV联合质量仿真 |

## 固定协议

- 模型：`meta-llama/Llama-3.2-1B`固定revision；
- harness：`lm_eval==0.4.13`；
- 任务：`piqa`、`hellaswag`；
- split：两个任务均使用validation；
- zero-shot：`num_fewshot=0`；
- 指标：`acc`和`acc_norm`；
- batch size：4；
- 随机种子：58；
- 最大长度：2048；
- W4/W4A4KV4使用同一个共享校准NQMC；
- 正式结果保存逐样本记录，用于后续答案翻转和候选边距分析。

每份结果还记录数据集validation split的row count、Hugging Face Dataset fingerprint、harness任务YAML SHA256和任务README SHA256，防止只记录任务名称而无法核对实际协议。

## 执行入口

单方法入口：

```bash
python scripts/evaluate_downstream_tasks.py \
  --method bf16 \
  --tasks piqa hellaswag \
  --batch-size 4 \
  --offline-model \
  --output results/raw/downstream_bf16.json
```

W4和W4A4KV4必须传入共享校准NQMC。W4A4KV4还读取版本化KV beta配置。每次harness前向都会创建独立的`NestQuantDynamicCache`，避免不同候选答案之间复用KV状态。

## 2026-10-24夜间运行

Windows任务计划名称：`NestQuant-PIQA-HellaSwag-20261024`。

- 触发时间：北京时间2026年10月24日00:05；
- 脚本：`scripts/run_downstream_full_overnight.sh`；
- 顺序：BF16 → W4 → W4A4KV4；
- 输出目录：`results/raw/downstream_full_20261024/`；
- 主日志：`overnight.log`；
- 总状态：`overnight_status.json`；
- 机器可读汇总：`summary.json`；
- 可审阅报告：`results/llama32_1b_downstream_full.md`；
- 模型和数据集均使用本地缓存离线运行；
- 已完成的单方法JSON会被跳过，因此支持再次触发后断点续跑。

## 已完成smoke

- BF16：PIQA/HellaSwag各2题，通过；
- W4：PIQA/HellaSwag各2题，通过；
- W4A4KV4 batch=2：各2题，通过；
- W4A4KV4 batch=4：各2题，通过；
- 完全离线BF16：各1题，通过；
- 数据缓存：PIQA validation 1,838题，HellaSwag validation 10,042题。

smoke分数的样本数太小，只用于验证代码路径，不能进入简历或正式质量结论。

完整三档结果生成后，`scripts/summarize_downstream_results.py` 会先核对模型、revision、任务、lm-eval版本、数据指纹、任务定义哈希以及逐样本身份，再计算分数差和相对BF16的逐题正确性变化。任一协议或样本集合不一致时，汇总会失败，不生成可用于比较的报告。

## 证据边界

W4通过NQMC真实低比特文件回载，但计算时仍使用BF16 Linear。A4与KV4仍是fake quant。下游准确率可以证明量化质量变化，不能证明模型显存按4-bit常驻或推理得到加速。

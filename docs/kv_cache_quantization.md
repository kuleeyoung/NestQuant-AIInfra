# NestQuant KV4 质量仿真

本项目已在 Llama-3.2-1B Base 上完成 NestQuant W4KV4 的完整 WikiText-2 质量仿真。实现位置是 RoPE 之后、写入 KV cache 之前：query 与 key 使用匹配的归一化 Hadamard 旋转，key 和 value 按每个 attention head 的 64 维向量独立归一化，再切成 8 维 E8 块执行多 beta 量化与浮点重构。

这一布局来自固定的作者公开实现：`parallel/model.py` 在 RoPE 后旋转 query/key 并在 cache 前量化 key/value；`parallel/quant_utils.py` 把最后一维之外的维度展平，因此 Llama-3.2-1B 的量化行是单个 64 维 head。上游固定提交为 `6b269f84264afad990a09248d8eeb45d5f6248ff`。

## 已实现模块

- `NestQuantDynamicCache`：在 `DynamicCache.update` 之前对 K/V 做 Algorithm 3 E8 fake quant。
- 匹配 Q/K Hadamard：在相同的 post-RoPE 坐标上旋转 query 与 key。
- 分层 K/V beta：支持从 JSON 为每层 key 和 value 分别加载 4 个 beta。
- 六段 train 校准：从 WikiText-2 train 取 6 个 2048-token 窗口，每层每类最多保留 65,536 个 E8 块，用 Appendix F 风格 DP 搜索 beta 子集，并在最大 beta 上加入附录 G 的 `4/q` 裕量。
- 公开默认 beta 消融：key 系数 `[3.50, 4.58, 6.47, 17.06]`，value 系数 `[3.53, 5.59, 9.62, 29.03]`，实际 beta 为系数除以 `q=14`。
- 审计统计：记录每层 beta 直方图、过载数、MSE、信号能量、最大误差、向量数和理论码率。

## 完整实验结果

四组实验都使用相同模型 revision、WikiText-2 test token、NQMC W4 block-LDLQ 文件和 Python 源码哈希。完整表格与哈希见 [`results/llama32_1b_w4kv4_ablation.md`](../results/llama32_1b_w4kv4_ablation.md)。

| 设置 | PPL | 相对 W4 |
|---|---:|---:|
| W4 LDLQ + floating DynamicCache | 10.276133 | 0.000% |
| W4 LDLQ + 匹配 Q/K 旋转 | 10.280920 | +0.047% |
| W4KV4 + 作者公开默认 beta | 10.712355 | +4.245% |
| W4KV4 + 本地六段 beta DP | 11.333720 | +10.292% |

匹配旋转对照与普通缓存只相差 0.047%，因此主要质量损失来自 KV4。本地 DP 在自己的校准样本上优化 E8 块重构误差，但测试集相对平方误差和 PPL 都差于公开默认 beta。这说明当前六段样本和局部 MSE 目标的泛化不足，不能把 beta 子集搜索等同于端到端任务最优。

## 理论码率与证据边界

每个 64 维 K/V head 的估算开销为：

- E8 residue：每个元素 4 bit；
- beta index：每 8 维 2 bit，即每元素 0.25 bit；
- 行范数：每 64 维 32 bit，即每元素 0.5 bit。

合计为 **4.75 bit/entry**。该数字是分析估算。当前 cache 实际保存的是反量化后的浮点张量，尚未实现 packed KV buffer、融合量化/反量化 kernel 或低比特 attention kernel，因此这些实验只证明 PPL 质量，不能证明 KV cache 显存已经下降，也不能证明推理加速。

## 复现命令

```powershell
python scripts/create_reference_kv_beta_config.py
python scripts/calibrate_llama_kv_betas.py --offline --output results/raw/calibration/kv_betas_train6_perhead_qkrot_margin4.json
python scripts/evaluate_llama_wikitext_ppl.py --methods nestquant_w4 --load-checkpoint results/raw/llama32_1b_nestquant_w4_ldlq_train64.nqmc --kv4 --kv-beta-config configs/kv_betas_upstream_defaults_q14.json --offline --output results/raw/llama32_1b_w4kv4_upstream_betas_full.json
python scripts/summarize_w4kv4_ablation.py
```

作者公开仓库说明 beta DP 脚本尚待发布，因此本地六段 DP 是本项目的实验实现，不冒充作者原始校准流水线。

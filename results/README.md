# 结果目录

只提交经核验的小型汇总表、配置快照和图表源数据。每个数字必须能追溯到 `run_id`、JSON manifest、命令、日志、配置哈希和代码版本。

大模型权重、缓存、原始数据和大体积日志留在本机目录，不上传 Git。论文参考值单独标注来源，不能混入本地实测结果。首轮 Llama-3.2-1B WikiText-2 质量仿真已完成，结论和边界见 [首轮结果说明](llama32_1b_wikitext2_first_run.md)，可复核 manifest 与 CSV 也保存在本目录。

当前推荐引用结果、历史结果和暂缓结果的分层见 [`docs/current_results.md`](../docs/current_results.md)。新报告和简历应优先引用 shared128、BF16 `lm_head` 的正式矩阵；首轮、train64 和普通 W4 NQMC 文档作为开发过程证据保留，不用较旧数字覆盖当前主结果。

PIQA/HellaSwag zero-shot 全量 BF16/W4/W4A4KV4 已于 2026-10-24 完成。经协议与逐样本身份校验的分数、耗时和正确性翻转见[正式下游报告](llama32_1b_downstream_full.md)和[机器可读汇总](llama32_1b_downstream_full.json)；约 57 MB/方法的原始逐样本 JSON 与日志继续保留在 `results/raw/downstream_full_20261024/`，不进入 Git。

NQMC 加载器已经完成从整文件/整 state-dict 常驻到分块校验、逐记录回载的改造。同一个 1.04 GB 文件两种加载方式各 3 次的内存与时间对照见[流式回载工程基准](llama32_1b_nqmc_load_benchmark.md)；该数据只属于加载阶段，不作为推理加速证据。

论文 §4.3 的局部 `W H`/`X H` 路径已经接入 W4、LDLQ 坐标变换、流式 NQMC 导出和独立回载。Llama-3.2-1B 首窗口三档控制与检查点闭环见[论文式 Hadamard W4/NQMC smoke](llama32_1b_paper_hadamard_w4_smoke.md)；该报告不把 2,047-token 差值写成正式质量提升。

完整 288,627-token 对照已经确认：相同范围的普通 W4 PPL 为 11.054059，论文式局部 Hadamard W4 为 10.917094，绝对下降 0.136964、相对下降 1.239%。固定协议、检查点哈希和结论边界见[论文式 Hadamard W4 完整报告](llama32_1b_paper_hadamard_w4_full.md)。

论文式 Hadamard 与 shared128 block-LDLQ 的组合 PPL 为 10.076729，相对仅 LDLQ 的 10.211760 继续下降 1.322%；两项收益近似可加。四因素消融、自包含 NQMC 和结论边界见[Hadamard/LDLQ 组合报告](llama32_1b_paper_hadamard_ldlq_ablation.md)。

局部 `X H` 坐标 A4 已接入该最佳 W4 检查点并完成全量评测：W4/W4A4 PPL 为 10.076729/10.688878，A4 增加 6.075%；112 个模块均在局部坐标量化，完整 overload 为 5.4411%。见[W4A4 完整报告](llama32_1b_paper_hadamard_w4a4_full.md)和[首窗口 smoke](llama32_1b_paper_hadamard_w4a4_smoke.md)。

A4 beta 已在互不重叠的 WikiText-2 train 窗口上完成校准与 held-out 选择。上游默认/global/projection-type/module 的 held-out PPL 分别为 9.782117/10.428573/10.557727/10.270711，因此不推广新配置并保留上游默认；详见[独立 held-out 报告](llama32_1b_a4_beta_holdout.md)和[机器可读汇总](llama32_1b_a4_beta_holdout.json)。原始逐模块统计留在 `results/raw/calibration/`，不进入 Git。

局部 Hadamard + shared128 LDLQ 最佳 W4 检查点的四档递进已经闭环：W4/W4KV4/W4A4/W4A4KV4 PPL 为 10.076729/10.524110/10.688878/11.231160。A4 的单独损失大于 KV4，组合存在 `+0.006048` NLL 的小幅额外交互；见[完整递进报告](llama32_1b_paper_hadamard_w4a4kv4_progression.md)和[机器可读汇总](llama32_1b_paper_hadamard_w4a4kv4_progression.json)。

固定作者 commit 的真实源码逐阶段对跑发现并修复了 E8 Voronoi 边界 tie-break 差异。新 NQRT v3 使用上游半整数 coset 语义，旧 NQRT v1/v2 由 legacy decoder 保持可复现；六组对齐和真实旧 checkpoint smoke 见[正式报告](fixed_upstream_codec_alignment.md)及[机器汇总](fixed_upstream_codec_alignment.json)。现有完整模型数字属于明确标注的 legacy codec 结果，后续将用 v3 重建正式端点。

论文 §4.3 的局部 `WH` 权重与在线 `XH` 输入路径已完成短 token 数值 smoke；1/7/112 个 Linear 的相对 L2 logits 误差为 0.9554%/1.0751%/1.0964%，本次 top-1 均一致。该结果只用于旋转语义排错，见 [Hadamard 坐标 smoke](llama32_1b_hadamard_coordinate_smoke.md)。

## WikiText-2 PPL 汇总

模型评测脚本每次运行都会在 `results/raw/` 生成独立 JSON manifest（包括失败和 smoke run）。完成若干次评测后，可用以下命令生成便于查看的 CSV：

```powershell
python scripts/summarize_ppl_results.py
```

默认输出 `results/ppl_summary.csv`。相对 BF16 的 PPL 变化只在同一份 manifest 同时含有 BF16 和对应方法结果时计算；汇总保留 run 状态、token 数、方法耗时、配置哈希和原始 manifest 路径。`smoke_completed` 仍是短测，不能当作完整 WikiText-2 结果。CSV 默认留在本地结果目录，不自动提交 Git。评测入口会在控制台标出数据加载、量化和 PPL 计算阶段，并在 manifest 中记录当前/失败方法，便于定位中断位置。


## Local smoke run

`raw/matrix_product_toy_smoke.json` is an n=8, m=8, k=16, seed 58 code-path check. It uses the beta step 0.5 from paper Section 5.1, but the matrices are much smaller than the paper's 4096 setting and the effective bit rates are not matched. The file is excluded by `.gitignore` and is not a paper reproduction result; rerun the same command to regenerate it.


Post-hoc effective W4 bitrate estimates and the tied embedding/`lm_head` caveat are documented in [llama32_1b_w4_rate_accounting.md](llama32_1b_w4_rate_accounting.md). They are packing assumptions, not measured checkpoint sizes.

后续已完成真实 NQMC 打包文件、回载和全量 WikiText-2 PPL；测得文件字节数、有效码率和证据边界见 [模型级结果](llama32_1b_nqmc_w4_full.md)。首轮文档中的“尚无模型级文件”仅对应当时的实验阶段。

W4KV4 已完成相同 NQMC W4 权重下的 DynamicCache、匹配 Q/K 旋转、本地六段 beta DP 和作者公开默认 beta 四组完整消融；自动校验后的指标、哈希和质量/存储/速度边界见 [W4KV4 消融结果](llama32_1b_w4kv4_ablation.md)。

W4A4KV4 已完成 batch=4 的同源递进实验；W4、W4KV4、W4A4KV4 的完整 PPL、峰值 CUDA allocated、哈希一致性与证据边界见 [W4A4KV4 递进结果](llama32_1b_w4a4kv4_progression.md)。

QuaRot 已完成 Llama-3.2-1B 兼容移植和同一测试 token 流的完整质量对照。旋转 BF16、GPTQ W4、W4KV4、W4A4、W4A4KV4 PPL 分别为 9.762884、10.508346、12.269130、15.850334、21.095586；实现修复、协议差异和可陈述边界见 [NestQuant 与 QuaRot 对照](llama32_1b_quarot_comparison.md)。

NestQuant 与 QuaRot 的 W4 已进一步对齐到相同的 128 个校准窗口和 BF16 `lm_head` 范围。NestQuant block-LDLQ W4 PPL 为 10.211760，QuaRot 兼容移植 GPTQ W4 为 10.508346；配置、哈希、范围差异和证据边界见 [共享校准与同范围 W4 对照](llama32_1b_shared_calibration_w4.md)。

共享校准和 BF16 `lm_head` 范围下的 NestQuant W4/W4KV4/W4A4/W4A4KV4 已全部完成，PPL 分别为 10.211760、10.668994、10.952032、11.504096。相同校准及测试哈希下的 QuaRot 兼容移植四项结果也已核对；完整单因素、组合交互、NQMC 三路回载一致性和证据边界见 [共享校准递进报告](llama32_1b_shared128_w4a4kv4_progression.md)。


The complete negative W4 and W4A4KV4 QA-LDLQ ablation, protocol differences, and raw-manifest index are in [the QA-LDLQ report](llama32_1b_qa_ldlq_ablation.md).

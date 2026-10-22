# Llama-3.2-1B 共享校准 W4/A4/KV4 递进实验

## 结论

在同一个 Llama-3.2-1B、WikiText-2 测试 token 流、128 个训练校准窗口和 BF16 `lm_head` 范围下，NestQuant 的完整递进结果为：W4 `10.211760`、W4KV4 `10.668994`、W4A4 `10.952032`、W4A4KV4 `11.504096`。相对于 W4，KV4、A4 和二者同时启用分别增加 `0.457234`、`0.740272` 和 `1.292336` PPL。

A4 与 KV4 的单独退化之和对应 PPL `11.409266`，实际 W4A4KV4 比它再高 `0.094830`。这说明在本次实验中两种运行时量化存在小幅正交互退化，不能只用两个单项结果相加预测组合结果。

## 固定协议

- 模型：`meta-llama/Llama-3.2-1B`，revision `4e20de362430cd3b72f300e6b0f18e50e7166e08`
- 数据：`Salesforce/wikitext` / `wikitext-2-raw-v1`，revision `b08601e04326c79dfdd32d625aee71d232d685c3`
- 校准：WikiText-2 train，128 × 2048 token
- 校准 token SHA256：`358a77b403d372a11957df15dbf46ef3e6f54025d315eaf3514c40bd561d90bb`
- 测试：141 × 2048 token，288,627 个 next-token 预测
- 测试 token SHA256：`54927bf19e6e64b4008eeb93370c370fb25db18cf68eac8015cfe094c464d567`
- 权重范围：112 个 Llama `Linear` 权重；`lm_head` 保持 BF16
- 权重算法：8-column E8 block-LDLQ，`q=14`，relative damp `0.01`
- A4：Linear 输入处的 E8 fake quant；浮点重构后继续计算
- KV4：post-RoPE K/V E8 fake quant；浮点重构后写入运行时 cache

## NestQuant 递进结果

| 配置 | PPL | mean NLL | 相对 W4 ΔPPL | 运行耗时 |
|---|---:|---:|---:|---:|
| W4 | 10.211760 | 2.323540 | 0 | 816.8 s |
| W4KV4 | 10.668994 | 2.367342 | +0.457234 | 301.5 s |
| W4A4 | 10.952032 | 2.393525 | +0.740272 | 741.9 s |
| W4A4KV4 | 11.504096 | 2.442703 | +1.292336 | 912.1 s |

不同运行耗时包含不同的 fake-quant 路径和统计开销，不能解释为低比特 kernel 的速度对比。

## 同协议 QuaRot 兼容移植对照

四个 QuaRot manifest 的校准哈希与测试哈希均和上面的 NestQuant 实验一致。对照使用 QuaRot 的 GPTQ W4 和标量 per-token A/K/V fake quant；这是为 Llama-3.2-1B 做过兼容修改的移植，不是 QuaRot 官方发布的该模型结果，也不是 CUDA kernel 性能复现。

| 配置 | NestQuant PPL | QuaRot 兼容移植 PPL | NestQuant - QuaRot |
|---|---:|---:|---:|
| W4 | 10.211760 | 10.508346 | -0.296586 |
| W4KV4 | 10.668994 | 12.269130 | -1.600136 |
| W4A4 | 10.952032 | 15.850334 | -4.898302 |
| W4A4KV4 | 11.504096 | 21.095586 | -9.591490 |

这张表证明的是在固定模型、数据、校准样本、测试 token 和量化范围下，两套已记录实现的 PPL 差异。由于 A/K/V 量化器结构不同，它不能单独证明 NestQuant 在所有实现或硬件上普遍优于 QuaRot。

## 权重文件回载闭环

共享校准 W4 权重已保存为 NQMC v1：

- 文件：`results/raw/llama32_1b_nestquant_w4_shared128_no_lm_head.nqmc`
- 大小：1,044,557,140 bytes（约 0.973 GiB）
- SHA256：`5eb8dd042887aaacb6b1c2d619019e50c422c868d06ed60fc668d1b69239713e`
- 内存量化后 PPL：`10.211759713549691`
- 导出后立即回载 PPL：`10.211759713549691`
- 独立进程仅回载 PPL：`10.211759713549691`

三条路径逐位对应到相同 PPL，说明保存和回载没有改变本次质量结果。NQMC 回载仍把权重反量化为 BF16 `Linear` 执行，因此这个闭环证明真实低比特权重存储和可回载性，不证明 INT4 GEMM 加速。

## 证据边界

- 已证明：固定协议 PPL、W4 NQMC 实际文件字节数、文件哈希、无 Hessian 独立回载、A4/KV4 单因素与组合消融。
- 尚未证明：A4/KV4 的真实低比特显存占用、端到端吞吐提升、首 token 延迟、逐 token 延迟、INT4 Tensor Core kernel。
- QuaRot 是兼容移植对照；报告中不能写成“官方 Llama-3.2-1B 结果”。

## 原始证据索引

- W4：`results/raw/llama32_1b_nestquant_w4_shared128_no_lm_head_full_20261022.json`
- W4 导出回载：`results/raw/llama32_1b_nestquant_w4_shared128_no_lm_head_export_full_20261022.json`
- W4 独立回载：`results/raw/llama32_1b_nestquant_w4_shared128_no_lm_head_loadonly_full_20261022.json`
- W4KV4：`results/raw/llama32_1b_nestquant_w4kv4_shared128_no_lm_head_full_20261022.json`
- W4A4：`results/raw/llama32_1b_nestquant_w4a4_shared128_no_lm_head_full_20261022.json`
- W4A4KV4：`results/raw/llama32_1b_nestquant_w4a4kv4_shared128_no_lm_head_full_20261022.json`
- QuaRot 四项：`results/raw/quarot_official_gptq128_w4*_full_20261022.json`

提交到 Git 的紧凑机器可读摘要见 [同名 JSON](llama32_1b_shared128_w4a4kv4_progression.json)。`results/raw/` 和 NQMC 大文件按仓库规则保留在本机，不上传 GitHub。

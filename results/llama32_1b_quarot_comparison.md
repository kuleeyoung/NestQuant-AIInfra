# Llama-3.2-1B：NestQuant 与 QuaRot 兼容移植版质量对照

实验日期：2026-10-22。

## 结论

在固定的 Llama-3.2-1B、WikiText-2 test token 流和统一 NLL 聚合器上，QuaRot 兼容移植版的 GPTQ W4 PPL 为 10.508346，说明权重旋转和 GPTQ 路径工作正常；加入 A4/KV4 后 PPL 为 21.095586。当前 NestQuant block-LDLQ W4A4KV4 为 11.578171。两者完整配置的差值是 9.517415，NestQuant 相对 QuaRot 兼容移植版降低 45.12% PPL。

该结果支持“在本项目固定协议和兼容移植实现中，NestQuant 的 E8 激活/KV 质量仿真明显优于 QuaRot 的标量 4-bit 路径”。它不等于复现了 QuaRot 论文在官方支持模型上的原始数字，也不证明真实低比特吞吐、显存或模型文件优势。

## 共同评测协议

- 模型：`meta-llama/Llama-3.2-1B`
- 模型 revision：`4e20de362430cd3b72f300e6b0f18e50e7166e08`
- 数据：`Salesforce/wikitext/wikitext-2-raw-v1` test
- 数据 revision：`b08601e04326c79dfdd32d625aee71d232d685c3`
- 上下文长度：2048
- 完整窗口：141
- 预测 token：288,627
- test token SHA256：`54927bf19e6e64b4008eeb93370c370fb25db18cf68eac8015cfe094c464d567`
- logits/loss：FP32 cross entropy，按全部 token 总 NLL 聚合

## QuaRot 兼容移植版结果

| 配置 | PPL | 相对上一控制增量 |
|---|---:|---:|
| 旋转 BF16 | 9.762884 | — |
| GPTQ W4 | 10.508346 | +0.745462，相对旋转 BF16 |
| GPTQ W4KV4 | 12.269130 | +1.760784，相对 W4 |
| GPTQ W4A4 | 15.850334 | +5.341988，相对 W4 |
| GPTQ W4A4KV4 | 21.095586 | +10.587239，相对 W4 |

正式量化配置采用官方 README 示例的关键参数：Hadamard rotation、GPTQ W4、`w_clip=true`、per-channel 权重、对称 per-token A4/K4/V4、`percdamp=0.01`、`act_order=false`、128 条长度 2048 的 WikiText-2 train 校准序列。校准 token SHA256 为 `358a77b403d372a11957df15dbf46ef3e6f54025d315eaf3514c40bd561d90bb`。

W4A4KV4 完整运行中，GPTQ 用时 234.17 秒，PPL 评测用时 32.24 秒。时间只用于记录本机实验成本；纯 PyTorch Hadamard fallback 和 fake quant 没有真实低比特 kernel，不能把这些时间解释成部署吞吐。

## NestQuant 对照

| 配置 | PPL | 相对上一控制增量 |
|---|---:|---:|
| WSL BF16 | 9.758112 | — |
| block-LDLQ W4 | 10.276601 | +0.518489，相对 BF16 |
| block-LDLQ W4KV4 | 10.732014 | +0.455413，相对 W4 |
| block-LDLQ W4A4KV4 | 11.578171 | +1.301570，相对 W4 |

NestQuant W4 使用 64 个训练窗口的 8-column E8 block-LDLQ checkpoint；A4/KV4 使用 E8 嵌套格、多 beta 和浮点重构质量仿真。

## 如何解释结果

QuaRot W4 只比 NestQuant W4 高 0.231745 PPL，因此不能说当前优势主要来自权重量化。差异主要出现在 A4/KV4：QuaRot 从 W4 到 W4A4KV4 增加 10.587239 PPL，NestQuant 增加 1.301570。当前证据指向 E8 向量量化、多 beta 选择及各自激活/KV 处理方式对低比特激活质量的差异。

A4 是 QuaRot 移植版的主要误差来源：W4A4 比 W4 增加 5.341988，而 W4KV4 增加 1.760784。A4 与 KV4 同时启用时还有叠加误差，不能把总退化简单分摊给单一模块。

## 公平性边界

两条路线共享模型、测试数据、token 流和 PPL 聚合器，但内部量化器并不相同：

- QuaRot 使用官方 GPTQ 标量 W4，保留 `lm_head` 为 BF16；NestQuant 使用 E8 block-LDLQ，并量化 `lm_head`。
- QuaRot A/K/V 是对称 per-token 标量 4-bit fake quant；NestQuant A/K/V 是 8 维 E8 嵌套格、多 beta fake quant。
- QuaRot 使用 128 个随机训练窗口；NestQuant block-LDLQ checkpoint 使用 64 个顺序训练窗口。
- QuaRot 官方仓库没有原生支持 Llama-3.2-1B。本结果来自明确记录的兼容移植：纯 PyTorch Hadamard fallback、GQA 修复、共享 embedding/lm_head 修复和 Transformers 4.46 GPTQ 设备修复。
- 两者的运行时 A/K/V 张量都以浮点重构形式参与计算。该实验衡量质量，不衡量真实低比特存储或 kernel 加速。

因此简历可写“完成同模型、同数据、同 token 流下的 QuaRot 兼容移植与质量对照”，不能写“官方 QuaRot 复现结果”或“实现 QuaRot CUDA kernel”。

## 原始证据

- `results/raw/quarot_rotated_bf16_full_20261022.json`
- `results/raw/quarot_official_gptq128_w4_full_20261022.json`
- `results/raw/quarot_official_gptq128_w4kv4_full_20261022.json`
- `results/raw/quarot_official_gptq128_w4a4_full_20261022.json`
- `results/raw/quarot_official_gptq128_w4a4kv4_full_20261022.json`
- `results/raw/llama32_1b_w4a4kv4_upstream_betas_batch4_full_20261021.json`
- `external_patches/quarot_llama32_pytorch_fallback.patch`

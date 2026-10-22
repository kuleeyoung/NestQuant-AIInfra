# Llama-3.2-1B 共享校准与同范围 W4 对照

实验日期：2026-10-22。

## 结论

NestQuant block-LDLQ 与 QuaRot 兼容移植版 GPTQ 现在使用完全相同的 128 个 WikiText-2 train 校准窗口。校准 token 总哈希均为 `358a77b403d372a11957df15dbf46ef3e6f54025d315eaf3514c40bd561d90bb`，测试 token 哈希均为 `54927bf19e6e64b4008eeb93370c370fb25db18cf68eac8015cfe094c464d567`。

当两者都保留 `lm_head` 为 BF16 时，NestQuant block-LDLQ W4 PPL 为 **10.211760**，QuaRot GPTQ W4 为 **10.508346**。NestQuant 低 **0.296586 PPL**，按 QuaRot PPL 计算相对降低 **2.82%**。

这是同模型、同校准 token、同测试 token、同 `lm_head` 范围的质量对照。两者的量化器仍不同：NestQuant 使用 E8 multi-beta block-LDLQ，QuaRot 使用旋转加标量 GPTQ，因此该结果衡量各自完整 W4 方法的质量，不是单独证明 E8、旋转或 LDLQ 某一个模块的贡献。

## 结果

| 方法 | 校准 | `lm_head` | 量化 Linear | 量化参数 | PPL |
|---|---|---|---:|---:|---:|
| NestQuant block-LDLQ W4 | 128 个共享窗口 | W4 | 113 | 1,235,746,816 | 10.277472 |
| NestQuant block-LDLQ W4 | 128 个共享窗口 | BF16 | 112 | 973,078,528 | **10.211760** |
| QuaRot 兼容移植 GPTQ W4 | 128 个共享窗口 | BF16 | 兼容移植范围 | 兼容移植范围 | 10.508346 |

在 NestQuant 内部，仅将 `lm_head` 从 W4 改为 BF16，PPL 降低 **0.065712**。这说明之前把量化范围差异混入算法比较会低估 NestQuant 的 W4 质量，也说明简历或报告必须写清 `lm_head` 是否量化。

## 协议与证据

- 模型：`meta-llama/Llama-3.2-1B`，revision `4e20de362430cd3b72f300e6b0f18e50e7166e08`。
- 校准：WikiText-2 train，128×2048 token，有放回随机窗口，seed 58。
- 评测：WikiText-2 test，141 个完整窗口，288,627 个 next-token 预测。
- NestQuant：8-column E8 block-LDLQ，relative damp 0.01，E8 multi-beta，BF16 反量化计算。
- QuaRot：Llama-3.2-1B 兼容移植，Hadamard rotation + GPTQ W4，BF16 `lm_head`。
- 五组 NestQuant Hessian 共 5,117,078,564 字节，是离线校准产物，不属于部署模型文件。

原始证据保存在被 Git 忽略的 `results/raw/`：

- `llama32_1b_nestquant_w4_shared128_full_20261022.json`
- `llama32_1b_nestquant_w4_shared128_no_lm_head_full_20261022.json`
- `quarot_official_gptq128_w4_full_20261022.json`

机器可读汇总为 [`llama32_1b_shared_calibration_w4.json`](llama32_1b_shared_calibration_w4.json)。

## 证据边界

本实验使用 fake quant / 浮点重构执行，只证明 PPL 质量。它不证明 INT4 GEMM、推理加速、低比特权重常驻显存或低比特激活存储。QuaRot 数据来自明确记录的 Llama-3.2-1B 兼容移植，也不应表述为 QuaRot 官方原生模型结果。

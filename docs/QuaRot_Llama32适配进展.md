# QuaRot Llama-3.2-1B 适配进展（2026-10-22）

## 已完成

- 在官方 QuaRot commit `5008669b08c1f11f9b64d52d16fddd47ca754c5a` 上建立 `adapt/llama32-1b-fake-quant` 分支。
- 增加无需 `fast_hadamard_transform` CUDA 扩展的纯 PyTorch 正交 Hadamard 后端。
- 修复上游 `matmul_hadUt_cuda` 调用签名问题。
- 修复 Llama-3.2-1B GQA 的 K-cache token-wise 重排宽度：由 `hidden_size=2048` 改为 `num_key_value_heads * head_dim=512`。
- 修复 Llama-3.2 输入 embedding 与 `lm_head` 共享权重时，RMSNorm 融合会污染输入 embedding 的问题；融合前显式解开共享参数。
- 增加固定模型 revision 和 `local_files_only` 加载参数。
- 6 项测试通过：Hadamard 与上游参考一致、正交/自逆、复合维度 fallback、CUDA BF16、GQA 形状、GQA KV 宽度。

## 等价性验收

统一 WikiText-2 token 流的一窗口结果：BF16 PPL 7.262661，旋转但不量化 PPL 7.260970，差值 -0.001692。8-token prompt 的 top-1 token 完全一致。BF16 舍入下最大 logit 误差 0.46875、平均绝对误差 0.05754。

该结果说明适配后的旋转路径可以继续用于量化实验，但完整 141 窗口的无量化控制仍应在正式结果前补跑。

## RTN 排错消融（不是正式 QuaRot 结论）

| 一窗口方法 | PPL |
|---|---:|
| BF16 | 7.262661 |
| 旋转，无量化 | 7.260970 |
| 旋转 + RTN W4 | 16.948037 |
| 旋转 + RTN W4A4 | 58.922184 |
| 旋转 + RTN W4KV4 | 24.628797 |
| 旋转 + RTN W4A4KV4 | 127.900317 |

这些数值只用于定位误差来源。QuaRot 官方 W4 主路线默认使用 GPTQ；RTN 是可选弱基线。因此不能把 127.90 写成“QuaRot W4A4KV4 正式结果”，也不能据此声称 NestQuant 优于 QuaRot。

## GPTQ 正式对照已经完成

最终采用官方 README 示例的关键设置：GPTQ W4、权重 MSE clipping、per-channel 权重、默认模型 dtype Hadamard、对称 per-token A4/K4/V4、128 条 WikiText-2 train 校准序列。完整结果如下：

| 配置 | PPL |
|---|---:|
| 旋转 BF16 | 9.762884 |
| GPTQ W4 | 10.508346 |
| GPTQ W4KV4 | 12.269130 |
| GPTQ W4A4 | 15.850334 |
| GPTQ W4A4KV4 | 21.095586 |

五项完整实验都使用相同模型 revision、test token SHA256 `54927bf19e6e64b4008eeb93370c370fb25db18cf68eac8015cfe094c464d567`、141 个窗口和 288,627 个预测 token。128 条 GPTQ 校准 token SHA256 为 `358a77b403d372a11957df15dbf46ef3e6f54025d315eaf3514c40bd561d90bb`。

正式解释、与 NestQuant 的范围差异和简历表述边界见 [`../results/llama32_1b_quarot_comparison.md`](../results/llama32_1b_quarot_comparison.md)。RTN 表格只保留为排错证据，不能代表 QuaRot 的正式质量。

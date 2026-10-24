# 上游 E8 语义 NQRT v3 最佳 W4 重建

## 结论

使用修正后的固定上游 E8 边界语义，重新量化并导出了论文式局部 Hadamard + shared128 block-LDLQ 的 Llama-3.2-1B W4 checkpoint。完整 WikiText-2 test 结果为 **PPL 10.076886**，覆盖 141 个 2048-token 窗口和 288,627 个预测 token。

旧 NQRT v2/legacy 语义结果为 10.076729。新结果仅增加 0.000158 PPL（0.001567%），说明 tie-break 修复对格式语义和可互操作性很重要，但对这个完整模型端点的总体质量影响很小。旧结果继续保留用于历史复现，不再作为严格上游 codec 语义的首选数字。

## 固定协议

- 模型：`meta-llama/Llama-3.2-1B`，revision `4e20de362430cd3b72f300e6b0f18e50e7166e08`；
- 数据：WikiText-2 raw v1 test，revision `b08601e04326c79dfdd32d625aee71d232d685c3`；
- test token SHA-256：`54927bf19e6e64b4008eeb93370c370fb25db18cf68eac8015cfe094c464d567`；
- W4：`q=14`，4 个 beta，CPU dither；
- 旋转：112 个支持的 Linear 保存 `W H`，在线执行 `X H`；
- 误差反馈：8-column E8 block-LDLQ，128 个 train 校准窗口，relative damping 0.01；
- 范围：112 个 Linear 量化，tied `lm_head`/embedding 保持 BF16；
- 执行：从 NQMC 逐记录反量化回模型 dtype 后评测，仍不代表 packed INT4 kernel 推理。

## 端点与 checkpoint

| 项目 | 数值 |
|---|---:|
| PPL | **10.0768864881** |
| mean NLL/token | 2.3102443348 |
| predicted tokens | 288,627 |
| checkpoint bytes | 1,044,566,690 |
| checkpoint SHA-256 | `383fd3387414f2a85869851367ab4ecae21260e8b8b9b8ec9c991fe2d65de2bc` |

容器为 NQMC v1，内部量化张量为 NQRT v3。审计得到 146 个唯一 tensor record：112 个量化记录和 34 个 dense 记录；112/112 量化记录均显式声明 `e8_upstream_half_coset_tie_v1`。checkpoint 在烟测和完整评测中均由独立加载路径读取。

## 结论边界

该实验把“固定上游 E8 tie-break 的数值对齐”推进到了真实 1B 模型、完整测试集和自包含 checkpoint。它证明 Python/PyTorch 参考路径的质量与序列化语义闭环，不证明低比特权重常驻显存、INT4 GEMM、吞吐提升或端到端推理加速。A4/KV4 的现有完整矩阵仍基于 legacy checkpoint；要把四档矩阵全部升级为 v3，需在这个新 checkpoint 上重跑 W4KV4、W4A4 和 W4A4KV4。

机器可读摘要见 [`llama32_1b_paper_hadamard_shared128_ldlq_nqrt3.json`](llama32_1b_paper_hadamard_shared128_ldlq_nqrt3.json)。原始 smoke/full manifest 位于 `results/raw/`，大体积 NQMC 不提交 Git。

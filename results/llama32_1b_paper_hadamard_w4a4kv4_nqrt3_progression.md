# NQRT v3 上游 E8 语义 W4/A4/KV4 递进结果

## 结论

同一个采用 `e8_upstream_half_coset_tie_v1` 的 Llama-3.2-1B NQMC checkpoint 上，完整 WikiText-2 test 得到：

| 配置 | PPL | 相对 W4 增量 | mean NLL |
|---|---:|---:|---:|
| W4 | **10.076886** | 0 | 2.310244 |
| W4KV4 | 10.515783 | +0.438897 | 2.352877 |
| W4A4 | 10.689190 | +0.612304 | 2.369233 |
| W4A4KV4 | 11.232054 | +1.155167 | 2.418772 |

A4 的单独质量损失仍大于 KV4。NLL 空间中，A4 增量为 `0.058989`，KV4 增量为 `0.042633`，二者直接相加为 `0.101622`；联合实测增量为 `0.108527`，因此交互项为 **`+0.006906 NLL`**。两项误差总体接近相加，但联合启用存在小幅额外放大。

## 固定协议与身份

- 模型：`meta-llama/Llama-3.2-1B` Base，固定 revision；
- 权重：论文式局部 `W H`/在线 `X H`，shared128 block-LDLQ，112 个 Linear W4，tied `lm_head` BF16；
- codec：NQRT v3 `e8_upstream_half_coset_tie_v1`；
- checkpoint SHA-256：`383fd3387414f2a85869851367ab4ecae21260e8b8b9b8ec9c991fe2d65de2bc`；
- 数据：WikiText-2 test，141 个 2048-token 窗口，288,627 个预测 token；
- test token SHA-256：`54927bf19e6e64b4008eeb93370c370fb25db18cf68eac8015cfe094c464d567`；
- A4：固定上游全局 beta，在 112 个模块的 `local_hadamard_input` 坐标量化；
- KV4：固定逐层 K/V beta，post-RoPE 匹配 Q/K Hadamard，逐 attention head E8 block；
- 四档均独立回载同一 checkpoint，并使用相同 token 顺序、NLL 聚合器和量化范围。

## 量化统计

- W4A4 的 A4 relative squared error 为 `0.638995%`，overload 为 `5.712612%`；
- W4A4KV4 的 A4 relative squared error 为 `0.639200%`，overload 为 `5.711704%`；
- W4KV4 与 W4A4KV4 的 KV4 relative squared error 分别为 `0.618834%` 和 `0.618802%`；
- 单独与联合路径的误差统计接近，说明 A4 hook 与 KV cache 路径组合后没有明显的重复量化或坐标错位。

## 与 legacy NQRT v2 比较

| 配置 | legacy PPL | NQRT v3 PPL | v3 - legacy |
|---|---:|---:|---:|
| W4 | 10.076729 | 10.076886 | +0.000158 |
| W4KV4 | 10.524110 | 10.515783 | -0.008327 |
| W4A4 | 10.688878 | 10.689190 | +0.000312 |
| W4A4KV4 | 11.231160 | 11.232054 | +0.000894 |

三个端点变化极小；W4KV4 的变化稍大但仍只有约 `-0.0791%`。整体上，E8 边界 tie-break 修复主要解决 codec 语义、上游互操作和 checkpoint 可解释性，并未实质改变该模型的总体质量结论。交互项从 legacy 的 `+0.006048` 变为 v3 的 `+0.006906 NLL`，两者都支持“基本可加、存在小幅额外放大”的判断。

## 证据边界

NQMC 是真实序列化的 W4 checkpoint，但评测时权重会反量化到模型 dtype 后执行 BF16 Linear。A4/KV4 在每次前向中量化后立即重建为浮点张量；当前没有 packed activation/KV buffer、INT4/E8 GEMM 或低比特 attention kernel。因此这些结果证明的是相同协议下的质量递进，不能解释为显存下降、吞吐提升或部署延迟。

机器可读摘要见 [`llama32_1b_paper_hadamard_w4a4kv4_nqrt3_progression.json`](llama32_1b_paper_hadamard_w4a4kv4_nqrt3_progression.json)。原始 manifest 保存在 `results/raw/`，不提交体积较大的逐模块统计。

# Llama-3.2-1B NQRT v3 checkpoint inspection

## 结论

新加入的 bounded-memory NQMC inspector 已在 1.04 GB 的真实 Llama-3.2-1B checkpoint 上完成验证。它在不加载模型、不读取 Hessian、不反量化权重的情况下，用约 **11.9 秒**完成 payload checksum、整文件 SHA-256 和 112 个 NQRT record 的语义审计。

| 项目 | 结果 |
|---|---:|
| NQMC 文件大小 | 1,044,566,690 B |
| 文件 SHA-256 | `383fd3387414f2a85869851367ab4ecae21260e8b8b9b8ec9c991fe2d65de2bc` |
| tensor 数（含 alias） | 147 |
| 唯一 record | 146 |
| 量化 record | 112 |
| dense record | 34 |
| tied alias | 1 |
| 量化参数 | 973,078,528 |
| 量化 record 实际码率 | **4.267428 bit/parameter** |
| NQMC / BF16 逻辑载荷 | **42.2623%** |

112/112 个量化 record 均满足：

- NQRT version：3；
- codec semantics：`e8_upstream_half_coset_tie_v1`；
- q：14；
- beta count：4；
- dither：保存 CPU generator state，而不是保存完整 dither tensor；
- model encoder：`block_ldlq`。

## 为什么整体比例不是 25%

`4.267428 bit/parameter` 只针对量化 record，包含 residue、beta index、row norm、generator state 和 record metadata。整个 NQMC 还包含保持浮点的 dense 状态，尤其是 tied embedding/`lm_head`。因此整体文件相对“同一批唯一状态值按 BF16 计算的逻辑载荷”为 42.2623%，不能把 W4 的名义 4 bit 直接写成“整个模型文件缩小到 25%”。

## 证据边界

检查器证明 checkpoint 的格式、身份、记录构成、codec 语义和存储账目可独立审计。它不执行 E8 解码或 GEMM，因此既不证明低比特权重常驻显存，也不证明吞吐、TTFT、TPOT 或端到端延迟改善。

机器可读摘要见 [`llama32_1b_nqrt3_checkpoint_inspection.json`](llama32_1b_nqrt3_checkpoint_inspection.json)，使用方法见[检查器文档](../docs/nqmc_inspection.md)。完整逐项输出保存在被 Git 忽略的 `results/raw/`。

# Llama-3.2-1B PIQA 与 HellaSwag 结果

> 本报告由原始 JSON 自动生成。只有通过协议与逐样本身份校验的结果才会进入表格。

## 汇总指标

| 任务 | 方法 | 样本数 | acc | 相对 BF16 | acc_norm | 相对 BF16 | 耗时（分钟） |
|---|---|---:|---:|---:|---:|---:|---:|
| piqa | bf16 | 1838 | 0.7524 | +0.00 pp | 0.7481 | +0.00 pp | 4.3 |
| piqa | w4 | 1838 | 0.7394 | -1.31 pp | 0.7367 | -1.14 pp | 4.4 |
| piqa | w4a4kv4 | 1838 | 0.7225 | -2.99 pp | 0.7078 | -4.03 pp | 260.6 |
| hellaswag | bf16 | 10042 | 0.4814 | +0.00 pp | 0.6423 | +0.00 pp | 4.3 |
| hellaswag | w4 | 10042 | 0.4707 | -1.07 pp | 0.6282 | -1.41 pp | 4.4 |
| hellaswag | w4a4kv4 | 10042 | 0.4494 | -3.20 pp | 0.5996 | -4.27 pp | 260.6 |

## 相对 BF16 的逐题正确性变化

### w4

- **piqa**
  - `acc,none`：BF16 正确但量化后错误 56；BF16 错误但量化后正确 32；净正确数变化 -24。
  - `acc_norm,none`：BF16 正确但量化后错误 63；BF16 错误但量化后正确 42；净正确数变化 -21。
- **hellaswag**
  - `acc,none`：BF16 正确但量化后错误 200；BF16 错误但量化后正确 93；净正确数变化 -107。
  - `acc_norm,none`：BF16 正确但量化后错误 305；BF16 错误但量化后正确 163；净正确数变化 -142。

### w4a4kv4

- **piqa**
  - `acc,none`：BF16 正确但量化后错误 111；BF16 错误但量化后正确 56；净正确数变化 -55。
  - `acc_norm,none`：BF16 正确但量化后错误 126；BF16 错误但量化后正确 52；净正确数变化 -74。
- **hellaswag**
  - `acc,none`：BF16 正确但量化后错误 435；BF16 错误但量化后正确 114；净正确数变化 -321。
  - `acc_norm,none`：BF16 正确但量化后错误 621；BF16 错误但量化后正确 192；净正确数变化 -429。

## 证据边界

W4 checkpoint is dequantized into BF16 Linear execution; A4/KV4 are fake quant. These results measure task quality, not low-bit kernel speed.

这里的分数用于判断量化质量。它们不构成真实低比特 kernel 的显存、吞吐或延迟证据。

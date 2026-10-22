# Llama-3.2-1B W4 码率与存储开销核算

这份核算解释当前 BF16、RTN W4 和项目 NestQuant W4 的 PPL 对比为什么还不是严格的等码率比较。数据来自已保存的完整评测 manifest 中 Linear 参数/RTN scale 计数，以及固定模型 revision 的配置文件；配置 SHA-256 和完整推导保存在同目录 JSON。这里计算的是**假设采用某种打包格式时的估算值**，当前代码没有输出任何低比特权重文件。

## 输入结构与共享权重

固定模型 revision 的配置给出 hidden size 2048、intermediate size 8192、16 层、32 个 query heads、8 个 KV heads、head dimension 64、vocabulary size 128256，并设置 `tie_word_embeddings=true`。因此 `lm_head.weight` 与输入 embedding 共享同一份参数存储。量化器遍历 `nn.Linear`，处理 `lm_head` 时会同时改写共享的输入 embedding 矩阵；本轮不能表述为“embedding 保持 BF16”。量化前向的其他非 Linear 参数（如 RMSNorm）仍是 BF16。

本轮共遍历 113 个 Linear 模块，参数总数为 1,235,746,816，与保存的 RTN/NestQuant 统计一致；按该 revision 的矩阵形状计算共 505,088 行、154,468,352 个 E8 块。由于 embedding 与 `lm_head` 是别名，这个矩阵只计为一份参数存储。

## 估算结果

| 方法 | 估算组成 | 有效码率 | 量化部分估算体积 |
|---|---|---:|---:|
| RTN W4 | 每权重 4 bit + 每 128 权重一个 FP32 scale | 4.2500 bit/weight | 656,490,496 bytes（约 626.08 MiB） |
| NestQuant W4，普通定宽打包假设 | 每个 q=14 residue 4 bit + 每 8D 块 beta ID 2 bit + 每行 FP32 范数 + 每层 4 个 FP32 beta + 固定运行元数据 | 4.2631 bit/weight | 658,512,668 bytes（约 628.01 MiB） |
| NestQuant W4，理想算术编码情景 | 将 residue 平均按 `log2(14)` bit 编码，其余元数据同上 | 4.0704 bit/weight | 628,755,101 bytes（约 599.63 MiB） |

计算公式：

- RTN：`4P + 32 × scale_count` bit，其中 `P=1,235,746,816`、`scale_count=9,654,272`。
- NestQuant 定宽假设：`4P + 2 × (P/8) + 32 × row_count + 32 × beta_count × layer_count + 96` bit。
- 理想算术编码情景只把第一项换为 `8 × log2(14) × (P/8)` bit。这个情景是理论比较，不是项目已实现的编码器或实测文件大小。

在简单定宽假设下，NestQuant 估算码率比当前 RTN 高约 0.0131 bit/weight，估算体积高约 0.31%。若 residue 真能接近理想熵编码，NestQuant 才显示约 4.22% 的体积优势；但这不能当成已实现收益。两方法的 PPL 为 11.1494 和 13.0215，仍应报告为当前 fake-quant 设置下的质量差异，不能称为等码率优劣结论。

上述估算不包含容器头、对齐填充、索引结构、校验数据和实现相关的码流开销。NestQuant 的 beta 表按每个 Linear 模块存 4 个 FP32 数估计；运行种子用于重建减性 dither。当前 `QuantizedRows.codes` 在内存中是 int64，模型权重在量化后恢复成 BF16；所以这些数字**不证明 checkpoint 文件变小、显存下降或推理加速**。

## 接下来如何把估算变成证据

1. 实现并测试确定的 residue 与 beta-index bit-packing 格式，验证 pack/unpack 往返与异常边界。
2. 用同一套真实存储计量逻辑导出 RTN 和 NestQuant 文件，按实际字节数报告参数码、scale、row norm、beta 表和容器开销。
3. 为公平比较调节两边的分组/元数据方案，使实际总字节数接近，再重复 PPL。
4. 之后才进入低比特 kernel、峰值显存和延迟/吞吐测量。

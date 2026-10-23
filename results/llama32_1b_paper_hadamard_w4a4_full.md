# Llama-3.2-1B 论文式 Hadamard + shared128 LDLQ W4A4 完整评测

## 结论

在完整 WikiText-2 test（288,627 个预测 token）上，论文式局部 `W H` / 在线 `X H`、shared128 block-LDLQ W4 与固定上游全局 beta A4 组合后的 PPL 为 **10.688878**。

- 相对同一 NQMC 的 W4 PPL 10.076729，A4 使 PPL 增加 **0.612150（6.075%）**；
- 相对 BF16 PPL 9.758112，总差距为 **0.930766（9.538%）**；
- 相对旧的普通坐标 shared128 W4A4 PPL 10.952032，局部 Hadamard 路径降低 **0.263154（2.403%）**。该对比同时改变了权重坐标和 A4 坐标，只能作为整体路径对照，不能单独归因于 A4；
- 112 个目标 Linear 全部在 `local_hadamard_input` 坐标量化，完整运行没有 NaN、Inf、shape 或显存错误。

固定全局 beta 建立了可复现基线，但 **5.4411% overload** 表明当前 A4 beta 与 Llama-3.2-1B 各层激活分布并不充分匹配。下一步应先做有界的逐层 beta 校准，再决定是否把 KV4 接入这条论文式路径。

## 固定协议

- 模型：`meta-llama/Llama-3.2-1B` Base，revision `4e20de362430cd3b72f300e6b0f18e50e7166e08`；
- 数据：WikiText-2 test，141 个 2048-token 窗口，288,627 个预测 token；
- 测试 token SHA-256：`54927bf19e6e64b4008eeb93370c370fb25db18cf68eac8015cfe094c464d567`；
- 权重：112 个 Llama Linear 使用论文式局部 Hadamard + shared128 8-column block-LDLQ W4；tied `lm_head` 保持 BF16；
- A4：Q/K/V/O 与 gate/up/down projection 的输入 `X H` 直接执行 Algorithm-3 E8 fake quant；
- A beta 系数：固定上游 `parallel/ppl.py` 默认 `[3.47, 4.74, 6.90, 18.11]`，`q=14`；
- A beta 值：`[0.247857, 0.338571, 0.492857, 1.293571]`；
- 评测 batch size：4；运行设备：RTX 5070 / WSL2 / BF16 compute。

## 结果

| 路径 | PPL | 相对 BF16 增量 | 相对 W4 增量 |
|---|---:|---:|---:|
| BF16 | 9.758112 | — | — |
| Hadamard + shared128 LDLQ W4 | 10.076729 | +0.318617 | — |
| Hadamard + shared128 LDLQ W4A4 | **10.688878** | +0.930766 | +0.612150（+6.075%） |

本次评测的总 NLL 为 683,816.179014，平均 token NLL 为 2.369203779。峰值 CUDA allocated 为 7,205,217,280 B，reserved 为 11,563,696,128 B。完整评测耗时 1,819.6 秒；这是质量仿真运行耗时，不是低比特推理延迟。

## 激活量化统计

| 指标 | 数值 |
|---|---:|
| 量化模块数 | 112 |
| 坐标 | `local_hadamard_input` |
| 激活 entry 数 | 94,623,498,240 |
| 激活向量数 | 32,342,016 |
| 相对平方误差 | 0.006390378（0.6390%） |
| overload 比例 | 0.054411248（5.4411%） |
| 分析码率 | 4.2609375 bit/entry |

分析码率包含 residue、每个 E8 block 的 beta 索引和每个向量的浮点行范数。运行时激活仍重构为浮点张量，因此 4.26094 bit/entry 是格式估算，不能当作实际激活显存占用或 kernel 吞吐。

## 证据与边界

- W4 检查点：`results/raw/llama32_1b_paper_hadamard_w4_shared128_ldlq.nqmc`；
- 检查点大小：1,044,561,090 B；
- 检查点 SHA-256：`f594a20cd2f2abfed88fc1bdeabe8efb7a71e4bef5509933a74ab327a841f821`；
- 原始运行清单：`results/raw/llama32_1b_paper_hadamard_w4a4_shared128_ldlq_full.json`；
- run id：`20261023T102632184339Z`；
- 代码 commit：`7aff0c0ca5062e3ae0aa57063feceb0cebdd32a4`；运行时工作树干净。

本结果证明同一 W4 检查点上的完整 A4 质量仿真和 PPL，不证明低比特激活常驻显存、INT4/E8 GEMM、延迟下降或吞吐提升。模型是 Llama-3.2-1B，本地结果不能直接替代论文 Llama-3-8B 主表。

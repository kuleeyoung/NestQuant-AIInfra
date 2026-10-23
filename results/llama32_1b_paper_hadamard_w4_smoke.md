# Llama-3.2-1B 论文式 Hadamard W4/NQMC smoke

本结果验证论文 §4.3 局部权重—激活匹配旋转的工程闭环。112 个 Llama Linear 把旋转后的 `W H` 以原模型 BF16 dtype 保存，前向在线计算 `X H`；与输入 embedding 共享权重的 `lm_head` 保持 BF16。NestQuant 直接量化旋转坐标权重，不再把误差逆旋转回普通 Linear。

固定同一个 WikiText-2 test 首窗口，只评测 2,047 个预测 token：

| 路径 | smoke PPL |
|---|---:|
| BF16 | 7.262661 |
| 普通 NestQuant W4，保留 `lm_head` | 8.194570 |
| 论文式 Hadamard NestQuant W4，内存量化 | 8.064950 |
| 论文式 W4 → NQMC → 立即回载 | 8.064950 |
| 新进程独立加载论文式 NQMC | 8.064950 |

论文式路径的 NQMC 文件为 1,044,559,427 字节，SHA-256 为 `942109f1475defebf8e6eee2d8f74d5abefa525e292e7f3a662e470ebb7e7da3`。文件保存版本化旋转格式、112 个转换模块、tied 跳过项、权重存储 dtype 和全部 packed W4 记录。独立进程只给出 NQMC 和基础模型即可恢复在线旋转结构，回载 PPL 与内存量化逐值一致。

导出端已从全量 `bytes` 拼接改为逐记录临时 payload 和最终原子替换；回载端使用分块 checksum 和逐记录反量化。两条路径都不要求在主机内存中额外保留完整 NQMC 字节副本。

该实验只证明真实模型执行、量化范围和检查点语义闭环。首窗口中论文式 W4 比普通 W4 低 0.129620 PPL，但 2,047 token 本身不足以支持质量提升结论。后续完整 288,627-token 同协议实验已确认普通 W4 为 11.054059、论文式 W4 为 10.917094，见[完整报告](llama32_1b_paper_hadamard_w4_full.md)。当前仍没有 A4 合并、低比特常驻或 kernel 加速证据。

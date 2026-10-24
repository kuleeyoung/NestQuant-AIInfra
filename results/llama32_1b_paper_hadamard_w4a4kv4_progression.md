# 局部 Hadamard + shared128 LDLQ 的 W4/A4/KV4 递进结果

## 结论

同一个 Llama-3.2-1B NQMC 检查点上，W4、W4KV4、W4A4、W4A4KV4 的完整 WikiText-2 test PPL 分别为：

| 配置 | PPL | 相对 W4 增量 | mean NLL |
|---|---:|---:|---:|
| W4 | **10.076729** | 0 | 2.310229 |
| W4KV4 | 10.524110 | +0.447381 | 2.353669 |
| W4A4 | 10.688878 | +0.612150 | 2.369204 |
| W4A4KV4 | 11.231160 | +1.154431 | 2.418692 |

A4 的单独质量损失大于 KV4。若两者在 NLL 空间完全相加，组合增量应为 `0.058975 + 0.043440 = 0.102415`；实测为 `0.108463`，交互项为 `+0.006048 NLL`。因此 A4 与 KV4 的误差总体接近相加，但存在小幅额外放大。

## 固定协议

- 模型：`meta-llama/Llama-3.2-1B` Base，固定 revision；
- 权重：论文式局部 `W H` / 在线 `X H`，shared128 block-LDLQ，112 个 Linear W4，`lm_head` BF16；
- checkpoint SHA-256：`f594a20cd2f2abfed88fc1bdeabe8efb7a71e4bef5509933a74ab327a841f821`；
- 数据：WikiText-2 test，141 个 2048-token 窗口，288,627 个预测 token；
- test token SHA-256：`54927bf19e6e64b4008eeb93370c370fb25db18cf68eac8015cfe094c464d567`；
- A4：固定上游全局 beta；此前独立 train held-out 已拒绝三个更差的校准 profile；
- KV4：固定上游 K/V beta，post-RoPE 匹配 Q/K Hadamard，逐 attention head E8 block；
- 四档均使用同一 checkpoint、token 顺序、NLL 聚合器和质量范围。

## 量化统计

- W4A4KV4 的 A4 覆盖 112 个模块，全部位于 `local_hadamard_input` 坐标；相对平方误差 `0.639043%`，overload `5.439937%`。
- KV4 覆盖 16 层。W4KV4 与 W4A4KV4 的 K/V 合计相对平方误差分别为 `0.619033%` 和 `0.619020%`，说明加入 A4 没有改变 KV4 编解码统计口径。
- W4A4 与 W4A4KV4 的 A4 误差统计几乎一致，说明两条 hook/cache 路径可以组合，没有发生重复量化或坐标错位的明显迹象。

## 原始证据

| 配置 | 原始 manifest SHA-256 |
|---|---|
| W4 | `ac9cbb6105026b1f42ab11c6cd6cc1406e9befb0b571eee26298872f9c97aa59` |
| W4A4 | `76fae64d2c0c2c0cbb32e90c850c084e84732836c2695383137811e810c420fa` |
| W4KV4 | `8fa3975b43e4d3d9035ed06c68d1d2554e4639c56be9bb290845d9f9336cf71a` |
| W4A4KV4 | `e49f976529c8d361b7d959a96e99b9cbc985dbbd167e6a609c1983c4051291b7` |

原始 JSON 保存在 `results/raw/`，不提交大体积逐层统计。机器可读核验摘要见同名 JSON。

## 证据边界

NQMC 是真实 packed W4 文件，但评测时先反量化到 BF16 Linear。A4/KV4 在每次前向中量化后立即重构成浮点张量；当前没有 packed activation/KV buffer、INT4/E8 GEMM 或低比特 attention kernel。因此这些数字证明的是相同协议下的质量递进，不能解释为显存下降、吞吐提升或部署延迟。

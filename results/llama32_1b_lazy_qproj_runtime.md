# Llama-3.2-1B 单层 Lazy NQMC Runtime 实验

## 结论

本阶段把 NQMC 的“可选择读取并分块解码单个张量”推进成了可执行的单层推理原型：`LazyNQMCLinear` 在 forward 时读取并解码指定权重，然后执行在线 Hadamard 和 PyTorch BF16 `Linear`。它支持两种互补策略：

- `none`：每次 forward 解码，调用完成后不保留 BF16 权重，适合用时间换驻留显存。
- `persistent`：第一次 forward 解码，此后复用 BF16 权重，适合热点层用显存换延迟。

这仍然不是 packed low-bit kernel。矩阵乘法操作数是解码后的 BF16 权重，因此不能据此宣称 4-bit GEMM、端到端吞吐提升或模型全量部署完成。

## 固定实验条件

| 项目 | 值 |
|---|---:|
| checkpoint | `llama32_1b_paper_hadamard_w4_shared128_ldlq_nqrt3.nqmc` |
| checkpoint SHA-256 | `383fd3387414f2a85869851367ab4ecae21260e8b8b9b8ec9c991fe2d65de2bc` |
| 权重 | `model.layers.0.self_attn.q_proj.weight` |
| 权重形状 | `[2048, 2048]` |
| 输入形状 | `[128, 2048]` |
| 设备 | RTX 5070 / WSL2 |
| PyTorch | `2.14.1+cu130` |
| 计算 dtype | BF16 |
| 在线坐标变换 | normalized Hadamard |
| 解码 chunk | 128 rows |

两种策略都在独立进程运行。首次 forward 只测一次；`none` 的后续统计为 10 次，`persistent` 为 30 次。计时覆盖 NQMC 文件读取、unpack/dequantization、在线 Hadamard 和 BF16 GEMM。CUDA context 初始化不计入。为核对 bitwise correctness，计时前参考路径已经读取同一条记录，因此这里的“首次 forward”仅表示模块内部尚未建立 persistent cache，**不是**操作系统 page cache 完全冷启动。

## 结果

| 策略 | 首次 forward | 后续均值 | 后续中位数 | 后续 CUDA peak 增量 | forward 后常驻解码权重 |
|---|---:|---:|---:|---:|---:|
| `none` | 122.04 ms | 102.08 ms | 101.71 ms | 22.28 MiB | 0 MiB |
| `persistent` | 118.68 ms | 1.19 ms | 1.15 ms | 1.50 MiB | 8 MiB |

`persistent` 的 steady-state 均值比每次重新解码快约 **85.7 倍**。这个倍数表示“避免重复解码”的收益，不表示矩阵乘法本身加速了 85.7 倍。两者首次 forward 同为约 0.12 秒，说明第一次都必须支付相同量级的记录读取和解码成本；它不能被用于推断冷盘 I/O 延迟。

steady-state CUDA peak 增量从 22.28 MiB 降到 1.50 MiB，是因为 `persistent` 的 8 MiB BF16 权重已经在测量基线中常驻；这 8 MiB 必须单独计入，不能把 1.50 MiB 解读成该层的全部显存占用。

## 正确性

参考路径先对同一记录执行非分块、完整 BF16 解码，再执行相同的在线 Hadamard 与 `F.linear`。两种缓存策略的输出都与参考结果逐 bit 相等：

- `torch.equal = True`
- 最大绝对误差：`0.0`
- 输出 checksum：`-550.4730224609375`

项目全量测试从 140 项增加到 144 项，新增测试覆盖：无缓存等价性、持久缓存复用与释放、在线 Hadamard 等价性，以及 dense 记录/非法参数/输入维度拒绝；144 项全部通过。

## 工程解释

这个实验给出了实际的显存—延迟策略，而不是只实现一个反序列化 API：

1. 对只访问一次或低频访问的层，可使用 `none`，避免整模型 BF16 权重长期驻留，但每次调用约有 100 ms 解码成本。
2. 对重复访问的热点层，可使用 `persistent`，首次调用后把该层变成普通 BF16 Linear；单个 q_proj 常驻 8 MiB。
3. 若把所有层都设为 `persistent`，最终会逐步恢复接近完整 BF16 权重的显存占用，所以它不是最终低比特部署方案。
4. 真正的下一阶段是模型级 cache/offload 调度（例如容量受限的 LRU），再往后才是避免 BF16 materialization 的 fused unpack/dequant + GEMM kernel。

原始机器结果位于 `results/raw/llama32_1b_lazy_qproj_none.json` 和 `results/raw/llama32_1b_lazy_qproj_persistent.json`。

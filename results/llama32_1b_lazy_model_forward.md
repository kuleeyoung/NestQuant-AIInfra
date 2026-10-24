# Llama-3.2-1B 真实 Lazy NQMC 模型 Forward

## 本阶段完成了什么

本阶段不再由基准脚本手动调用少数 q_proj，而是从原始 Hugging Face Llama 结构出发，读取 NQMC manifest 和 Hadamard metadata，自动完成：

1. 校验 checkpoint state names 与模型 state dict 完全一致；
2. 校验所有量化记录都对应二维 Linear weight，shape 一致；
3. 校验 Hadamard converted names 与量化模块集合一致；
4. 按记录加载 embedding、norm、bias 等 34 个 dense tensors；
5. 保留 `lm_head.weight → model.embed_tokens.weight` tied alias 语义；
6. 把 112 个 attention/MLP projection 替换为共享 LRU 的 `LazyNQMCLinear`；
7. 将仅剩的 dense state 移到 GPU，在真实 `LlamaForCausalLM.forward` 中按需解码权重。

所有结构和运行参数在首次模型修改前验证。转换失败不会留下部分替换的模块。

## 实验设置

| 项目 | 值 |
|---|---:|
| 模型 | `meta-llama/Llama-3.2-1B` |
| revision | `4e20de362430cd3b72f300e6b0f18e50e7166e08` |
| checkpoint | paper-Hadamard shared128 LDLQ NQRT v3 |
| 输入 | `[1, 8]` token IDs |
| lazy cache | 256 MiB |
| decode chunk | 128 rows |
| 设备 | RTX 5070 / WSL2 |
| PyTorch | `2.14.1+cu130` |

基线先完成一次不计入正式结果的 forward warm-up，随后计时，避免把 CUDA 算子首次初始化归入基线。lazy forward 运行两次，用于观察第二次是否能从 LRU 获益。两条路径分时驻留 GPU，没有同时保留两套模型。

## 正确性

完整回载基线会把所有 packed records 解码进模型 BF16 参数；lazy 路径仅在模块执行时解码。两者输出 logits：

- `torch.equal = True`
- 最大绝对误差：`0.0`
- 平均绝对误差：`0.0`
- checksum 均为 `3386097.5`

这证明自动模块替换、dense state 加载、tied weight、在线 Hadamard、分块解码和真实 Transformer 数据流组合后仍保持逐 bit 等价。

## 加载与显存

| 指标 | 完整 BF16 回载 | Lazy 256 MiB LRU | 变化 |
|---|---:|---:|---:|
| load/convert/move | 31.68 s | 2.11 s | lazy 快 14.99× |
| CUDA peak allocated | 2,508.36 MB | 895.10 MB | 降低 64.32% |
| forward 后 allocated | 2,508.33 MB | 803.35 MB | 降低 67.97% |

这里的 load 加速来自跳过 112 个权重的启动时完整解码，而不是磁盘格式读取更快。lazy 的 256 MiB 只限制 LRU 中保留的解码权重；895 MB peak 还包含 dense model state、activation、输出和 miss 解码临时张量。

## Forward 与缓存行为

| 路径 | forward 1 | forward 2 | 每轮 hit | 每轮 miss/decode |
|---|---:|---:|---:|---:|
| 完整回载基线 | 52.61 ms | — | 不适用 | 不适用 |
| Lazy LRU | 23.70 s | 24.23 s | 0 | 112 |

lazy 平均为 23.96 s，是 warm baseline 的约 455.5 倍。第一轮产生 98 次淘汰；第二轮产生 112 次淘汰。第二轮仍然 0 hit，说明不是首次启动现象。

原因是标准 dense Transformer 的权重访问接近顺序扫描：

```text
layer 0 的 7 个 projection
→ layer 1 的 7 个 projection
→ ...
→ layer 15 的 7 个 projection
```

256 MiB 只能保留最后约 14 个不同尺寸的 projection。当下一轮重新从 layer 0 开始时，加载早期权重会不断淘汰上一轮末尾的权重；等执行再次走到 layer 15 时，这些末尾权重也已经被淘汰。因此通用 LRU 出现顺序扫描型 cache pollution，不能利用跨 forward 复用。

## 结论与设计转向

这次结果同时验证了两个方向：

1. **成立的方向**：packed checkpoint + 自动 lazy 模块替换确实能在真实模型上把 CUDA allocated peak 降低约 64%，而且 logits 逐 bit 一致。
2. **不成立的方向**：对密集 Transformer 使用远小于全模型工作集的普通 LRU，不会获得推理加速；当前实现反而因每轮解码 112 个权重慢约 456 倍。

因此下一步不应继续微调 LRU 淘汰规则，而应转向和执行顺序配合的方案：

- layer-window streaming：一个 Transformer block 执行完成后立即释放其权重；
- CPU/NVMe packed record 预取；
- 双 CUDA stream，把下一层 decode 与当前层计算重叠；
- 最终用 fused unpack/dequant + GEMM 避免完整 BF16 weight materialization。

在当前 Python/PyTorch 解码速度下，即使预取可以隐藏一部分 I/O，也无法隐藏每层大量 Python/Torch unpack 工作。因此真正解决约 456 倍延迟差距的关键仍是 compiled CUDA decode 或 fused kernel，而不是更复杂的 Python cache policy。

## 证据边界

这是一次真实 Llama forward，不再只是单层或合成访问序列。但每个 miss 仍会生成 BF16 权重，矩阵乘法仍是普通 PyTorch BF16 GEMM。结果不能表述为 4-bit kernel、INT4 Tensor Core 推理、生产吞吐提升或完整 serving runtime。

原始机器输出位于 `results/raw/llama32_1b_lazy_model_forward.json`。

# NestQuant A4 与 W4A4KV4 质量仿真

本项目已在 Llama-3.2-1B Base 上完成 A4 与 W4A4KV4 路径。实现对齐作者公开代码的 hook 范围：每层对 Q/K/V/O 和 gate/up/down 七类 Linear 的输入做量化，不量化最终 `lm_head` 输入。

每个激活向量按最后一维独立处理：先做归一化 Hadamard，随后使用 Algorithm 3 的行范数归一化、8 维 E8 分块和 Opt-beta，反量化后再做逆 Hadamard。公开默认 activation beta 系数为 `[3.47, 4.74, 6.90, 18.11]`，实际 beta 是这些系数除以 `q=14`。

Q/K/V 共享同一个输入，gate/up 也共享同一个输入。实现用源 Tensor 的弱引用在一次 forward 生命周期内复用确定性的量化重构，使每层实际执行 4 次而不是 7 次 E8 量化；统计仍分别覆盖七类投影。单元测试验证了复用次数、模块范围和输出形状。

## 完整结果

为了提高 RTX 5070 利用率，完整实验把 4 个互相独立的 2048-token 窗口合并到一个 forward。每个 batch 创建新 cache，PPL 仍由全测试集 288,627 个预测 token 的总 NLL 计算。三组运行的模型、数据、token、Python 源码和 NQMC 文件哈希相同。

| 设置 | PPL | 相对 W4 | 相对 W4KV4 |
|---|---:|---:|---:|
| W4 LDLQ | 10.276601 | 0.000% | -4.243% |
| W4KV4 | 10.732014 | +4.432% | 0.000% |
| W4A4KV4 | 11.578171 | +12.665% | +7.884% |

完整 manifest 哈希、峰值显存和证据边界见 [`results/llama32_1b_w4a4kv4_progression.md`](../results/llama32_1b_w4a4kv4_progression.md)。当前配置中，A4 是比 KV4 更主要的新增 PPL 损失来源，因此后续 QA-LDLQ 应优先针对 activation quantization 的误差传播进行优化。

## 批处理与硬件

batch=4 运行时观察到 GPU 利用率约 98%、`nvidia-smi` 显存使用约 10.75 GiB；manifest 中 PyTorch 峰值 allocated 为 6.835 GiB。两者口径不同：前者包含进程与 CUDA 上下文等总占用，后者只统计 PyTorch allocator。batch=4 已接近当前 11.91 GiB 可用显存的合理上限。

## 证据边界

A4 在每个 Linear 前把激活重构为 BF16，KV4 也把 cache 内容重构为浮点张量。它们证明的是质量变化，没有使用低比特 GEMM、packed KV cache 或低比特 attention kernel。参考实现的 Python 耗时不能解释为部署性能，PyTorch allocated 也不能解释为真实 A4/KV4 显存收益。

## 复现命令

```powershell
python scripts/evaluate_llama_wikitext_ppl.py --methods nestquant_w4 --load-checkpoint results/raw/llama32_1b_nestquant_w4_ldlq_train64.nqmc --a4 --kv4 --kv-beta-config configs/kv_betas_upstream_defaults_q14.json --eval-batch-size 4 --offline --output results/raw/llama32_1b_w4a4kv4_full.json
python scripts/summarize_w4a4kv4.py
```

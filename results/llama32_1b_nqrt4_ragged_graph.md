# Llama-3.2-1B NQRT4 Ragged CUDA Graph 正确性验证

## 目标

验证同一固定 batch CUDA Graph 中，不同请求可以使用不同 KV cache position，并在一个旧请求
继续生成时释放另一 slot、清除旧 KV、接入不同 prompt 长度的新请求。

## 结果

设备为 NVIDIA GeForce RTX 5070，模型为完整 Llama-3.2-1B packed NQRT4 runtime。

| 场景 | 两个请求的位置 | Graph vs ragged eager 最大误差 | 相对独立完整前向最大误差 | argmax |
|---|---|---|---|---|
| 暂停一个请求后恢复 | 17 / 16 | 0.0 / 0.0 | 0.1875 / 0.1875 | 全部一致 |
| 释放旧 slot 并接入新 prompt | 17 / 3 | 0.0 / 0.0 | 0.1875 / 0.43359375 | 全部一致 |

Graph 与相同缓存状态的 eager 路径逐元素一致。相对完整序列前向的非零 BF16 logits 差异同时
出现在 eager cache 路径，因此不归因于 Graph；验收同时要求最终 token argmax 一致。

## 实现要点

- KV cache 保持固定地址，逐 batch row 使用独立二维 position；
- 使用 `scatter_` 原地写入，避免布尔索引产生动态大小张量；
- 四维 attention mask 按每个请求的有效长度屏蔽旧 slot 残留区域；
- slot 回收时原地清零该 batch row，不替换 cache 张量；
- Graph 捕获对样本 token/position/mask/length 做私有 clone，防止静态输入缓冲区覆盖调用者张量。

## 复现

```bash
python scripts/validate_ragged_graph_decode.py \
  results/raw/llama32_1b_paper_hadamard_w4_shared128_ldlq_nqrt4.nqmc \
  --offline \
  --output results/raw/ragged_graph_decode_validation.json
```

## 证据边界

当前是两个请求、两种确定性状态转换的正确性验证。它还不是完整 continuous-batching engine，
尚未覆盖随机到达、采样、EOS 实际模型输出、batch 1/2/4/8 ragged Pool、长时间 soak test、
tail latency 或网络接口。

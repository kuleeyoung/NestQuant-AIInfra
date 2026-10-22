# Llama-3.2-1B W4 → W4KV4 → W4A4KV4

三组完整 WikiText-2 test 运行均使用 batch=4，并通过模型、数据、token、源码与 NQMC 哈希一致性检查。

| 设置 | PPL | 相对 W4 | 相对 W4KV4 | 峰值 CUDA allocated |
|---|---:|---:|---:|---:|
| W4 LDLQ | 10.276601 | +0.000% | -4.243% | 4.664 GiB |
| W4KV4 | 10.732014 | +4.432% | +0.000% | 4.664 GiB |
| W4A4KV4 | 11.578171 | +12.665% | +7.884% | 6.835 GiB |

## 解释

- KV4 在 W4 上增加约 4.43% PPL；继续加入 A4 后，相对 W4KV4 再增加约 7.88%。当前配置中 A4 是更主要的新增质量损失来源。
- A4 覆盖每层 q/k/v/o 与 gate/up/down 七类投影输入，使用作者公开默认 activation beta；KV4 使用作者公开默认 key/value beta。
- batch=4 将本机 RTX 5070 利用率提高到接近满载，但不会形成推理加速证据，因为 A4/KV4 都在 Python/PyTorch 中重构为浮点张量。

## 证据边界

权重来自真实 packed NQMC 文件，但载入后以 BF16 执行；A4 和 KV4 都是质量仿真，没有 packed activation/KV buffer 或低比特 GEMM/attention kernel。因此这里只能陈述 PPL 与参考实现运行资源，不能陈述真实低比特延迟、吞吐或显存收益。

## 可复核哈希

- token：`54927bf19e6e64b4008eeb93370c370fb25db18cf68eac8015cfe094c464d567`
- NQMC：`5cd2b02c892497e4345c5ca5dbefc2d20fccd2038044356e65e517eca334f199`
- Python 源码：`1c5015d1c5ed8aebb6d229385b3aa1db15226841b7910ce77e155f5e0e4ff47f`

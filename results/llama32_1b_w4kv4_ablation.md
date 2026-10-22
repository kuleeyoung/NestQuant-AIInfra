# Llama-3.2-1B W4KV4 消融实验

四个完整运行已经通过模型、数据、token、代码和 NQMC 文件哈希一致性检查。

| 设置 | PPL | 相对 W4 | KV 相对平方误差 | KV 过载率 | 估算 bit/entry |
|---|---:|---:|---:|---:|---:|
| W4 LDLQ + floating DynamicCache | 10.276133 | +0.000% | — | — | — |
| W4 LDLQ + matched Q/K rotation | 10.280920 | +0.047% | — | — | — |
| W4KV4 + public-upstream default betas | 10.712355 | +4.245% | 0.619009% | 5.389879% | 4.75 |
| W4KV4 + local six-chunk beta DP | 11.333720 | +10.292% | 1.299673% | 0.239980% | 4.75 |

## 结论

- 匹配 Q/K Hadamard 旋转相对普通 DynamicCache 的 PPL 变化为 +0.047%，因此缓存适配与旋转本身只产生很小的 BF16 数值漂移。
- 公开实现默认 beta 的 W4KV4 PPL 相对 W4 增加 4.245%。
- 本地六段校准 DP 的 W4KV4 PPL 相对 W4 增加 10.292%；公开默认 beta 比它低 0.621365 PPL。
- 该反例说明在六段训练校准样本上按 E8 块重构误差选出的 beta 不保证测试集重构误差或端到端 PPL 最优；校准目标、样本规模与跨层敏感度仍需研究。

## 证据边界

当前实现把量化后的 K/V 重构为浮点张量再参与注意力计算，因此证明的是质量仿真。4.75 bit/entry 是包含 4-bit residue、每 8 维 2-bit beta index 和每 64 维 32-bit 行范数的理论估算。尚未分配真实 packed KV cache，也没有低比特 attention kernel，所以不能据此声称显存降低或推理加速。

公开默认 beta 来自固定上游提交的 `parallel/ppl.py`，系数除以 `q` 的行为由 `coder.py` 核对。公开仓库未提供其 beta DP 脚本，本地六段校准属于项目自研消融，不应描述为作者原始校准流程。

## 可复核信息

- 模型 revision：`4e20de362430cd3b72f300e6b0f18e50e7166e08`
- 数据 revision：`b08601e04326c79dfdd32d625aee71d232d685c3`
- token SHA-256：`54927bf19e6e64b4008eeb93370c370fb25db18cf68eac8015cfe094c464d567`
- NQMC SHA-256：`5cd2b02c892497e4345c5ca5dbefc2d20fccd2038044356e65e517eca334f199`
- 当前 Python 源码 SHA-256：`73de39ec2a6a7580f012cf0807b0ddc065b03f413fb36f503bbe7df39f4e1bd3`

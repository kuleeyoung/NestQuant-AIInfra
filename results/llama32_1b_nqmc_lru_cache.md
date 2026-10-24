# Llama-3.2-1B NQMC 解码权重 LRU 实验

## 实现目标

单层 `none` 与 `persistent` 只能表达两个极端：每次重新解码，或者让所有访问过的 BF16 权重永久驻留。本阶段新增共享的 `LazyNQMCWeightCache`，让多个 `LazyNQMCLinear` 在明确的常驻字节预算内复用解码权重，并按 least recently used 顺序淘汰。

缓存 key 是 `(canonical weight name, device, compute dtype)`。别名会归一到同一个 checkpoint 记录，而不同设备或 dtype 不会错误共享张量。缓存公开 `hits`、`misses`、`decodes`、`evictions`、`bypasses`、当前/峰值常驻字节和命中率。单个权重大于总预算时仍能执行，但会 bypass，避免一次超大记录破坏预算语义。

## 实验条件

| 项目 | 值 |
|---|---:|
| checkpoint | `llama32_1b_paper_hadamard_w4_shared128_ldlq_nqrt3.nqmc` |
| checkpoint SHA-256 | `383fd3387414f2a85869851367ab4ecae21260e8b8b9b8ec9c991fe2d65de2bc` |
| 权重 | 第 0、1、2 层 `self_attn.q_proj.weight` |
| 每个解码权重 | 8 MiB BF16 |
| 缓存预算 | 16 MiB，即两个权重 |
| 输入 | `[128, 2048]` BF16 |
| 解码 chunk | 128 rows |
| 在线变换 | normalized Hadamard |
| 环境 | WSL2 / RTX 5070 / PyTorch 2.14.1+cu130 |

计时前执行一次完整路径 warm-up，然后清空缓存与统计，排除 CUDA 算子首次初始化。实验测的是稳定运行期的顺序访问；不包含真实 Transformer 层间计算，也没有尝试模拟磁盘冷缓存。

## 结果

| 访问模式 | 序列 | hit / miss | decode | eviction | 命中率 | hit 均值 | miss 均值 | 峰值常驻 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 工作集适配预算 | A,B,A,B,A,B | 4 / 2 | 2 | 0 | 66.67% | 0.600 ms | 113.29 ms | 16 MiB |
| 工作集超过预算 | A,B,C,A,B,C | 0 / 6 | 6 | 4 | 0% | — | 99.16 ms | 16 MiB |

第一组在 A、B 首次解码后全部命中，说明热点工作集能被预算覆盖时，LRU 会把后续调用变成在线 Hadamard 加普通 BF16 GEMM。第二组的三个权重循环访问，但预算只能容纳两个；每次需要的权重恰好刚被淘汰，因此发生典型 cache thrashing，LRU 无法带来收益。

这个结果说明“有缓存”本身不等于“会加速”。性能取决于：

1. 解码权重工作集是否能放进预算；
2. 访问序列是否具有时间局部性；
3. miss 的解码代价与 hit 的计算代价之比；
4. 是否能预取或把解码与其他计算重叠。

## 正确性与预算语义

所有实际访问的 q_proj 输出都与“完整非分块 BF16 解码 + 相同 Hadamard + 相同 `F.linear`”逐 bit 相等，最大绝对误差为 0。

两组实验的缓存当前与峰值常驻量都没有超过 16 MiB。但这个预算只约束**保留在 LRU 中的 BF16 权重**，不约束：

- 当前 miss 的 unpack/dequant 临时张量；
- activation 和输出；
- CUDA allocator reserve；
- 并发 forward 中仍被调用方引用、但已经从 LRU 账本淘汰的张量。

真实 miss 的 CUDA peak 增量仍约为 22.28 MiB。因此不能把“16 MiB LRU 预算”描述成“整个运行时只用 16 MiB 显存”。当前锁可以防止同一进程中重复并发加载，但尚未实现带 lease/reference count 的严格并发物理显存上限。

## 工程边界和下一步

这是模型级权重缓存策略原型，但还不是完整模型替换：访问序列由基准脚本显式驱动，GEMM 仍消费物化后的 BF16 权重。下一步应把 Transformer 中选定的 `HadamardRotatedLinear` 自动替换成共享 LRU 的 `LazyNQMCLinear`，运行一个小规模真实 forward，验证模块名、bias、设备、顺序和输出 logits。随后才适合加入预取/双缓冲，再进一步开发 fused unpack/dequant + GEMM kernel。

原始机器结果保存在 `results/raw/llama32_1b_lru_locality.json` 与 `results/raw/llama32_1b_lru_thrash.json`。

# 2026-10-29 夜间推进报告

最终审计时间：2026-10-29 13:36（Asia/Shanghai）。原定 08:30 的可见汇报未被
后台持续目标自动触发，本报告按实际完成和审计时间记录，不伪造准时状态。

## 结论

本轮完成了直接 packed W4A4 CUDA M16 内核的指令级正确性闭环、完整
Llama-3.2-1B 接入、性能复测、运行时分派审计、全量测试和发布包验证。
本地简历项目完成度由约 95% 提升到 **97%**；剩余工作主要是需要用户选择的
下一条内核路线，以及用户确认后的 Git 提交/远程发布，不再是基础功能缺失。

## 今晚新增成果

### 1. 修复 A4 解码边界错误

- 定位到 4,096 个真实激活 block 中 3 个 block 的不一致；
- 根因不是量化公式错误，而是两个数学等价写法在 FP32 中运算顺序不同，
  在 E8 半陪集边界产生最后一位舍入差异；
- CUDA 改为严格复现 PyTorch 的 `(x - 0.5) - round(x - 0.5)` 运算顺序；
- CUDA 激活解码与 Python/upstream、Triton 对 4,096 个 block **逐值一致**。

FP32 non-associativity（浮点非结合性）指 `(a+b)+c` 与 `a+(b+c)` 在有限精度
浮点数中不保证得到相同末位；在离散量化边界上，末位差异可能改变码字选择。

### 2. 完成 4-lane cooperative E8 解码

- 每 4 个 CUDA lane 协作恢复一个 8 维 E8 block；
- 使用 subgroup shuffle 在 lane 间完成奇偶校验、最大误差坐标选择和归约；
- 避免原实现中每个 lane 重复执行完整 8 维解码。

lane 是 CUDA 线程束中的单个执行线程；cooperative（协作式）表示多个线程共同
处理一个逻辑数据块，而不是各自重复全部工作。

### 3. 完整 1B M16 性能与分派审计

测试范围：Llama-3.2-1B 全部 16 层、112 个 packed 投影，batch 1、sequence 16、
3 次独立进程试验，每轮 7 次稳态重复。下表为三轮中位数的中位数。

| 后端 | prefill 中位延迟 | TTFT | TPOT |
|---|---:|---:|---:|
| tiled Triton W4A4 | 178.93 ms | 180.12 ms | 163.27 ms |
| CUDA MMA M16 | 117.17 ms | 115.77 ms | 164.62 ms |

- prefill 加速中位数：**1.527x**（1.518～1.544x）；
- TTFT 中位数降低：**35.7%**；
- 每轮自定义 CUDA M16 内核实际命中：**1,232 次**；
- 每轮 M1 decode 自动回退 tiled Triton：**448 次**；
- 两条路径 packed 权重显存相同：518,457,088 bytes。

TTFT（Time To First Token）是从请求开始到第一个 token 可用的时间；TPOT
（Time Per Output Token）是稳态生成每个后续 token 的平均时间。

实际编译扩展另经 `cuobjdump --dump-sass` 验证，包含
`HMMA.16816.F32.BF16` Tensor Core 指令；二进制 SHA-256 和匹配 SASS 已保存到
`results/raw/w4a4_cuda_mma_binary_audit.json`。SASS 是 GPU 最终执行的机器指令，
因此该证据强于只检查 CUDA 源码中的内联 PTX。

### 4. 精度边界审计

- 112/112 个真实投影均执行自定义内核；
- 投影级最大绝对误差上界：0.1597900390625；
- 投影级最大平均绝对误差：1.08617e-5；
- 但 16 层重复 A4 量化会放大 BF16 累加顺序差异，完整模型 logits 相对 L2
  差异为 0.20639，生成序列在第三个报告 token 分叉。

因此当前可以陈述“被测 M16 prefill 三轮加速中位数 1.527x”，但不能陈述“端到端质量无损”。
该后端保持 explicit opt-in（显式选择），不替换默认质量路径。

### 5. 测试、发布与仓库审计

- 正式源码树：305 passed + 37 subtests；
- 隔离发布树：305 passed + 37 subtests；
- wheel 构建成功；
- sdist 构建成功，并能从 sdist 重建 wheel；
- 隔离安装后确认新增 CUDA/Triton 源文件全部包含；
- 仓库审计：435 个可发布文件，0 findings；
- `git diff --check` 无空白错误，仅有现有行尾格式提示。

三轮性能 JSON 同时固化 Python 3.12.3、PyTorch 2.14.1+cu130、CUDA 13.0、
Triton 3.8.0、Transformers 5.20.2、输入种子和 token SHA-256；汇总器会拒绝
协议或输入哈希不一致的试验，避免把不同环境的数据混成一个加速数字。

wheel 是可直接安装的 Python 二进制分发包；sdist 是源码分发包。两条路径都通过，
说明项目不依赖当前工作目录中的偶然文件。

## 当前完成度

| 层级 | 当前完成度 | 证据边界 |
|---|---:|---|
| E8/NestQuant 核心算法 | 92% | 核心 codec、multi-beta、LDLQ/QA-LDLQ 和边界测试完成；非 2 次幂逐项上游对齐未完成 |
| 1B 量化质量评测 | 95% | WikiText-2、消融、下游和 packed W4A4 PPL 完成；CUDA M16 质量等价未成立 |
| 真实低比特存储 | 98% | NQMC/NQRT4、NQA4、NQKV v2、校验和独立回载完成 |
| packed CUDA runtime | 92% | 完整 1B W4A4KV4 路径和 M16 内核完成；M1/其他 M tile 未做 |
| CUDA Graph decode | 95% | 固定 batch 和 capacity-8 ragged runner 已验证；多容量 ragged pool 未统一 |
| 在线推理服务原型 | 80% | continuous batching、负载评测、容量曲线和 10 分钟 soak 完成；无网络 API/生产 SLA |
| 跨模型规模验证 | 20% | 用户已明确不纳入当前路线，不阻塞本地结项 |
| GitHub 发布准备 | 97% | 文档、测试、wheel/sdist 和审计闭环；未替用户 commit/push/release |
| 作为 AI Infra 简历项目 | **97%** | 已具备算法、格式、kernel、完整模型、服务评测、故障复盘与证据边界 |

项目不是 100%，因为生产级网络服务、跨模型验证、直接 INT4 Tensor Core GEMM、
所有 M 形状内核和论文所有细节逐项复现都不在当前已验证范围。

## 明早需要用户选择的路线

下一项开发会改变项目重点，因此今晚没有擅自选择：

1. **M1 decode 内核（建议）**：直接改善自回归生成 TPOT，最符合 AI Infra 推理项目；
2. **M32/M64/M128 prefill 内核**：扩展长 prompt 覆盖，但不会改善逐 token decode；
3. **M16 长文本质量协议**：先研究累加顺序导致的质量分叉，再扩展 kernel 形状。

建议选择第 1 条。当前运行时计数已经证明 M16 命中 1,232 次、M1 回退 448 次，
说明 decode 是最清晰且最有价值的下一处未覆盖热点。但 M1 通常更接近 GEMV
（矩阵-向量乘），不能简单照搬 M16 的 MMA tile，预计需要新的数据布局与流水线。

## 可复现入口

- 内核正确性与微基准：`scripts/build_test_w4a4_mma.py`
- 112 投影审计：`scripts/validate_w4a4_mma_projections.py`
- 完整模型对照：`scripts/benchmark_full_nqrt_model.py`
- 复现说明：`docs/w4a4_cuda_mma_reproduction.md`
- 完整模型报告：`results/llama32_1b_w4a4_cuda_mma_m16.md`
- 三轮汇总 JSON：`results/raw/llama32_1b_w4a4_cuda_mma_m16_trials_summary.json`
- 编译指令审计：`results/raw/w4a4_cuda_mma_binary_audit.json`

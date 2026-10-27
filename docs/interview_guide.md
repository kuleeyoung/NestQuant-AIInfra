# NestQuant-AIInfra 面试讲解手册

这份材料用于 AI Infra、模型压缩和推理优化岗位。所有数字只引用仓库内已经完成的本地实验；
论文结果、质量仿真、真实低比特存储和 packed kernel 性能必须分开陈述。

## 90 秒项目介绍

> 我围绕 NestQuant 做了一个从量化算法到推理 runtime 的完整项目。算法层实现了 E8 最近
> 格点、嵌套码、multi-beta、Hadamard 旋转和 Hessian-aware block-LDLQ，并在
> Llama-3.2-1B 上完成 WikiText-2、PIQA 和 HellaSwag 评测。工程层没有停在 fake quant：
> 我设计了 NQMC/NQRT4 低比特 checkpoint，支持哈希校验、流式加载、随机访问和独立回载，
> 然后写了 CUDA kernel，从 packed E8 码流在线恢复 BF16 tile 并使用 Tensor Core 累加，
> 使完整模型 warm CUDA allocation 从 2390.1 MiB 降到 1028.6 MiB。全 packed 路径仍慢于
> BF16，所以我没有把它包装成加速结果，而是继续用 profiler 定位 E8 解码和 launch 开销，
> 接入 CUDA Graph、ragged KV cache 和 continuous batching，最后实现 packed/BF16 混合
> 驻留策略。512 MiB 预算相对全 packed 把 TTFT 和 TPOT 中位数降低 42.4% 和 45.8%，
> 同时比全 BF16 少 36.3% CUDA allocated memory。项目目前有 246 项测试、25 个参数化
> 子测试，并通过 wheel、sdist、隔离安装、链接和敏感信息审计。

正常语速约 85～100 秒。面试时不必机械背诵，应围绕“算法—存储—执行—调度—权衡”五层展开。

## 简历四条

1. 实现 E8 嵌套格量化、multi-beta、Hadamard 旋转和 Hessian-aware block-LDLQ，在固定
   Llama-3.2-1B/WikiText-2 协议上完成 W4/W4KV4/W4A4/W4A4KV4 PPL、PIQA/HellaSwag
   与消融评测；当前 NQRT4 W4 PPL 为 10.070482，BF16 基线为 9.758112。
2. 设计 NQPK/NQRT4/NQMC 低比特 checkpoint，支持 payload/file SHA-256、流式校验、
   O(1) tensor/alias 定位和独立回载；实现从 packed E8 stream 在线恢复 BF16 tile 并通过
   Tensor Core 累加的 CUDA runtime，使完整 1B 模型 warm CUDA allocation 从 2390.1 MiB
   降至 1028.6 MiB（-57.0%）。
3. 实现固定地址 ragged KV cache、CUDA Graph Pool、slot 回收/复用和 continuous-batching
   admission；10 分钟真实 GPU soak 完成 17,264 个请求和 86,348 个 token，无残留请求，
   并通过配对负载扫描定位当前配置约 25～30 req/s 的饱和拐点。
4. 实现 packed/BF16 混合权重驻留，将 112 个投影组成 64 个共享变换原子组；在 RTX 5070
   六档全模型测试中，512 MiB 预算相对全 packed 将 TTFT/TPOT 中位数降低 42.4%/45.8%，
   同时相对全 BF16 减少 36.3% CUDA allocated memory，并通过 ragged CUDA Graph 正确性验证。

数字来源依次为：

- [NQRT4 W4 质量](../results/llama32_1b_nqrt4_stateless_w4.md)
- [完整模型 packed runtime](../results/llama32_1b_nqrt4_full_e2e.md)
- [continuous batching soak](../results/llama32_1b_nqrt4_continuous_batch_soak.md)
- [服务容量扫描](../results/llama32_1b_nqrt4_service_capacity.md)
- [混合驻留 Pareto](../results/llama32_1b_nqrt4_hybrid_pareto.md)

## 从零讲项目的主线

```text
为什么量化
  模型权重显存与带宽成本高
        ↓
为什么不是普通 INT4
  E8 向量量化利用 8 维几何结构，multi-beta 控制有限码本 overload
        ↓
怎样保证模型质量
  Hadamard 缓解 outlier，LDLQ 用校准 Hessian 反馈量化误差
        ↓
怎样成为真实文件
  NQPK 位流 → NQRT tensor record → NQMC model checkpoint
        ↓
怎样真正执行
  packed stream → E8 恢复 → BF16 tile → Tensor Core MMA
        ↓
怎样进入动态推理
  CUDA Graph → ragged KV slot → continuous batching → workload benchmark
        ↓
性能仍慢怎么办
  profiler 定位根因，不隐瞒负结果；用混合驻留提供显存—延迟 Pareto
```

## 高频追问与回答要点

### 1. 这个项目是论文复现还是你自己的系统？

NestQuant 的 E8 嵌套量化思想来自论文和作者代码；本项目重新实现并测试了核心数学模块，
又新增了真实模型容器、stateless random-access dither、packed CUDA/Tensor Core runtime、
CUDA Graph、continuous batching、服务 workload 和混合驻留。不能称为论文所有细节逐项复现，
也不能把作者结果当成本地结果。

### 2. 4 bit 到底代表什么？

它不是每个标量权重直接对应一个有符号 INT4。NQRT4 的 code 描述 8 维 E8/Voronoi residue，
还需要 beta ID、row norm、dither seed 和 codec semantics 才能恢复权重。因此当前 kernel 是
“从低比特码流恢复 BF16 tile，再做 BF16 Tensor Core MMA”，不是直接 INT4 MMA。

### 3. beta 是什么？为什么需要多个 beta？

beta 控制嵌套格的粗细比例，也决定有限 residue code 的覆盖范围。beta 小时粒度细但容易
overload；beta 大时覆盖范围大但重构更粗。multi-beta 为不同 row/block 选择候选尺度，在
量化误差与 overload 之间折中。

### 4. overload 是什么？

overload 是输入落到有限码本可可靠表示范围之外。它不同于普通舍入误差：码字可能发生跨边界
折返，产生很大的重构误差。项目用归一化、Hadamard 和多个 beta 降低它的影响。

### 5. LDLQ 为什么需要 Hessian？

普通量化最小化权重本身的误差；LDLQ 用校准激活近似 Hessian，对输出更敏感的方向赋予更高
代价，并按 LDL 分解顺序把当前量化误差反馈给后续坐标。项目曾发现 QA-LDLQ 的 Hessian
归一化存在 2048 倍语义差异，修正后消除了严重退化，但没有优于普通 LDLQ，因此保留负结果。

### 6. 为什么 Hadamard 有用？

归一化 Hadamard 是正交变换，不改变 L2 范数，却能把少数大 outlier 的能量扩散到多个坐标，
让固定比特量化更均匀。离线保存 `W H` 后，运行时必须对输入做匹配的 `X H`，否则坐标语义不一致。

### 7. 文件变小是否证明推理显存下降？

不能。文件大小只证明存储压缩；如果加载后把整个权重反量化成 BF16，运行时显存并没有保持低比特。
本项目分别测量文件码率、反量化质量和 packed-resident CUDA allocation，避免混淆三种结论。

### 8. 为什么 packed 比 BF16 慢？

BF16 resident 直接调用高度优化的 dense GEMM；packed 路径每次都要读取 code/beta、重放 hash
dither、做两次 E8 最近点恢复、生成 BF16 fragment，再执行 MMA。完整模型每 token 还有 112 个
projection 调用和 Split-K/归约开销。packed 节省显存，但增加大量在线指令和 launch。

### 9. CUDA Graph 解决了什么？

它把固定形状的一串 GPU 操作预先捕获，replay 时减少 Python、dispatcher 和 kernel launch 的
CPU 开销。它不能消除 E8 解码计算，也不自动支持动态 batch、动态地址或任意 cache position，
所以项目又实现固定地址 slot、active mask 和 per-request position。

### 10. continuous batching 与普通动态 batching 有什么区别？

普通 batching 常等整批请求结束；continuous batching 在每个 decode step 都允许完成请求退出、
新请求进入并复用 slot，从而减少短请求被长请求阻塞。难点是 KV cache 生命周期、请求独立位置、
取消和 slot 复用时不能串数据。

### 11. 为什么吞吐平台化后尾延迟继续上升？

GPU 服务率已经接近上限，新增请求主要进入队列而不是提高完成速率。根据排队论，利用率接近 1
时等待时间会快速上升，因此 30→50 offered req/s 的 token throughput 只从 121.61 增至
129.10，而 TTFT P95 从 335.99 ms 增至 746.94 ms。

### 12. 为什么混合驻留使用“节省毫秒/增量 MiB”？

不同 projection 从 packed 换成 BF16 的显存成本和延迟收益不同。该比率近似单位显存的收益，
便于在预算内优先驻留热点。它是确定性贪心启发式，不是背包问题的全局最优证明；报告的 Pareto
也只针对六个实测方案。

### 13. 怎样证明没有只挑最好的一次？

正式 benchmark 固定 warmup、repetition、请求内容、模型/data revision 和统计口径；服务实验
报告 P50/P95/P99，混合驻留每点重复 20 次。validator 会重新检查配置哈希、预算、projection
选择、单调性、端点和 Pareto 前沿。

### 14. 项目中最重要的一次排错是什么？

J2 CUDA extension 使用默认 stream 启动 kernel，而模型由 PyTorch current stream 和 CUDA
Graph 驱动，导致操作可能漏出捕获。修复方式是从 PyTorch 获取 current stream 并在同一 stream
发射；随后用 Graph/eager logits、argmax 和连续 token 回归，而不只检查“能运行”。详见
[故障复盘](incident_postmortems.md)。

### 15. 如果继续优化，下一步是什么？

先用 Nsight Compute 判断 E8 partial kernel 的 instruction stall、register pressure 和 occupancy；
然后比较专用 M=1 GEMV 与当前补零 m16 MMA，最后评估 QKV、gate/up 和 Hadamard 融合。不会在
没有 profiler 证据时盲调 block size，也不会把更强 GPU 当成修复 kernel 结构的办法。

## 不能说的内容

- 不能说“完整复现论文所有结果”；
- 不能说“A4/KV4 已真实低比特常驻执行”；
- 不能说“实现 INT4 Tensor Core GEMM”；
- 不能说“packed 比 BF16 更快”；
- 不能把 engine offered req/s 写成生产 HTTP QPS；
- 不能把单张 RTX 5070、固定请求分布的结果外推到其他 GPU 或 3B/8B；
- 不能把六个实测点称为所有 (2^{64}) 组合的全局 Pareto 最优解。

## 面试前自测

不打开代码，尝试在两分钟内回答：

1. NQRT4 的 4 bit 编码的是什么？
2. 为什么还需要 beta ID、row norm 和 dither seed？
3. fake quant、真实 packed 文件、packed-resident execution 有什么区别？
4. 为什么全 packed 节省 57% 显存却慢于 BF16？
5. CUDA Graph 和 continuous batching 各解决哪一层问题？
6. 512 MiB 混合方案的收益和边界是什么？
7. 你亲自定位并修复过哪三个问题？
8. 哪些结果是负结果，为什么仍有价值？

任何一个问题答不清时，应回到对应代码和正式报告，而不是继续背更多数字。

# 本地硬件路线与项目完工标准

## 定义

本路线将项目的本地完工边界固定为：在单张 RTX 5070 12 GB 上，完成
Llama-3.2-1B 的可复现量化、真实 packed 权重执行、动态请求调度原型、
服务级测量和可公开仓库。3B/8B 属于后续规模扩展，不阻塞本地项目结项。

“完工”不等于论文全部细节、生产级在线服务或真正的 INT4 Tensor Core GEMM。
以下三项明确保留为后续工作：跨模型规模验证、packed A/KV compute、直接
INT4 MMA 数据通路。

## 完工验收门

只有以下五个门全部通过，才将本地项目标记为 complete。

### Gate 1：证据与仓库固化

- `configs/project.json`、README、结果索引和简历边界保持一致；
- 代码按 codec、runtime、Graph、serving、docs/tests 拆成可审查提交；
- clean clone 后能安装并运行 CPU 测试；CUDA 机器能运行 E8、J2、stream 和
  CUDA Graph smoke；
- 每个正式数字都能追溯到命令、配置、原始 JSON、设备和版本；
- 不提交模型权重、Hugging Face token、编译缓存或大体积 profiler trace。

验收：完整测试通过；`git diff --check` 通过；公开文档不存在互相矛盾的状态。

### Gate 2：多 batch CUDA Graph Pool

- 预捕获 batch 1/2/4/8 Graph；显存允许时加入 batch 16；
- 根据活跃请求数选择最小可容纳 Graph，并用 padding slot 填补空位；
- token、per-request cache position、active mask 可在 replay 间更新；
- Graph 与 packed eager 的逐步 logits、argmax 和多 token 生成一致；
- 记录每张 Graph 的捕获时间、私有显存、P50/P95/P99 replay latency。

验收：活跃请求数 1～8 的输出没有跨 slot 污染，数值与 packed eager 一致；
相对 eager 的收益和 Graph 显存成本均有机器可读结果。

当前进度（2026-10-26）：batch 1/2/4/8 的容量路由、padding 和有效行裁剪已实现；
RTX 5070 上 active 1/2/3/4/5/7/8 与 packed eager 的 logits 最大误差均为 0，四张图
捕获时的额外 allocated memory 合计约 131.7 MiB。这一阶段只覆盖同步位置；后续路线 B
已补上 batch-2 的 per-request position、active mask 和 slot 复用验证。

路线 B 进展（2026-10-26）：新增固定地址 ragged cache controller，在 batch-2 packed 1B
CUDA Graph 中验证请求位置 17/16，以及旧 slot 释放后位置 17/3 的新旧请求共存。两种场景
Graph 与 ragged eager 的 logits 最大误差均为 0，argmax 与独立完整前向一致。下一步需将
ragged runner 合并到 batch 1/2/4/8 Pool，完成随机 workload 回归后关闭 Gate 2。

### Gate 3：Continuous Batching 调度原型

- 实现 waiting/running/finished 请求状态和 slot 分配、回收、取消；
- 支持不同 prompt length、不同生成进度和 EOS/max-token 结束；
- 新请求可在旧请求完成后复用 slot，不等待整批结束；
- KV cache 在 slot 复用时不会串请求；
- 超出固定 cache/Graph 支持范围的请求会在入队前明确拒绝，不会静默破坏缓存。

验收：确定性混合请求测试和随机 workload soak test 均通过；连续批处理相对
串行/静态批处理的吞吐、TTFT 和 TPOT 有同机对照。

当前进度（2026-10-26）：waiting/running/finished/cancelled 状态、FIFO admission、EOS/
max-token 结束、slot 释放与复用的确定性状态机已实现并测试；host-driven greedy engine
已在 packed 1B 上完成三请求/两 slot 独立生成对照，并通过 200 请求随机 soak。24 请求
closed-loop 中即时 admission 为 96.47 token/s，静态 batch-wave 为 196.16 token/s；瓶颈是
10 次小批量 prefill 与 decode 串行。该负结果触发了下述 C1 策略优化。

C1 完成（2026-10-26）：实现 `min_prefill_batch`、`max_wait_steps` 和 idle fast path，并扫描
六组 closed-loop 策略及 20/100/250 ms 三档固定间隔 open-loop workload。capacity-8 的本机
默认选为 `8/4`：闭环 215.76 token/s；50 req/s 下 142.33 token/s、TTFT P95 224.97 ms；
4 req/s 下 TTFT P95 59.46 ms，未产生凑批等待惩罚。配置固化于 `configs/serving.v1.json`。

D1 完成（2026-10-27）：在 RTX 5070 上完成 600 秒 packed 1B 真实 GPU soak。固定
capacity 8、目标 backlog 16，共提交并完成 17,264 个请求、生成 86,348 token，平均
143.79 token/s；368 次有限 logits 检查全部通过，排空后请求注册表为 0。CUDA allocated
采样范围仅 512 bytes，reserved 在首分钟扩容后固定；60 秒后的 CPU RSS 总范围约 11.0 MiB，
呈离散扩容后平台化，未观察到逐请求线性泄漏。独立完整前向抽查严格一致，且每个服务输出
token 的 max-logit gap 均为 0。报告见
`results/llama32_1b_nqrt4_continuous_batch_soak.md`。Gate 3 在上述单进程、greedy、固定容量
原型边界内关闭；网络 API、随机采样和生产 SLA 仍属于 Gate 4/后续工作。

### Gate 4：服务级 benchmark 与显存—延迟策略

- workload generator 覆盖固定速率、突发和混合 prompt/output length；
- 报告 P50/P95/P99 TTFT、TPOT、端到端延迟、请求吞吐、token 吞吐、GPU
  显存和失败率；
- 实现至少一种显存—延迟优化：优先选择 packed/BF16 混合权重缓存；
- 按“节省时间/额外 BF16 字节”选择热点 projection，并生成 Pareto 曲线；
- 运行长时间 soak test，检查 CUDA allocated/reserved、CPU RSS、NaN 和缓存泄漏。

验收：结果可重复，性能结论有 warmup、重复次数和统计口径；不存在仅报告最好
单次结果的情况。

A/服务 benchmark 完成（2026-10-27）：新增统一 open-loop harness，覆盖 50 req/s 固定到达、
16 请求三轮突发和 20 req/s Poisson 双峰长短请求；`8/4` 与 `1/0` 各重复 3 次。计时前预热
4～48 的全部 prompt width，三种负载共 18 次正式 run 全部完成且无注册表残留。`8/4` 相对
`1/0` 的 token throughput 分别提高 42.1%/46.2%/16.2%，TTFT P95 分别降低
47.5%/51.0%/33.8%，E2E P95 分别降低 47.8%/51.3%/44.1%。报告见
`results/llama32_1b_nqrt4_service_benchmark.md`。网络 API 不作为当前 engine benchmark 的已完成项。

混合驻留 A 完成（2026-10-27）：112 个 projection 被划分为 64 个原子组，按 family-median
预测节省时间/增量 MiB 的确定性贪心策略选择 BF16 resident 权重。RTX 5070 上完成
0/256/512/768/1024/1400 MiB 六点全模型实测与独立校验；六点均属于本次有限候选集的
三维 Pareto 前沿。512 MiB 点相对全 packed 将 TTFT/TPOT 中位数降低 42.4%/45.8%，
同时比全 BF16 resident 少 36.3% CUDA allocated memory。该方案还通过 batch-2 ragged
CUDA Graph 不同 position、暂停恢复与 slot 复用正确性验证。见
`results/llama32_1b_nqrt4_hybrid_pareto.md`。Gate 4 在单卡、本地 engine、有限 workload
和贪心候选集合的边界内关闭；不包含网络 API、生产 SLA 或全组合最优性证明。

容量补充（2026-10-27）：对 `8/4` 做 4/10/20/30/40/50 req/s 配对负载扫描，固定每个
repetition 的请求内容、只改变到达率。token throughput 在 30 req/s 后由 121.61 token/s
逐渐平台化到 50 req/s 的 129.10 token/s，而 TTFT P95 从 335.99 ms 上升到 746.94 ms；
本配置饱和拐点约 25～30 req/s。见 `results/llama32_1b_nqrt4_service_capacity.md`。

### Gate 5：公开交付与面试材料

- README 提供五分钟快速开始、架构图、复现命令、结果表和证据边界；
- 提供一份从 E8 算法到 NQMC、J2 runtime、Graph Pool、Scheduler 的架构说明；
- 提供一份故障复盘，至少包含 CUDA current-stream 漏捕获问题；
- 提供简历四条、90 秒项目介绍和常见追问答案；
- GitHub 仓库不依赖本机绝对路径，未上传私密或大体积原始产物。

验收：在新目录按 README 完成安装和 smoke；简历中的每个数字可定位到仓库报告。

当前进度（2026-10-27）：README 已增加五分钟审阅路径与系统主线图；架构说明覆盖从 E8、
NQMC/NQRT4 到 J2、Graph、scheduler 和混合驻留；新增工程故障复盘、简历四条、90 秒介绍、
15 个高频追问和发布检查清单。隔离发布树已完成 source wheel、246 项测试、sdist→wheel、
独立安装、console entry point 和 CUDA 源文件检查；仓库审计 381 个可发布文件、0 finding。
Gate 5 的技术与材料验收已经通过。尚未替用户执行 Git commit、创建远程仓库或发布 release，
这些属于对外状态变更，应在用户确认仓库归属与提交方案后执行。

## 完工时的目标成熟度

“当前”列已按 2026-10-27 的 D1 soak、服务负载和容量扫描更新。

| 层级 | 当前 | 本地完工目标 | 完工判据 |
|---|---:|---:|---|
| E8/NestQuant 核心算法 | 90% | 92% | 核心模块和边界测试稳定；未实现论文细节列入矩阵 |
| 1B 量化质量评测 | 90% | 95% | 正式 PPL、下游任务、消融和结果索引一致 |
| 真实低比特存储 | 95% | 98% | NQMC/NQRT v4、随机访问、校验、独立回载和文档完整 |
| packed CUDA runtime | 85% | 85% | 完整模型、长时间负载与混合驻留 Pareto 已验证；仍慢于 BF16 resident |
| CUDA Graph decode | 90% | 95% | 固定 batch Pool 与 capacity-8 ragged Graph 已验证；尚差 ragged 多容量池统一 |
| 在线推理服务 | 78% | 80% | engine 调度、slot 生命周期、tail latency、容量曲线与 soak 完成；无网络 API |
| 跨模型规模验证 | 20% | 20% | 明确保留为云端扩展，不作为本地完工门槛 |
| GitHub 发布准备 | 95% | 95% | 材料、隔离发布树、wheel、测试和审计通过；仅剩用户确认后的提交与远程发布 |
| 作为简历项目 | 95% | 92%～95% | 简历四条、90 秒介绍、追问、故障复盘及数字证据已闭环 |

本地路线不会把下列项目写成 100%：论文全量复现、生产 serving、跨模型规模、
packed A/KV、INT4 Tensor Core GEMM。保留这些边界比虚构 100% 更可信。

## 执行顺序与时间盒

| 阶段 | 目标 | 预计时间 | 停止条件 |
|---|---|---:|---|
| A | 状态、测试、复现入口和 GitHub 固化 | 2～4 天 | Gate 1 通过 |
| B | batch 1/2/4/8 Graph Pool | 3～6 天 | Gate 2 通过 |
| C | continuous batching 调度原型 | 5～10 天 | Gate 3 通过 |
| D | 服务级 benchmark + 混合权重缓存 | 4～8 天 | Gate 4 通过 |
| E | README、架构、故障复盘和面试材料 | 2～4 天 | Gate 5 通过 |

预计总工期为 3～5 周。阶段 B/C 若发现 Transformers `StaticCache` 不能安全支持
per-request position，不通过隐藏同步或复制来伪造结果：先建立最小复现，再选择
自定义 cache update 或固定最大 batch active-mask 方案，并记录偏差。

## 本地完工后的可陈述成果

本地结项后可以陈述：项目在 Llama-3.2-1B 上完成 NestQuant 质量评测、真实
NQRT4 packed 权重存储和 BF16 Tensor Core 累加 runtime，并实现多 batch CUDA
Graph Pool 与 continuous-batching 调度原型；在固定 workload 下报告显存、P50/P95/P99
TTFT/TPOT 和吞吐，形成可复现的显存—延迟 Pareto 曲线。

仍不得陈述：在 3B/8B 上验证、超过 BF16 resident 延迟、生产级 QPS、真实 packed
A4/KV4、INT4 Tensor Core GEMM 或论文所有数字逐项复现。

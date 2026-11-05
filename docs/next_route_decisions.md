# 下一阶段路线决策清单

更新时间：2026-11-02

本文只记录会明显改变开发重心的选择。能通过测试、证据一致性或安全边界唯一确定的工作，
不列成选择题。

## 决策 1：Gate 4 的下一项核心工作（已选择并完成 A）

### A：packed/BF16 混合权重缓存 Pareto（推荐）

状态：已于 2026-10-27 完成。六个预算点、独立 validator、可提交精简摘要及 512 MiB
ragged CUDA Graph 正确性均已闭环；详见
`results/llama32_1b_nqrt4_hybrid_pareto.md`。以下内容保留为当时的决策依据。

将耗时最高的一部分 projection 保持为 BF16 resident，其余继续使用 NQRT4 packed，按照
“节省的延迟 / 额外 BF16 bytes”选择热点层，测量完整模型的显存—TTFT—TPOT 曲线。

选择理由：

- 当前服务容量已经定位为约 129 token/s 的计算平台，显存并未接近 12 GB 上限；
- 完整 packed runtime 仍慢于 BF16 resident，混合缓存直接针对已证实瓶颈；
- 能把“量化省显存但变慢”推进为可配置的系统权衡，是当前最强的 AI Infra 故事；
- 完成后可关闭路线图 Gate 4 的剩余核心条目。

成本和风险：需要定义缓存预算、热点评分和 Graph 重捕获规则；若热点分散，Pareto 改善可能有限，
但负结果仍能给出可解释的容量边界。

### B：先做 HTTP/JSON 网络服务

在现有 engine 外增加 tokenizer、请求协议、异步队列、取消/超时、健康检查和指标端点。

选择理由：产物更像可调用服务，演示直观；可以测网络端到端 TTFT 和错误处理。

成本和风险：会把时间转移到 Web 框架和协议工程，不能解决 packed kernel 慢于 BF16 的核心性能
问题；若只做薄 API，简历技术增量可能小于混合缓存。

### C：先关闭 Gate 4，转 GitHub/面试交付

保留当前 packed 性能负结果，不再做新的 runtime 优化，优先 clean-clone、提交整理、故障复盘、
90 秒介绍和追问材料。

选择理由：当前项目已经可写简历，能最快形成公开交付。

成本和风险：路线图中混合缓存 Pareto 仍未完成，packed runtime 的显存—延迟选择只能引用离散点，
项目技术上限低于路线 A。

## 决策 2：本地 Gate 4 关闭后的发展方向

### A：GitHub 发布与面试材料闭环（推荐）

状态：本地材料与发布检查已于 2026-10-27 完成。README 五分钟路径、架构、故障复盘、
简历四条、90 秒介绍、高频追问、隔离发布树和仓库审计均已闭环。剩余工作是由用户确认
GitHub 仓库归属、是否保留当前历史以及提交拆分后，执行对外发布。

整理提交边界、从干净目录复跑安装/smoke、更新架构图和故障复盘，并生成四条可核验简历描述、
90 秒项目介绍和高频追问答案。优点是把现有大量技术证据转化为可展示成果，风险最低。

### B：HTTP/JSON 网络服务

在现有 engine 外增加 tokenizer、异步请求队列、取消/超时、健康检查与指标端点。演示更直观，
但主要增量偏 Web/协议工程，不解决 kernel 性能瓶颈。

### C：云端 3B/8B 跨规模验证

租用更大显存 GPU，将同一协议扩展到 Llama-3.2-3B 与 Llama-3.1-8B。它能增强结论的外部有效性，
但需要云资源、模型下载和较长正式评测；当前本地硬件无法公平完成全部矩阵。

## 决策 3：是否安装绘图库生成正式曲线图

当前虚拟环境没有 matplotlib，JSON 和 Markdown 表格已经完整。推荐在决定进入公开交付阶段时，
将绘图依赖锁入单独的 analysis/dev requirements，再生成 SVG/PNG；现在不为单张图临时安装未锁定依赖。
这不是阻塞项。

## 决策 4：直接 packed W4A4 CUDA 内核的下一覆盖形状

M16 correctness 和完整 1B sequence-16 prefill 已完成：相对 tiled Triton 的中位
三轮延迟中位数为 117.17 对 178.93 ms（加速中位数 1.527x），每轮运行时计数确认 1,232 次自定义内核命中；
该阶段最初观察到 M1 decode 每轮 448 次自动回退，用户随后选择路线 A；当前状态如下：

### A：M1 decode/GEMV（推荐）

状态：已于 2026-10-29 完成。新增 4-lane direct-packed M1 GEMV，112/112 个真实投影
通过审计；严格 M1-only 三轮实验将 TPOT 中位数从 170.01 降至 94.53 ms，
加速中位数 1.803x（1.790～1.815x），每轮 896 次 M1 命中且 decode fallback 为 0。
生成序列仍会分叉，因此保持显式 opt-in。详见
`results/llama32_1b_w4a4_cuda_gemv_m1.md`。

### B：M32/M64/M128 prefill（下一候选）

状态：已完成。M2/M4/M8/M32/M64/M128 已与 M1/M16 组成固定形状 CUDA family；
M32/M64/M128 q-proj 相对 tuned Triton 为 1.642x/1.909x/1.503x，完整 1B
seq32/64 prefill 为 1.464x/1.603x。任意 M 由 M128 分块和尾块 padding 覆盖，详见
`results/w4a4_large_m_multiwarp_probe.md` 和
`results/w4a4_dynamic_m_chunked_staged.md`。

### C：建立长文本/生成质量协议（已进入正式质量门阶段）

已完成逐 token logit margin、逐层敏感度、多 prompt、确定性 token-ID 反例、raw-FP32、
partial-reduction 和 FP64 oracle 归因。静态敏感层黑名单被反例否决：层误差可能抵消或叠加，
回退更多层不保证单调更安全。随后建立 production NQKV Graph teacher-forced 协议，使 baseline
和 split-K 在每一步接收相同 token，并采用 AB/BA 顺序、ABA/BAB 重放和 context-level bootstrap。
首轮 32×32 的置信区间不足；2026-11-05 完成的 2,048-context 正式实验覆盖 65,536 个配对
decode 决策，相对 PPL 为 -0.0932%，20,000 次 context-bootstrap 95% CI 为
[-0.5664%, +0.3658%]，通过预声明的 1% 非劣性门槛，确定性重放误差为 0。结合既有 10 对
fresh-process 服务 A/B，决策更新为：在线服务对精确 `M=1,K=8192,N=2048` down-projection
默认启用 split-8，保留 `--disable-m1-down-splitk` 回滚；底层通用 Linear API 与 benchmark
仍要求显式选择，避免把该结论外推到其他模型、形状或实验协议。argmax agreement 为 77.774%，
因此默认策略建立在平均概率质量非劣性上，不声称自回归 token 完全一致。

## 决策 5：双 NQKV CUDA Graph 之后如何消除逐 token 主机状态检查

当前状态（2026-11-02）：完整 1B continuous runner 已捕获普通 decode 和页边界 pack
两个 Graph。连续写满两个页面时，Graph 与 eager 的 logits、packed payload、回载 K/V
逐位一致；边界 eager fallback 为 0，终态物理页全部回收。真实 1B 单次 smoke 从原始
eager 的 8.03 提升到 9.49 token/s，边界步从旧 fallback 的 244.2 ms 降到 88.2 ms。

剩余问题：runner 每个 decode 步仍读取 GPU hot length 来识别边界；命中边界时由 CPU
预留物理页并在 Graph 后提交 handle。模型和 KV 编码已经不回到 CPU，但这次状态读取会
形成同步点。以下路线都会改变运行时接口或页池生命周期，因此留给用户确认。

### A：调度器显式维护并传入 boundary slots（推荐的低风险路线）

让 continuous-batching scheduler 在已有请求长度状态上计算下一 token 是否填满页面，
runner 直接消费 host-side slot 列表，不再读取 GPU hot length。保留当前按需页分配和双
Graph，改动集中、容易验证，适合先获得稳定的 TPOT 收益。

代价：runner API 增加边界提示；必须用断言抽查调度器长度与 GPU 元数据一致，避免静默漂移。

### B：每槽预留物理页 ring，由 GPU 自主换页（高上限路线）

为活跃槽位预先准备一组 page ID，边界 Graph 自行选取下一页和推进元数据，CPU 不参与
单 token 换页。理论上可以去掉边界同步，并更接近生产 paged-KV runtime。

代价：显存预留增加；取消、slot 复用、超长请求、页耗尽和异常回滚都必须重新设计，
测试矩阵明显扩大。当前单机 smoke 尚不足以证明这项复杂度必然换来可观收益。

状态更新（2026-11-02）：已完成真实 1B Probe 0。normal 控制面中位仅占同步分段总时间
0.85%；boundary 控制面占 4.46%，但每 16 token 才发生一次。全部已测控制面成本的
乐观摊销上界约为 0.8 ms/token（约 1.1%）。已触发 `<2%` 停止条件，当前不进入完整
reserve-window/page-ring 实现。若未来 decode 降到 20 ms 以下或更大 capacity 使控制面
超过 5%，再重新打开本路线。详见 `results/nqkv_graph_control_plane_probe0.md`。

数据面随后完成 M2 down-projection split-K 优化：仅对 `M=2, K=8192, N=2048`
启用 split-8。单层由 2094.68 us 降至 497.28 us（4.21x），CUDA Graph replay 为
484.11 us；完整 1B normal + boundary 双 Graph 的 GPU 总时间由 150.385 ms 降至
117.654 ms（-21.77%），boundary fallback 仍为 0。该结果说明当前优先级应继续放在
已定位的数据面长 K 形状，而不是重开收益上限约 1.1% 的控制面 allocator 路线。详见
`results/w4a4_m2_down_splitk.md`。

继续按同版本 BF16-KV 做分阶段 A/B 后，普通 prompt-8 decode 的 NQKV4 开销仅
1.83%，而 prompt-8/prompt-15 prefill 开销分别为 25.48%/25.97%。因此新增仅覆盖
“完整 capacity、空缓存、等长且小于一页”的 dense prefill 批量写入路径。prompt-15
prefill 由 149.00 ms 降至 132.35 ms，边界 workload 吞吐由 22.53 提升至
24.20 token/s，相对 BF16 的差距由 16.01% 缩至 9.79%；65 项 NQKV runtime 测试
及 16 个参数化子测试通过。短 prompt 总吞吐仍有跨 run 波动，不宣称普遍服务加速。
详见 `results/nqkv_dense_prefill_optimization.md`。

在此基础上已将 graph-safe K1 writer 扩展为 multi-token ragged sub-page writer。
`[2,15,8,64]` 单层 GPU-only launch 为 42.01 us，相对原事务式 writer 的
315.34 us 为 7.51x；默认 backend 将 mask/prefix/page-room 校验提升到每个 model
forward 一次，16 层复用固定地址 launch。真实 smoke 审计命中 64 次 ragged layer
launch、0 fallback、0 终态残留；五次吞吐中位 12.41 token/s，TPOT P95 中位
171.02 ms。写满/跨页仍回退原事务路径，未扩大到 GPU 页面分配。详见
`results/nqkv_ragged_subpage_writer.md`。

随后完成 20 对同进程、同模型、同请求、奇偶轮换顺序的开关 A/B。ragged GPU
writer 将吞吐中位从 12.34 提升到 13.61 token/s，配对中位 +9.72%，10,000 次
bootstrap 95% CI 为 [+7.84%, +11.49%]；TTFT/TPOT/E2E P95 分别改善
11.56%/9.41%/9.28%，四项均为 20/20 对改善且置信区间不跨 0。结果超过预设
5% 继续门槛，ragged sub-page 保持默认，并允许进入单页写满与 GPU pack 融合；仍不
扩大为通用 GPU allocator。原始协议见 `results/nqkv_ragged_writer_paired_ab.md`。

单页写满与 GPU pack 融合也已完成：控制面在 forward 前跨层预留目标页，ragged
writer 写满 hot page 后直接调用 graph-safe pack/publish，forward 成功后统一提交；
新增发布后回滚测试，能够撤回物理页并恢复完整 hot page。20 对正式 A/B 的吞吐配对
中位提升 **16.22%**，TTFT/TPOT/E2E P95 分别改善 **16.16%/13.02%/15.02%**，
四项均 20/20 胜出。该分支达到继续门槛并默认开启；范围仍限定为单页内连续前缀，
跨页 prefill 保持事务 fallback。下一明确候选是“单次跨页 prefill 的 GPU 分段写入”，
不直接扩张为通用 GPU allocator。证据见 `results/nqkv_full_page_pack_paired_ab.md`。

“单次跨页 prefill 的 GPU 分段写入”也已完成第一阶段：支持连续前缀、最多跨一个
页边界，并允许同一 batch 中跨页行与不跨页行混合。每行独立计算第二段源 offset，
完整 forward 成功后才提交跨层页面；直接 Hugging Face Cache 调用继续使用原事务路径。
故障注入验证了 post-KV 异常会取消预留并释放全部受影响请求，避免继续使用部分更新
的 KV。prompt-20 的 20 对 A/B 中吞吐配对中位提升 **22.25%**，TTFT/TPOT/E2E
P95 分别改善 **20.24%/17.37%/19.31%**，四项 20/20 胜出。下一候选变为支持
“一次写入两个或更多完整页面”的一般化分段计划；这会需要多页预留表与多次 pack，
仍不等同于 GPU allocator。证据见 `results/nqkv_cross_page_paired_ab.md`。

### C：停止深入 runtime，转公开提交整理

保留当前双 Graph 结果和同步边界，优先拆分提交、clean-clone 复跑和 GitHub 发布。
这是交付风险最低的路线，但不会继续缩小与 BF16 KV Graph 10.45 token/s 的差距。

## 暂不需要决定的已完成结论

- `8/4` 保持为当前 capacity-8、本机 workload 的默认 admission；硬件或分布变化时重调；
- 服务基准明确采用 steady-state 口径，4～48 prompt width 全预热；冷形状诊断单独保留；
- 容量扫描使用 paired workload，只改变 arrival rate；
- 网络服务、生产 SLA 和 3B/8B 不写入当前已完成声明；packed A/KV 只能写成
  Llama-3.2-1B correctness-first 原型，不外推为生产支持、TPOT 加速或质量无损。

# 当前正式结果与历史结果分层

本页是项目结果的唯一推荐入口。它不覆盖历史实验，而是说明当前应引用哪一组结果。历史文档保留当时的代码、协议和结论，不应用其中较旧的数字替代当前正式结果。

> Codec 版本说明：NQRT v3 `e8_upstream_half_coset_tie_v1` 的 W4/W4KV4/W4A4/W4A4KV4 全量矩阵已经完成。NQRT v2 legacy 结果继续保留用于历史复现，不再作为当前首选数字。详见 [v3 完整递进](../results/llama32_1b_paper_hadamard_w4a4kv4_nqrt3_progression.md)和[固定上游对齐报告](../results/fixed_upstream_codec_alignment.md)。

## 当前主协议

- 模型：`meta-llama/Llama-3.2-1B` Base，revision `4e20de362430cd3b72f300e6b0f18e50e7166e08`；
- 数据：`Salesforce/wikitext` / `wikitext-2-raw-v1` test，revision `b08601e04326c79dfdd32d625aee71d232d685c3`；
- 评测：context 2048，141 个完整窗口，288,627 个预测 token；
- 测试 token SHA256：`54927bf19e6e64b4008eeb93370c370fb25db18cf68eac8015cfe094c464d567`；
- 校准：固定的 128 个 WikiText-2 train 窗口，选中 token SHA256 `358a77b403d372a11957df15dbf46ef3e6f54025d315eaf3514c40bd561d90bb`；
- 公平范围：112 个 Llama `Linear` 权重使用 W4，`lm_head` 保持 BF16；
- 质量主协议：W4 从低比特 NQMC 回载后反量化为 BF16 Linear；表中历史 A4/KV4
  组合结果为浮点重构质量仿真。另有独立的 NQRT4/NQA4/NQKV v2 packed runtime
  证据，必须按下表对应报告单独引用，不能与历史 fake-quant 耗时混用。

## 推荐引用结果

| 结论 | 当前值 | 推荐来源 | 证据状态 |
|---|---:|---|---|
| BF16 PPL | 9.758112 | [共享校准递进报告](../results/llama32_1b_shared128_w4a4kv4_progression.md)的同协议基线 | 正式质量基线 |
| NestQuant W4 PPL | 10.211760 | [共享校准递进报告](../results/llama32_1b_shared128_w4a4kv4_progression.md) | 正式质量结果；BF16 `lm_head` |
| NestQuant W4KV4 PPL | 10.668994 | 同上 | 正式质量仿真 |
| NestQuant W4A4 PPL | 10.952032 | 同上 | 正式质量仿真 |
| NestQuant W4A4KV4 PPL | 11.504096 | 同上 | 正式质量仿真 |
| 论文式 Hadamard W4 PPL | 10.917094 | [局部 Hadamard W4 完整报告](../results/llama32_1b_paper_hadamard_w4_full.md) | 同协议普通 W4 为 11.054059；无 LDLQ |
| 论文式 Hadamard + shared128 LDLQ W4 PPL | **10.076886** | [NQRT v3 上游语义端点](../results/llama32_1b_paper_hadamard_shared128_ldlq_nqrt3.md) | 当前首选 W4；112/112 记录为上游 E8 语义；真实 NQMC 回载 |
| 论文式 Hadamard + shared128 LDLQ W4A4 PPL | **10.689190** | [NQRT v3 完整递进](../results/llama32_1b_paper_hadamard_w4a4kv4_nqrt3_progression.md) | 固定上游全局 A beta；局部 `X H` 坐标 fake quant |
| 论文式 Hadamard + shared128 LDLQ W4KV4/W4A4KV4 PPL | **10.515783 / 11.232054** | [NQRT v3 完整递进](../results/llama32_1b_paper_hadamard_w4a4kv4_nqrt3_progression.md) | 同一 v3 NQMC；A4/KV4 为浮点重建质量仿真 |
| A4 beta held-out 选择 | 保留上游默认 | [独立 train held-out 报告](../results/llama32_1b_a4_beta_holdout.md) | 新 module 最优 PPL 10.270711，仍差于默认 9.782117；未读取 test 选配置 |
| QuaRot 兼容移植 GPTQ W4 PPL | 10.508346 | [QuaRot 对照](../results/llama32_1b_quarot_comparison.md) | 同校准/测试兼容移植，不是官方 Llama-3.2 结果 |
| QuaRot 兼容移植 W4A4KV4 PPL | 21.095586 | 同上 | fake-quant 兼容移植 |
| 共享校准 W4 NQMC | 1,044,557,140 B | [共享校准递进报告](../results/llama32_1b_shared128_w4a4kv4_progression.md) | 真实文件和独立回载；不证明低比特执行 |
| QA-LDLQ 修正尺度 W4/W4A4KV4 | 10.283140 / 11.584234 | [QA-LDLQ 报告](../results/llama32_1b_qa_ldlq_ablation.md) | 负结果；未优于普通 LDLQ |
| PIQA BF16/W4/W4A4KV4 `acc_norm` | 0.7481 / 0.7367 / 0.7078 | [下游评测报告](../results/llama32_1b_downstream_full.md) | 1,838 题；协议与逐样本身份校验通过 |
| HellaSwag BF16/W4/W4A4KV4 `acc_norm` | 0.6423 / 0.6282 / 0.5996 | [下游评测报告](../results/llama32_1b_downstream_full.md) | 10,042 题；协议与逐样本身份校验通过 |
| 完整 1B warm CUDA allocation | BF16 2,390.1 MB / NQRT4 packed 1,028.6 MB（-57.0%） | [端到端 runtime 报告](../results/llama32_1b_nqrt4_full_e2e.md) | 112/112 投影 packed 常驻；当前仍慢于 BF16 resident |
| CUDA Graph batch 1/16 | packed eager 40.43/40.34 ms → Graph 16.61/20.90 ms（2.43×/1.93×） | [Graph runner 报告](../results/llama32_1b_nqrt4_graph_runner.md) | 固定 batch decode-forward；32-token greedy 一致；不含服务开销 |
| Continuous batching 服务负载 | `8/4` 相对 `1/0`：steady/burst/bimodal token throughput `+42.1%/+46.2%/+16.2%` | [服务级 engine benchmark](../results/llama32_1b_nqrt4_service_benchmark.md) | 18 次 open-loop 合成 token-ID run；无 tokenizer/网络；不是生产 QPS |
| Continuous batching 容量拐点 | 配对负载约 25～30 req/s；30→50 offered req/s 时 token throughput 121.61→129.10、TTFT P95 335.99→746.94 ms | [服务容量扫描](../results/llama32_1b_nqrt4_service_capacity.md) | 固定长度分布的有限 engine sweep；不是 HTTP QPS 或无限时长稳定速率 |
| Packed/BF16 混合驻留 | 512 MiB 预算相对全 packed：TTFT -42.4%、TPOT -45.8%；相对全 BF16：CUDA allocated -36.3% | [混合驻留 Pareto 报告](../results/llama32_1b_nqrt4_hybrid_pareto.md) | 六个预算点的有限候选 Pareto；贪心启发式，不是全组合最优证明 |
| 真实 packed W4A4 PPL | **10.683118** | [packed W4A4KV4 runtime 报告](../results/llama32_1b_packed_w4a4kv4_runtime.md) | 112/112 投影；完整 WikiText-2；`use_cache=False`，因此不含 KV4 |
| 完整 1B packed W4A4KV4 生成 | 112 W4/A4 投影、16 NQKV 层、标准 `model.generate()` 生成 3 token | 同上 | correctness-first smoke；不证明相对 BF16 的 TPOT 加速或生产 serving |
| CUDA M16 W4A4 prefill | 三轮中位数 178.93 → 117.17 ms，**1.527×**（1.518～1.544×） | [CUDA M16 完整模型报告](../results/llama32_1b_w4a4_cuda_mma_m16.md) | 每轮 1,232 次自定义内核命中；显式 opt-in；最终 logits 不等价，不能声称质量无损 |
| CUDA M1 W4A4 decode | TPOT 三轮中位数 170.01 → 94.53 ms，**1.803×**（1.790～1.815×） | [CUDA M1 完整模型报告](../results/llama32_1b_w4a4_cuda_gemv_m1.md) | M1-only 单因素实验；每轮 896 次 M1 命中；prefill logits 相等，生成随后分叉 |
| CUDA staged W4A4 M32/M64/M128 | q-proj 相对 tuned Triton **1.642× / 1.909× / 1.503×**；完整 1B seq32/64 prefill **1.464× / 1.603×** | [large-M kernel 报告](../results/w4a4_large_m_multiwarp_probe.md) | M64 512-window PPL +0.170%、评测 1.557×；但 logits/greedy token 不等价；不是原生 INT4 Tensor Core GEMM |
| CUDA chunked W4A4 任意 M | 完整 1B seq40/120 prefill **1.453× / 1.394×** | [动态 M 报告](../results/w4a4_dynamic_m_chunked_staged.md) | 真实服务分布 prefill 固定 family 仅覆盖 7.62%；M128 分块+尾块 padding 覆盖全部观测 M，额外行数 11.19%；logits/token 不等价 |
| Packed W4A4 continuous batching | CUDA vs tiled：token throughput **+51.96%**，TTFT/TPOT/E2E P95 **-37.00%/-31.76%/-36.24%** | [动态 M 服务 A/B](../results/w4a4_dynamic_m_chunked_staged.md) | selected 8/4、steady 50 req/s、3 次配对合成 token-ID engine run；48/48 请求完成；不含 tokenizer/网络，不是生产 SLA |
| Packed W4A4 burst/bimodal serving | burst throughput **+61.21%**、TTFT P95 **-40.60%**；bimodal throughput **+49.84%**、TTFT P95 **-37.55%** | [动态 M 服务 A/B](../results/w4a4_dynamic_m_chunked_staged.md) | 每 profile 3 次配对；每 run 48/48 请求完成、0 失败；单进程合成 token-ID engine，不是网络 QPS |
| 完整 1B W4A4KV4 双 CUDA Graph | eager NQKV **8.03** → 初版双 Graph **9.49** → M2 down split-K 三次中位 **11.89 token/s**；TPOT P95 **181.67 ms**；同版本 BF16-KV 为 **14.17 token/s / 151.75 ms** | [W4A4KV4 continuous smoke](../results/llama32_1b_w4a4kv4_continuous_smoke.md)、[M2 split-K](../results/w4a4_m2_down_splitk.md) | 特定 `M2 8192→2048` 下投影启用 split-8；NQKV4 吞吐仍低 16.1%、TPOT 高 19.7%；小型 token-ID smoke，不是生产 SLA或交替配对置信区间 |
| NQKV4 dense prefill | prompt-15 prefill **149.00 → 132.35 ms**；固定边界 workload **22.53 → 24.20 token/s**；相对 BF16 吞吐差距 **16.01% → 9.79%** | [dense prefill 优化](../results/nqkv_dense_prefill_optimization.md) | 仅完整 capacity、空缓存、等长且小于一页；短 prompt 总吞吐有跨 run 波动，不外推到 ragged/跨页/生产负载 |
| NQKV4 ragged sub-page writer | 单层 GPU-only launch **42.01 us / 7.51x**；20 对同进程交替 A/B：吞吐 **12.34 → 13.61 token/s（+9.72%）**，TTFT/TPOT/E2E P95 分别改善 **11.56%/9.41%/9.28%**，四项均 20/20 改善且 bootstrap 95% CI 不跨 0 | [ragged writer](../results/nqkv_ragged_subpage_writer.md)、[paired A/B](../results/nqkv_ragged_writer_paired_ab.md) | 仅连续前缀且不触及页边界；同模型、同 CUDA context、同请求并轮换顺序；token-ID engine，不外推网络生产 SLA |
| NQKV4 单页写满与 GPU pack 融合 | 20 对同进程交替 A/B：吞吐 **18.61 → 21.60 token/s（配对中位 +16.22%）**；TTFT/TPOT/E2E P95 分别改善 **16.16%/13.02%/15.02%**；四项均 20/20 改善，bootstrap 95% CI 下界分别为 **14.41%/14.72%/11.47%/13.42%** | [单页融合 paired A/B](../results/nqkv_full_page_pack_paired_ab.md) | 仅连续前缀、单页内写入且至少一行刚好写满；跨页仍走事务 fallback；token-ID engine，不是通用 GPU allocator 或网络 SLA |
| NQKV4 单边界分段 prefill | 固定 prompt-20 的 20 对同进程交替 A/B：吞吐 **17.40 → 21.11 token/s（配对中位 +22.25%）**；TTFT/TPOT/E2E P95 分别改善 **20.24%/17.37%/19.31%**；四项均 20/20 胜出，bootstrap 95% CI 下界分别为 **20.12%/19.40%/14.45%/17.97%** | [跨页分段 paired A/B](../results/nqkv_cross_page_paired_ab.md) | 仅连续前缀且最多跨一个页边界、第二页必须为部分热尾；直接 HF Cache 保持 fallback，runner 负责提交/失败关闭；不是任意多页 writer 或网络 SLA |
| NQKV4 通用多页 segmented prefill | 1B prompt 32/48/64/80（2～5 页）smoke 全部通过；prompt-64、capacity-2 的 20 对单变量 A/B 中，batched page encode 相对逐槽位 encode：吞吐 **13.03 → 15.94 token/s（+25.12%）**，TTFT/TPOT/E2E P95 分别改善 **22.14%/21.62%/20.89%**；四项均 20/20 改善，CI 均不跨 0 | [批量编码独立 A/B](../results/nqkv_batched_pack_isolated_paired_ab.md)、[组合路径 A/B](../results/nqkv_multi_page_batched_pack_paired_ab.md)；本地原始证据路径：`results/raw/nqkv_multi_page_smoke.json`（不随仓库发布） | 已过 ≥5% 性能门并成为 runner 默认多页路径；仅 1B token-ID engine、固定 prompt-64/capacity-2，不外推网络 SLA 或跨模型规模；旧逐槽位实现保留为 reference/fallback |
| W4A4KV4 HTTP 在线服务 | 真实 1B checkpoint 成功启动；12 请求、100 ms 固定到达、每请求 3 token 的 NDJSON open-loop smoke：12/12 完成、0 失败/过载，**20.24 token/s**，TTFT P50/P95 **429.58/550.31 ms**，TPOT P50/P95 **58.34/66.01 ms**，E2E P50/P95 **546.86/665.79 ms** | [在线服务说明](online_service.md)、[HTTP smoke](../results/http_service_smoke_v1.md) | 单机单 GPU 小型客户端观测 smoke，包含 tokenizer/socket/排队；已有独立长时与跨页 soak，但仍无鉴权/TLS、多副本或生产 SLA 证据 |
| W4A4KV4 HTTP 容量扫描 | 2/4/8 offered req/s 均 24/24 完成且 0 过载；16 req/s 为 19/24 完成、20.8% 429；32 req/s 为 13/24 完成、45.8% 429；最高无损档 **8 req/s**，峰值完成吞吐 **8.46 req/s / 25.39 token/s**；TTFT P95 在 8→16 req/s 时 **301.53→771.16 ms** | [HTTP capacity sweep](../results/http_capacity_sweep_v1.md) | 每档仅 24 请求的有限本机扫描；定位局部容量拐点，不是长时稳态或生产 SLA |
| W4A4KV4 HTTP 600 秒 soak | 4,800 计划、4,792 完成、8 个 429、0 失败/超时；**7.984 req/s / 23.953 token/s**；TTFT/TPOT/E2E P95 **765.23/65.98/892.65 ms**；60 秒后 RSS range **323,584 B**，CUDA allocated/reserved range **0 B**；终态 inflight/registered/page allocated 全为 0 | [HTTP soak](../results/http_service_soak_600s_v1.md) | 短 prompt 未填满 16-token page，证明服务与 hot-tail 生命周期稳定，不是跨页 cold-page soak 或生产 SLA |
| W4A4KV4 跨页 HTTP soak | 29-token prompt、120 秒、2 offered req/s：**240/240 完成、0 失败/超时/429**；NQKV allocated-page peak **32**（16 层×2 并发），终态 allocated/free **0/192**；TTFT/TPOT/E2E P95 **195.32/60.90/319.10 ms**；RSS range **69,632 B**，CUDA allocated/reserved range **0 B** | [cross-page soak](../results/http_cross_page_soak_120s_v1.md) | 单一低负载的 120 秒生命周期验证；P99 TTFT/E2E 为 1.53/1.64 s，保留长尾；不是饱和容量或生产 SLA |
| W4A4KV4 跨页 HTTP 容量扫描 | 同一 29-token prompt：1/2 req/s 均 24/24 完成、0 过载；4 req/s 为 21/24、12.5% 429；8 req/s 为 15/24、37.5% 429；最高无损档 **2 req/s**，峰值完成吞吐 **2.63 req/s / 7.89 token/s** | [cross-page capacity sweep](../results/http_cross_page_capacity_sweep_v1.md) | 每档 24 请求、同进程趋势；本进程 TPOT P95 174～203 ms，而独立 soak 为 60.90 ms，存在显著跨进程波动，绝对延迟不提升为稳定基线 |
| W4A4KV4 跨进程延迟复现 | 3 个全新服务进程，每轮显式预热后固定 2 req/s、24 请求：72/72 完成、0 失败/过载；TPOT P95 **68.60/64.24/64.55 ms（CV 3.02%）**，TTFT P95 **189.13/174.06/172.68 ms（CV 4.17%）**；测量期均为 P1，SM 时钟均值 2885.7～2888.0 MHz | [fresh-process reproducibility](../results/http_fresh_process_repro_v1.md) | 证明受控协议下跨进程结果可复现，并未复现旧容量扫描的 174～203 ms TPOT；不能倒推出旧异常的唯一根因，也不是生产 SLA |
| W4A4KV4 受控跨页容量复测 | 固定显式预热、12 client workers、24 请求：2/4/6 req/s 均 24/24 无损；8/10/12 req/s 分别有 1/3/5 个 429；最高无损档 **6 req/s**，首个过载档 **8 req/s**，峰值完成吞吐 **6.36 req/s / 19.09 token/s** | [controlled capacity sweep](../results/http_cross_page_capacity_repro_v2.md) | 每档只有 24 请求；client workers 必须高于服务 max-inflight 才不会由客户端排队掩盖背压；是单机局部容量拐点，不是长时 SLA |
| W4A4KV4 Nsight 2→4 req/s A/B | 完成吞吐 **2.03→3.91 req/s（1.92x）**；TPOT P95 **69.40→70.80 ms（1.02x）**，TTFT P95 **182.97→458.67 ms（2.51x）**；GPU 利用率 **47.1%→70.3%**；CUDA Graph 与 kernel launch/request 分别降至 **0.83x/0.84x** | [Nsight offered-load A/B](../results/http_nsys_cross_page_2v4_ab_v1.md) | WSL Nsight 得到 CUDA API/OS runtime，但没有 GPU kernel timeline；可判断排队与批处理摊销，不能声称具体 kernel duration/stall 根因 |
| W4A4KV4 服务端 TTFT 分解 | 2 req/s 排队仅占服务端 TTFT 均值 **0.2%**；4/6 req/s 升至 **45.1%/43.7%**。4 req/s 的 command/scheduler queue P95 为 **146.69/126.97 ms**，而 prefill P95 稳定为 **170.50 ms**；客户端与服务端 TTFT P95 仅差 **2.32 ms** | [server-side phase decomposition](../results/http_phase_timing_2_4_6rps_v1.md) | 每档一次全新进程、24 请求；均值阶段可加和，来自不同请求的各阶段 P95 不可直接相加；是归因基线，不是生产 SLA |
| W4A4KV4 admission policy A/B | 4 req/s、每策略 3 个全新进程：立即准入使 scheduler waiting 均值中位数 **-56.1%**，但 command queue **+63.2%**、TPOT P95 **+123.2%**、E2E P95 **+33.5%**、吞吐 **-3.1%**；保留默认 `min_prefill_batch=2,max_wait_steps=2` | [admission-policy A/B](../results/http_admission_policy_4rps_ab_v1.md) | 两策略各3轮但按顺序运行、不是交替配对；`max_wait_steps=1` 单轮探针同样显著退化，因此未扩展重复数 |
| M1 down split-K 性能门槛 | 4 req/s、10 对交错 fresh-process A/B、480/480 请求成功：吞吐 **3.833→3.862 req/s（+0.77%）**，TTFT/TPOT/E2E P95 **-12.48%/-5.50%/-15.19%**；分别有 8/10、9/10、7/10、9/10 对改善 | [split-K 服务 A/B](../results/w4a4_m1_down_splitk_service_ab_v1.md)、[10 对波动分析](../results/w4a4_m1_down_splitk_variance_analysis_v2.md) | 仅提升精确 `M=1,K=8192,N=2048` split-8 路径；性能通过，但不能单独决定默认策略；单机单 GPU、小请求样本，不是生产 SLA |
| M1 down split-K 自回归差异 | 相同 HTTP workload SHA-256 下，12/12 请求的 off 输出均为 `482,15704,13`，on 输出均为 `482,220,679`；runner 层精确复现，首次分歧时输入相同、baseline top-1/top-2 margin **0.25**、最大 logit 误差 **1.71875**；on 的 HTTP TPOT P95 仍改善 **16.69%** | [HTTP 正确性报告](../results/w4a4_m1_down_splitk_http_correctness_v1.md)、[logit 诊断](../results/w4a4_m1_down_splitk_logit_diagnosis_v1.md) | 证明自回归输出不等价，因此不声称 token 无损；后续正式 PPL 非劣性门通过后，服务按质量契约默认启用，并保留 `--disable-m1-down-splitk` 回滚 |
| M1 down split-K 层敏感度 | 16 个 down-projection 逐层 Graph sweep：第 10 层单独启用即使第二 token **15704→220**，最大 logit 误差 **1.53125**；第 14 层误差 **1.328125** 但 token 未翻转；仅第 10 层回退、其余 15 层启用时固定三 token 恢复为 `482,15704,13` | [层敏感度报告](../results/w4a4_m1_down_splitk_layer_sensitivity_v1.md) | 单 prompt 定位，不证明第 10 层是通用敏感层；该阶段只用于构造后续多 prompt 与正式质量协议，不作为当前默认策略依据 |
| M1 down split-K 多 prompt 筛选 | 8 个不同主题/长度 prompt、每个 3 个 greedy token：全 16 层 split-K 仅 **3/8** token 序列一致；只回退第 10 层为 **5/8**；逐失败 prompt 定位后回退 `{1,10,11,14}`、其余 12 层 split-K 达到 **8/8**；显式 fallback HTTP smoke **8/8 完成、0 失败**且 token 全为 `482,15704,13` | [多 prompt 报告](../results/w4a4_m1_down_splitk_prompt_suite_v1.md)、[HTTP smoke](../results/http_m1_splitk_hybrid_smoke_v1.md) | 手选小样本且 logits 仍可不同；证明敏感层依赖输入，也证明静态四层 fallback 值得扩大筛选，但不构成默认启用或性能收益证据 |
| M1 down split-K token-ID 反例筛选 | 固定 seed、长度 2/4/8/16/32/64 各 2 个输入、每个 3 token：全层 split-K **6/12** 一致，四层 fallback **8/12**；`len2_sample1` 中全层策略一致但 fallback 反而翻转 | [token-ID 报告](../results/w4a4_m1_down_splitk_token_screen_v1.md) | 否决静态层黑名单：层误差会抵消或叠加，回退更多层不保证单调更安全；当前默认策略采用完整已验证形状和正式平均质量契约，不采用黑名单 |
| M1 split-K 数值稳定性基线 | split-2/4/8 隔离加速 **1.83x/2.38x/2.60x**；raw-FP32 最大误差均约 **3e-6 mean / 4e-6 worst**；暴露 partial 后，FP64/Kahan/pairwise reducer 仍约 **2.9e-6**，相对原 reducer 最多仅差约 **9.54e-7** | [数值稳定性报告](../results/w4a4_m1_splitk_numeric_stability_v1.md) | 降低 split 数无显著改善；最终 reduction 不是主因，误差主要已在各 partition 独立累加及丢失连续舍入状态时产生，下一步需改分区内部累加 |
| M1 down split-K 混合长度门槛 | prompt 4～32、output 2～8 的 4 对 deterministic fresh-process engine A/B：token throughput **4/4 改善、配对中位 +0.62%**；TPOT P95 **3/4 改善、中位 -9.17%**；decode-only step 均值 **4/4 改善、中位 -15.90%**；E2E P95 仅 2/4 改善、中位 **+1.90%**；一对哈希版 engine 输入/输出一致 | [mixed-length A/B](../results/w4a4_m1_down_splitk_mixed_ab_v1.md)、[饱和探针](../results/w4a4_m1_down_splitk_mixed_saturation_probe_v1.md) | 局部 engine 哈希不能证明 HTTP 自回归等价；该限制仍保留。默认启用依据是后续 10 对 HTTP 性能 A/B 与正式 PPL 非劣性门，而不是本项局部哈希 |
| W4A4KV4 固定 capacity 2→4 探针 | 4 req/s 下吞吐 **3.90→3.08 req/s**，TTFT/TPOT P95 **439.20/65.53→1640.80/331.84 ms**；prefill 均值 **165.51→395.13 ms**，显存峰值约 **2552→2751 MiB** | [fixed-capacity probe](../results/http_capacity2_vs4_probe_v1.md) | 单轮方向探针；当前 Graph 固定执行 M4，active mask 不跳过非活跃 Linear 行。否决直接扩大固定容量，不代表可变 M Graph Pool 无效 |

服务 workload 的逐要求完成证据见[服务级 benchmark 验收矩阵](service_benchmark_acceptance.md)；
Gate 4 的混合驻留证据见[正式 Pareto 报告](../results/llama32_1b_nqrt4_hybrid_pareto.md)。

BF16 的 9.758112 来自统一 WSL/CUDA 基线；早期 9.757045 和 QuaRot 兼容路径的 9.762884 属于不同运行路径。它们可以用于各自历史消融，但不能在当前主表中互相替换。

## 2026-11-04 split-K 质量门与发布验证

| 项目 | 当前结果 | 证据边界 |
|---|---|---|
| M1 split-K teacher-forced 预筛 | 32 个上下文 × 32 个 target token，共 1,024 个配对 decode 决策；相对 PPL `-0.3539%`，context-bootstrap 95% CI `[-5.4196%, +4.5418%]`，重放最大 logit 误差 0 | 尚未观察到平均质量退化，但置信区间上界超过预声明的 1% 非劣性阈值，因此不能通过质量门；split-K 继续默认关闭。见 [teacher-forced 报告](../results/w4a4_m1_splitk_teacher_forced_quality_v1.md) |
| 2048-context 正式质量门 | **passed**；2,048 contexts、65,536 个配对 decode 决策；相对 PPL **-0.0932%**，20,000 次 context-bootstrap 95% CI **[-0.5664%, +0.3658%]**，argmax agreement **77.774%**，重放最大 logit 误差 0 | CI 上界低于预声明的 +1% margin，支持 sampled teacher-forced M1 decode 非劣性；不证明自由生成 token 一致、完整语义质量或跨模型泛化 |
| 正式工作区回归 | 434 passed、64 subtests passed | WSL/CUDA 当前工作区；包含服务 split-K 默认策略、显式 rollback 和发布证据拒绝测试，以及本机 CUTLASS CuTe headers 可用时的条件测试 |
| 隔离发布树 | 432 passed、2 skipped、62 subtests passed；wheel、sdist→wheel、独立安装、七个 console entry point 和 CUDA runtime 源文件检查通过 | 两个 skip 均为干净副本没有外部 CUTLASS CuTe headers，不是失败或测试文件遗漏；新质量验证命令已从安装后的 wheel 执行 `--help` |
| 仓库审计 | 608 个可发布文件、0 finding | 检查本机绝对路径、敏感内容和不应发布产物；不替代真实 GitHub clone 后的最终 smoke |

## 历史但仍有解释价值的结果

| 结果 | 用途 | 为什么不作为当前主结果 |
|---|---|---|
| 首轮 BF16/RTN/NestQuant W4 | 展示项目从 RTN 到 E8 的早期验证 | 尚未使用 shared128 LDLQ 和统一 `lm_head` 范围 |
| 普通 W4 NQMC 659,279,649 B | 验证第一版模型容器、码率和回载 | 权重算法与当前 shared128 block-LDLQ 不同 |
| train64 block-LDLQ W4 10.276133 | 展示校准规模消融与独立回载 | 当前公平协议使用 shared128 和 BF16 `lm_head` |
| 两个 seed 的逐 Linear 旋转消融 | 证明项目定义的旋转可与 LDLQ 叠加 | 不是论文所有运行时旋转语义的逐项复现 |
| QA-LDLQ `C=155` 退化结果 | 展示 Hessian 尺度错误的定位过程 | 已由尺度审计解释，不能作为最终 QA 配置 |

## 下游评测解释

PIQA/HellaSwag 完整评测已于 2026-10-24 完成。W4 相对 BF16 的 `acc_norm` 分别下降 1.1425/1.4141 个百分点；W4A4KV4 分别下降 4.0261/4.2721 个百分点。W4A4KV4 用时 15,633.48 秒，而 BF16/W4 分别为 257.46/264.34 秒；该差异反映当前 Python/PyTorch fake-quant 参考实现的重复量化、解码和 kernel launch 成本，不代表生产低比特 kernel 的性能。

当前三档结果只能说明 A4 与 KV4 联合启用后的增量退化，不能把全部差值归因于 A4 或 KV4 单项。若要分离来源，需要在相同逐样本协议下补 W4A4 与 W4KV4 消融。

## 表述边界

- 可以陈述真实 NQMC 文件字节数、哈希、独立回载和 PPL 一致性；
- 可以陈述论文式 `W H`/`X H` 已完成 112 层、自包含 NQMC、独立回载和完整 288,627-token PPL；同协议无 LDLQ W4 中 PPL 从 11.054059 降至 10.917094；
- 可以陈述上游语义 NQRT v3 的论文式 Hadamard + shared128 LDLQ 组合 PPL 为 10.076886；历史 legacy 四因素消融仍支持两项收益近似可加；
- 可以陈述 NQRT v3 组合检查点接入局部坐标 A4 后 PPL 为 10.689190；A4 relative squared error 为 0.6390%，overload 为 5.7126%；
- 可以陈述固定协议下的 PPL 与消融；
- 可以陈述 PIQA/HellaSwag 全量 zero-shot 下游质量和逐题正确性翻转；PIQA 不是论文 zero-shot 主表任务，不能写成论文主表复现；
- 历史质量主表中的 A4/KV4 数字仍来自浮点重构仿真；但独立 runtime 已实现
  NQA4 packed 输入和 NQKV v2 packed GPU page。必须分别引用，不能用 runtime smoke
  替换质量主表，也不能把 payload 减少写成端到端显存或 TPOT 加速；
- 可以陈述完整 1B sequence-16 的 CUDA M16 W4A4 prefill 相对 tiled Triton
  三轮加速中位数 1.527×，但必须同时说明该项 M16 隔离实验使用 BF16 Tensor Core MMA、当时 M1 自动回退，且多层
  累加差异导致最终 logits 分叉，因此不是 INT4 GEMM 或质量无损结果；
- 可以陈述严格 M1-only 单因素实验把完整 1B decode TPOT 从 170.01 ms 降至
  94.53 ms，三轮加速中位数 1.803×；必须同时说明它是 CUDA Core GEMV、不是
  Tensor Core/INT4 GEMM，并且生成序列会在后续 token 分叉；
- 已有 packed NQRT 解码融合 BF16 Tensor Core dot 的真实 block 原型；100 次同协议测量中它相对标量 packed block 加速 1.57～1.73×，但仍慢于 BF16 常驻 block，不能声称 INT4 GEMM、端到端模型加速或生产吞吐提升；
- 长 prefill 的大 M tile 分派在 seq 32/64/128 上相对标量 packed block 加速 2.31×/3.80×/6.16×，同时仍比 BF16 常驻 block 慢 21.9%/26.7%/45.3%；可以陈述 packed runtime 优化收益，不能改写为 BF16 加速收益；
- profiler 与无语义编译期消融将剩余瓶颈定位到 fused NQRT kernel 内的 stateless RNG/E8 重建：七个 kernel 占 GPU 自耗时 79%～85%，移除 RNG 与两次 E8 投影可减少代表性投影 61%～76% 的 kernel 时间；该消融输出无效，只能作为性能归因，不能作为模型结果；
- 可以陈述完整 1B packed 权重路径的 warm CUDA allocation 下降 57.0%，但必须同时说明它慢于 BF16 resident；
- 可以陈述固定 batch CUDA Graph 相对 packed eager 的 2.43×/1.93× decode-forward 加速，但不能外推为 continuous batching、网络服务 QPS 或相对 BF16 加速；
- 可以陈述固定本机 workload 下 `8/4` admission 相对即时 `1/0` 的 engine 吞吐与尾延迟改善，但不能改写成 HTTP QPS、生产 SLA 或跨硬件通用最优配置；
- QuaRot 数字来自有明确补丁的兼容移植，不能写成官方发布的 Llama-3.2-1B 结果；
- 当前实现覆盖论文核心模块，但仍存在 beta 来源、非 2 次幂/分布式旋转语义、
  M2 及更宽 M tile、CUDA W4A4 质量协议和生产运行时方面的差距，不能写成全文逐项复现完成。

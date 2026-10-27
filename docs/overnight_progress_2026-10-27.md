# 2026-10-27 夜间推进报告

## 范围

用户选择路线 A：完整服务级 engine benchmark。遇到改变项目重心的路线选择时暂不执行，
继续完成有明确最优解的测试、证据、发布和复现工作。

## 已完成工程实现

1. 新增 `nestquant/runtime/service_benchmark.py`：确定性 fixed/Poisson/burst arrival、uniform/bimodal
   长度生成、插值百分位、跨 repetition 汇总和 paired seed；
2. 新增 `scripts/benchmark_service_workloads.py`：逐请求 TTFT/TPOT/E2E、arrival lag、queue
   depth、request/token throughput、完成率、CUDA peak allocated/reserved、最慢 step；
3. 新增版本化配置 `service_benchmark.v1.json` 与 `service_capacity_sweep.v1.json`；
4. 新增 `validate_service_benchmark.py`：硬性检查实验矩阵、完成率、失败/残留、百分位次序和
   prompt-width 预热覆盖；
5. continuous scheduler 增加 terminal request forget，长时间服务不永久保留历史请求对象；
6. 发布审计增加 Markdown 本地链接存在性和仓库边界检查；
7. 新增隔离发布树 smoke：构建 source wheel 与 sdist→wheel、运行全测试、验证 console
   entry points 和 CUDA/C++ package data、从 wheel 安装目录全新编译并运行 CUDA E8；
8. 新增 benchmark evidence promotion：只有原始报告通过验收、且 profile/policy/repetition 与指定
   配置一致时才生成精简摘要；摘要记录 raw/config 两份文件的 SHA-256；
9. 强化 benchmark validator：逐项检查 run ID、profile×policy×repetition 唯一完整性、失败、
   拒绝、终态残留、吞吐/显存数值和延迟分位数，并输出显式请求计数；
10. 新增 `verify_service_evidence.py` 和安装后命令 `nestquant-verify-service-evidence`：重新计算
    raw/config 哈希和完整摘要，任何原始数据、配置或摘要漂移都会拒绝通过；
11. 新增 `docs/service_benchmark_acceptance.md`，将固定速率、突发、混合长度、三类延迟分位数、
    吞吐、失败率、显存、soak、证据追溯和发布包逐项映射到直接证据；
12. 收紧 workload 配置校验：拒绝布尔/小数 request count、burst size、cache length，拒绝
    非有限或越界 bimodal fraction，统计聚合拒绝 NaN/Inf，避免静默截断或失真指标。

## 正式实验

### 10 分钟 GPU soak

- 17,264/17,264 请求完成；
- 86,348 token，143.79 token/s；
- 368/368 finite-logit checks；
- 排空后 scheduler registry 为 0；
- CUDA allocated 采样范围 512 bytes；
- 报告：`results/llama32_1b_nqrt4_continuous_batch_soak.md`。

### 三类服务负载，8/4 与 1/0

steady 50 req/s、16×3 burst、Poisson 20 req/s bimodal，各策略各 3 次，共 18 run、864
请求全部完成。`8/4` 相对 `1/0`：

| workload | token throughput | TTFT P95 | E2E P95 |
|---|---:|---:|---:|
| steady | +42.1% | -47.5% | -47.8% |
| burst | +46.2% | -51.0% | -51.3% |
| bimodal | +16.2% | -33.8% | -44.1% |

报告：`results/llama32_1b_nqrt4_service_benchmark.md`。

### 配对容量扫描

4/10/20/30/40/50 req/s，每档 3 次，同一 repetition 请求内容完全一致。30→50 offered
req/s 时 token throughput 121.61→129.10 token/s，TTFT P95 335.99→746.94 ms；饱和
拐点约 25～30 req/s。

报告：`results/llama32_1b_nqrt4_service_capacity.md`。

## 发现并修复的问题

1. BF16 并列最大 logits 会让不同 argmax 路径选择不同 token：验收改为同时记录严格序列
   相等和 chosen-token max-logit gap，只有 gap=0 才接受并列；
2. 首次 query width 初始化产生约 2.2 秒冷形状停顿：诊断 JSON 保留，正式 steady-state
   结果预热 4～48 全部 prompt width；
3. 第一版容量扫描各速率使用不同 seed，输出长度混淆吞吐：正式版启用 paired workload，
   只改变 arrival interval；
4. wheel 未包含 CUDA/C++ 源码：`pyproject.toml` 增加 runtime package-data，并通过安装包内
   全新 CUDA 编译/运行验证；
5. 首次安装包验证错误地从 clean source 导入：改到 `/tmp` 执行并硬断言模块位于
   `install_dir`，消除 smoke 假阳性；
6. evidence promotion 测试曾跨测试模块导入 fixture，干净 pytest collection 无法找到 `tests`
   包：改为测试文件自含最小样例，并增加错误配置拒绝用例。

## 验证状态

- 全仓：241 passed + 25 subtests；
- unified runtime：CPU/CUDA E8/J2/current-stream/full-model Graph 全部通过；
- Graph smoke：token 一致，本轮 16.45 ms vs eager 43.02 ms（只作 smoke）；
- repository audit：敏感项、绝对路径、大文件和 Markdown 本地链接 0 findings；
- `git diff --check`：通过，仅有行尾转换提示；
- isolated publishable tree：source wheel、sdist→wheel、241 tests、console、安装资源和 CUDA E8 fresh compile/run 通过。
- 两份正式 benchmark validation 均为 864/864 完成、failed/rejected/retained 全部为 0；
- 服务报告已经显式列出 TTFT/TPOT/E2E 的 P50/P95/P99，原始逐 run 数据保留在 JSON。

## 当前成熟度

| 方向 | 当前判断 |
|---|---:|
| packed CUDA runtime | 80% |
| CUDA Graph decode | 90% |
| 在线推理 engine 原型 | 78% |
| GitHub 发布准备 | 90% |
| 简历项目 | 90%～92% |

## 明早需要选择

完整选项和理由见 `docs/next_route_decisions.md`。首要选择：

- A：packed/BF16 混合权重缓存 Pareto（推荐，直接解决已证实的延迟瓶颈）；
- B：先做 HTTP/JSON 网络服务；
- C：停止新增 runtime 功能，转 GitHub 提交整理与面试交付。

在用户选择前，不擅自进入上述任何会改变项目重心的路线。

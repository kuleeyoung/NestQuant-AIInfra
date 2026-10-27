# 统一复现工作流

## 目标

`nestquant-reproduce` 将分散的实验脚本组织为一个配置驱动、可预检、可断点续跑、带危险操作保护的入口。版本化配置位于 [`configs/reproduction.v1.json`](../configs/reproduction.v1.json)，执行器位于 [`nestquant/experiments/reproduction.py`](../nestquant/experiments/reproduction.py)。

它解决三个问题：

1. 外部使用者可以先看到完整步骤和命令，不需要从多个文档猜测顺序；
2. smoke 与完整 Llama 实验分开，普通检查不会意外触发耗时量化；
3. 结果可以自动核对完成状态、PPL、校准哈希和测试哈希。

## 阶段

| 阶段 | 做什么 | 是否运行大模型 |
|---|---|---:|
| `plan` | 检查环境并展开 smoke/full 的实际命令、输出路径和已有产物状态 | 否 |
| `preflight` | 检查 Python、PyTorch、CUDA、GPU 和必要配置文件 | 否 |
| `smoke` | 运行全部单元测试和确定性的 8×8 toy 矩阵乘积 | 否 |
| `validate` | 校验审核版摘要；可选反查本地 raw manifest | 否 |
| `full` | Hessian → W4 NQMC → W4/W4KV4/W4A4/W4A4KV4 | 是 |
| `all` | preflight、smoke、full 和本轮生成结果校验 | 是 |

## 推荐使用顺序

以下命令在 WSL2 的项目虚拟环境中执行：

```bash
cd /path/to/NestQuant-AIInfra
source .venv/bin/activate
```

如果虚拟环境不在仓库的 `.venv/` 中，先激活已有环境即可。夜间评测脚本也支持通过
`NESTQUANT_VENV_ROOT=/path/to/venv` 指定环境位置，不需要修改脚本源码。

第一步只查看计划：

```bash
python scripts/run_reproduction.py --stage plan
```

第二步运行低成本 smoke：

```bash
python scripts/run_reproduction.py --stage smoke
```

第三步校验 Git 中的审核版结果；如果本机保留了 `results/raw/`，同时强制核对原始结果：

```bash
python scripts/run_reproduction.py --stage validate --require-raw-evidence
```

查看完整实验将执行什么，但不执行：

```bash
python scripts/run_reproduction.py --stage full --dry-run
```

只有明确需要从现有产物继续完整复现时，才执行：

```bash
python scripts/run_reproduction.py \
  --stage all \
  --confirm-full RUN_FULL_LLAMA_REPRODUCTION
```

## 断点续跑规则

- 每个 full step 都声明 `expected_outputs`；全部存在时默认标记为 `skipped_existing`。
- Hessian 收集器还会检查 `.pt` 与 `.pt.json` 是否成对存在。
- `--force` 会忽略执行器的产物跳过逻辑，应只在明确需要重建时使用。
- 工作流报告默认写入 `results/raw/reproduction_run.json`，该目录不提交 Git。
- full 和 all 在真实执行时必须检测到 CUDA，并要求准确输入确认字符串。

## 完整流程的六个步骤

1. 使用版本化的 128-window manifest 收集 16 层和 `lm_head` Hessian；
2. 保留 `lm_head` BF16，以 block-LDLQ 量化其余 112 个 Linear 并导出 NQMC；
3. 从同一 NQMC 独立评测 W4；
4. 从同一 NQMC 评测 W4KV4；
5. 从同一 NQMC 评测 W4A4；
6. 从同一 NQMC 评测 W4A4KV4。

`all` 完成后会对四个新 manifest 检查：

- `status == completed`；
- PPL 与审核版结果一致；
- test token SHA256 一致；
- calibration token SHA256 一致。

## 证据边界

统一入口提高的是复现可靠性和工程可读性，不会改变实验性质。W4 NQMC 是真实低比特权重文件；A4/KV4 仍是浮点重构后的 fake quant；当前流程不产生低比特 kernel 延迟或吞吐证据。

## Packed serving engine benchmark

服务负载评测独立于质量复现入口，需要本机 NQRT4 checkpoint 和 CUDA。以下命令按版本化
配置运行 steady、burst、Poisson bimodal 三类 open-loop workload，并在计时前预热全部
配置 prompt width：

```bash
python scripts/benchmark_service_workloads.py \
  results/raw/llama32_1b_paper_hadamard_w4_shared128_ldlq_nqrt4.nqmc \
  --config configs/service_benchmark.v1.json \
  --offline \
  --output results/raw/service_benchmark_v1.json

python scripts/validate_service_benchmark.py \
  results/raw/service_benchmark_v1.json \
  --output results/raw/service_benchmark_v1_validation.json

python scripts/promote_service_benchmark.py \
  results/raw/service_benchmark_v1.json \
  --config configs/service_benchmark.v1.json \
  --output results/llama32_1b_nqrt4_service_benchmark.json

python scripts/verify_service_evidence.py \
  results/raw/service_benchmark_v1.json \
  --config configs/service_benchmark.v1.json \
  --summary results/llama32_1b_nqrt4_service_benchmark.json
```

验收命令会硬性检查 profile/policy/repetition 矩阵、请求完成率、失败、拒绝与残留状态、延迟
百分位次序，以及 steady-state 报告是否预热每个配置支持的 prompt width。最后一条命令会
重新计算 raw/config SHA-256 并逐字段重建摘要，用于发现事后漂移。正式汇总和
证据边界见[服务级 engine benchmark](../results/llama32_1b_nqrt4_service_benchmark.md)。

在同一请求内容下只改变 offered request rate 的容量扫描使用：

```bash
python scripts/benchmark_service_workloads.py \
  results/raw/llama32_1b_paper_hadamard_w4_shared128_ldlq_nqrt4.nqmc \
  --config configs/service_capacity_sweep.v1.json \
  --offline \
  --output results/raw/service_capacity_sweep_v1.json

python scripts/validate_service_benchmark.py \
  results/raw/service_capacity_sweep_v1.json \
  --output results/raw/service_capacity_sweep_v1_validation.json

python scripts/promote_service_benchmark.py \
  results/raw/service_capacity_sweep_v1.json \
  --config configs/service_capacity_sweep.v1.json \
  --output results/llama32_1b_nqrt4_service_capacity.json

python scripts/verify_service_evidence.py \
  results/raw/service_capacity_sweep_v1.json \
  --config configs/service_capacity_sweep.v1.json \
  --summary results/llama32_1b_nqrt4_service_capacity.json
```

`couple_profile_seeds=true` 保证同一 repetition 的 4～50 req/s 六档只改变 arrival interval，
不会用不同输出长度分布制造虚假的吞吐差异。

## 发布树独立 smoke

以下脚本只复制 Git 会发布的 tracked 与 non-ignored untracked 文件到 `/tmp`，在隔离目录构建
wheel 并运行测试，用来发现对 `results/raw`、本仓库编译缓存或未发布文件的隐式依赖：

```bash
NESTQUANT_PYTHON=/path/to/python bash scripts/smoke_publishable_tree.sh
```

脚本使用 `mktemp` 创建独立目录，不修改当前工作树；输出会打印临时源码目录和 wheel 目录，
便于失败后审计。在 CUDA 发布机上设置 `NESTQUANT_VERIFY_INSTALLED_CUDA=1`，还会从 wheel
安装目录直接编译并运行 CUDA E8 smoke，验证 package-data 路径而不只是文件名。

## 下游任务评测

PIQA/HellaSwag 入口为 `scripts/evaluate_downstream_tasks.py`，固定使用 `lm_eval==0.4.13`。它支持 `bf16`、`w4` 和 `w4a4kv4` 三档；W4从同一个NQMC回载，W4A4KV4为每次harness前向创建独立的量化cache。正式夜间脚本为 `scripts/run_downstream_full_overnight.sh`，按BF16、W4、W4A4KV4顺序运行，并支持根据已完成JSON断点续跑。三档完成后，`scripts/summarize_downstream_results.py`核对协议、数据指纹和逐样本身份，生成机器可读汇总与Markdown报告。

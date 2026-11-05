# GitHub 发布检查清单

## 自动检查

在 WSL 项目环境执行：

```bash
python scripts/audit_repository.py \
  --json-output results/raw/repository_audit.json

python scripts/verify_splitk_quality_evidence.py \
  results/raw/m1_splitk_teacher_forced_2048x32_1pct_gate_v1.json \
  --analysis results/raw/m1_splitk_teacher_forced_2048x32_1pct_analysis_v1.json

NESTQUANT_PYTHON=/path/to/python \
  bash scripts/smoke_publishable_tree.sh

NESTQUANT_PYTHON=/path/to/python \
  bash scripts/smoke_clean_clone.sh

git diff --check
```

Windows PowerShell 使用现有 PyTorch 环境时，可以执行等价入口：

```powershell
.\scripts\smoke_clean_clone.ps1 -Python .\.venv\Scripts\python.exe
```

2026-11-05 最新验证快照：

- 正式工作区：434 passed、64 subtests passed；
- 隔离发布树：432 passed、2 skipped、62 subtests passed；两个 skip 均因隔离目录不携带本机
  CUTLASS CuTe headers；
- wheel、sdist→wheel、独立安装、七个 console entry point 和打包后的 CUDA runtime 源文件通过；
- 仓库审计：590 个可发布文件、0 finding；此前多出的本地 `tmp/` 已从发布清单排除；
- Windows clean clone：临时 commit → `git clone --no-local` → wheel → 独立安装 →
  clone 外导入 → 三个 CPU 示例 → 七个 console entry point 全部通过，结束后 clone 仍为 clean；
- `git diff --check` 通过，仅有 Windows/WSL 行尾转换提醒。

需要 CUDA 安装包 fresh compile 时：

```bash
NESTQUANT_VERIFY_INSTALLED_CUDA=1 \
NESTQUANT_PYTHON=/path/to/python \
  bash scripts/smoke_publishable_tree.sh
```

## 人工检查

- README 顶部在五分钟内能说明问题、贡献、三组核心结果和限制；
- `docs/current_results.md` 是当前正式数字的唯一推荐入口；
- 历史结果没有覆盖当前结果；
- 论文数字明确标为论文证据，本地数字明确标出设备与协议；
- 不提交 `results/raw/`、checkpoint、模型权重、profiler trace、编译缓存和 Hugging Face token；
- README 命令不包含本机用户名或绝对路径；
- 所有简历数字可以点击到正式报告；
- GitHub About/description 不写“生产级”“无损”“INT4 Tensor Core”或“超过 BF16”；
- release tag 前记录测试数、Python/PyTorch/CUDA 和 GPU；
- 从 GitHub 实际 clone 一次，再按 README 执行 CPU smoke。

## 推荐提交拆分

当前工作树的逐文件职责、重叠文件暂存方法和每个提交的验收条件见
[`docs/commit_plan.md`](commit_plan.md)。clean-clone 脚本会在系统临时目录中创建本地
commit 并执行真实 `git clone --no-local`，不会提交或改写当前工作仓库。

当前工作树包含从 kernel、Graph 到 serving 的多个阶段。发布前建议按职责形成可审查提交，而不是
一个超大 commit：

1. `feat: add packed NQRT4 CUDA runtime`
2. `feat: add CUDA Graph and ragged cache runtime`
3. `feat: add continuous batching and service benchmarks`
4. `feat: add packed BF16 hybrid residency policy`
5. `docs: finalize evidence, interview guide, and release checks`

拆分时只使用非破坏性的 `git add <paths>`/`git commit`；不要用 `git reset --hard` 或覆盖用户已有修改。

## 发布边界

当前版本可以称为：单卡 Llama-3.2-1B 的研究型量化与推理 runtime 原型。

当前版本不能称为：论文全量复现、生产 serving、跨模型验证、直接 INT4 Tensor Core GEMM、
所有 M 形状均有专用内核，或 CUDA M16 路径端到端质量无损。

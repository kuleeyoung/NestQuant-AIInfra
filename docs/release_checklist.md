# GitHub 发布检查清单

## 自动检查

在 WSL 项目环境执行：

```bash
python scripts/audit_repository.py \
  --json-output results/raw/repository_audit.json

NESTQUANT_PYTHON=/path/to/python \
  bash scripts/smoke_publishable_tree.sh

git diff --check
```

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

当前版本不能称为：论文全量复现、生产 serving、跨模型验证、packed A/KV runtime 或直接 INT4
Tensor Core GEMM。

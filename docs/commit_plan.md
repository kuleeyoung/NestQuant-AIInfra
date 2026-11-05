# 当前工作树提交计划

这份计划用于把当前跨越 CUDA kernel、KV4、在线服务、质量门禁和发布材料的工作树，整理成可以独立审查、按依赖顺序落地的 Git 提交。计划本身不执行 `commit`，也不改写已有历史。

## 原则

- 每个提交都应通过与其职责相关的单元测试，最终提交再运行全量测试和发布验证。
- 对 `README.md`、`pyproject.toml`、`nestquant/runtime/__init__.py` 等包含多个阶段修改的文件，使用 `git add -p <file>` 按补丁块暂存，不能把整份文件提前塞进某个提交。
- `results/raw/`、模型权重、profiler trace、编译缓存、token 和 `tmp/` 不进入提交。
- 不使用 `git reset --hard`、`git checkout -- <file>` 等可能覆盖现有修改的命令。

## 提交 1：packed W4A4 与 NQKV KV4 基础运行时

建议标题：`feat: add packed W4A4 and paged NQKV runtime`

职责：提交低比特数据格式、页池、写入器、attention 与 CUDA/Triton kernel 的基础实现，不包含 HTTP 服务策略和最终质量结论。

主要范围：

- `nestquant/core/kv_format.py`
- `nestquant/runtime/cuda_w4a4_*.py`
- `nestquant/runtime/nqrt4_w4a4_*`
- `nestquant/runtime/triton_w4a4_linear.py`
- `nestquant/runtime/nqkv_page.py`、`nqkv_pool.py`、`nqkv_hot_cache.py`
- `nestquant/runtime/nqkv_*writer.py`、`nqkv_attention*.py`、`nqkv_segment_plan.py`
- 对应的 core/runtime 单元测试
- `docs/nqkv_format.md`、`docs/nestquant_runtime_semantics.md`

验收：相关单元测试通过，packed 数据不能先完整反量化成 BF16 再冒充 KV4 attention。

## 提交 2：动态 paged KV 与 continuous batching 服务

建议标题：`feat: add dynamic W4A4KV4 continuous batching service`

职责：提交固定地址元数据、批量 slot/page 管理、CUDA Graph runner、模型适配器、在线调度与 HTTP 层。

主要范围：

- `nestquant/runtime/nqkv_batch_manager.py`
- `nestquant/runtime/nqkv_graph_*.py`
- `nestquant/runtime/nqkv_batched_*.py`
- `nestquant/runtime/nqkv_llama_adapter.py`
- `nestquant/runtime/paged_nqkv_cache.py`
- `nestquant/runtime/online_service.py`、`http_service.py`
- `scripts/serve_w4a4kv4.py`、HTTP benchmark/soak/summarizer 脚本
- `configs/service_benchmark_nqkv_*.json`
- 对应服务与调度测试、`docs/online_service.md`

验收：容量 1/2/4/8、混合长度、跨页和 slot 复用测试通过；明确区分测试 oracle 与生产 packed 路径。

## 提交 3：服务 benchmark、profiling 与可复现实验证据

建议标题：`perf: add W4A4KV4 benchmark and profiling protocols`

职责：提交性能测量工具、A/B 协议、聚合器和可公开的 Markdown/JSON 摘要；原始庞大 trace 继续忽略。

主要范围：

- `nestquant/runtime/service_benchmark.py`
- `scripts/benchmark_*`、`scripts/profile_*`、`scripts/capture_*`
- `scripts/summarize_*`
- service benchmark 配置和对应测试
- 不含 split-K 质量门禁的公开 `results/` 报告

验收：报告包含硬件、命令、样本数、warmup、P50/P95/P99、TTFT/TPOT，并保持同系统 BF16 paged-KV A/B 边界。

## 提交 4：M1 split-K 数值优化与质量门禁

建议标题：`perf: quality-gate M1 split-K service default`

职责：提交 down-projection split-K 实现、诊断工具、teacher-forced 评测、统计分析、证据验证器及服务默认策略。

主要范围：

- `nestquant/models/fused_linear.py` 中的 split-K 补丁块
- `scripts/*m1_splitk*`、`scripts/verify_splitk_quality_evidence.py`
- `nestquant/cli/verify_splitk_quality_evidence.py`
- `scripts/serve_w4a4kv4.py` 中的 split-K 默认与回滚开关补丁块
- split-K 单元测试与正式公开报告
- `pyproject.toml` 中 `nestquant-verify-splitk-quality` entry point 补丁块

验收：2,048 contexts / 65,536 decisions 的 1% 非劣性门禁通过；报告保留 argmax 不完全一致的事实，不宣称 token-equivalent。

## 提交 5：发布审计、clean clone 与证据文档

建议标题：`build: finalize release audit and clean-clone verification`

职责：提交仓库审计、wheel/sdist 独立安装、真实 Git clone smoke、README、面试材料和证据边界。

主要范围：

- `scripts/smoke_publishable_tree.sh`
- `scripts/smoke_clean_clone.sh`
- `scripts/audit_repository.py` 及测试
- `docs/release_checklist.md`、本文件及其余最终文档
- `README.md`、`.gitignore`、`pyproject.toml` 的剩余补丁块

验收：全量测试、发布树测试、clean-clone 验证、仓库 audit 和 `git diff --check` 全部通过。

## 实际暂存流程

每个提交依次执行：

```bash
git status --short
git add <该提交独占的文件或目录>
git add -p <跨提交共享的文件>
git diff --cached --stat
git diff --cached --check
python -m pytest <该提交相关测试>
git commit -m "<建议标题>"
```

完成五个提交后再执行：

```bash
python -m pytest -q
NESTQUANT_PYTHON=/path/to/python bash scripts/smoke_publishable_tree.sh
NESTQUANT_PYTHON=/path/to/python bash scripts/smoke_clean_clone.sh
# Windows 主环境可等价执行：
# .\scripts\smoke_clean_clone.ps1 -Python .\.venv\Scripts\python.exe
git status --short
```

若任一步发现依赖必须跨提交才能运行，应把最小依赖前移到同一提交，而不是暂时提交一个已知不可运行的状态。

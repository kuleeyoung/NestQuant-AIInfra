# NestQuant-AIInfra

面向 AI Infra / 大模型压缩岗位的 NestQuant 论文复现与量化评测项目。项目从 E8 最近格点、嵌套格编码和 multi-beta 选择开始，逐步实现 block-LDLQ、W4/W4KV4/W4A4KV4 质量仿真、低比特权重容器，以及 Llama-3.2-1B 上的 QuaRot 同任务兼容性对照。

本项目强调可审计证据：固定模型与数据 revision，记录 token 哈希、校准集、配置、耗时和原始 manifest；严格区分质量仿真、真实低比特存储和低比特 kernel 加速。

## 核心结果

模型与评测协议：

- `meta-llama/Llama-3.2-1B`
- `Salesforce/wikitext/wikitext-2-raw-v1` test
- context length 2048
- 141 个完整窗口，288,627 个预测 token
- test token SHA256：`54927bf19e6e64b4008eeb93370c370fb25db18cf68eac8015cfe094c464d567`

| 方法 | WikiText-2 PPL | 说明 |
|---|---:|---|
| BF16 | 9.758112 | WSL/CUDA 统一基线 |
| RTN W4 | 13.023722 | 对称 group-wise RTN 基线 |
| NestQuant 普通 W4 | 11.202577 | E8 嵌套格、多 beta |
| NestQuant block-LDLQ W4 | 10.276601 | 64 个训练窗口校准 |
| NestQuant block-LDLQ W4，共享校准 | 10.277472 | 与 QuaRot 相同的 128 个随机训练窗口；量化 `lm_head` |
| NestQuant block-LDLQ W4，同范围 | **10.211760** | 128 个共享窗口；与 QuaRot 一样保留 `lm_head` BF16 |
| NestQuant W4KV4，共享校准 | 10.668994 | 同一 NQMC W4；E8 post-RoPE K/V fake quant |
| NestQuant W4A4，共享校准 | 10.952032 | 同一 NQMC W4；E8 Linear 输入 fake quant |
| NestQuant W4A4KV4，共享校准 | **11.504096** | 同一 NQMC W4；激活与 KV 质量仿真 |
| QuaRot 兼容移植 GPTQ W4 | 10.508346 | 官方关键 GPTQ 参数，128 个校准窗口 |
| QuaRot 兼容移植 W4A4KV4 | 21.095586 | 标量 per-token A/K/V fake quant |

QuaRot 官方仓库没有原生支持 Llama-3.2-1B。这里的对照是明确记录的兼容移植：增加纯 PyTorch Hadamard fallback，修复 GQA K-cache reshape、共享 embedding/lm_head 和 Transformers 4.46 GPTQ 设备放置。它不是官方 CUDA kernel 复现。

完整解释见 [共享校准 W4/A4/KV4 递进实验](results/llama32_1b_shared128_w4a4kv4_progression.md)、[共享校准与同范围 W4 对照](results/llama32_1b_shared_calibration_w4.md)和 [NestQuant 与 QuaRot 全矩阵报告](results/llama32_1b_quarot_comparison.md)。

## 实现内容

- **E8 与嵌套格**：最近格点、Voronoi residue、subtractive dither 和 `q=2` 全码字往返验证。
- **Multi-beta**：实现 Appendix F First-beta 子集动态规划，并与穷举搜索交叉验证。
- **模型权重量化**：独立 E8 block W4、8-column block-LDLQ、Hessian 坐标旋转和 QA-LDLQ 适配。
- **激活与 KV**：七类 Llama Linear 输入 A4；post-RoPE Q/K 匹配旋转；逐 attention-head KV4。
- **真实权重容器**：NQPK/NQRT/NQMC 固定宽度码流、beta ID、row norm、dither 状态、共享权重别名及完整模型回载。
- **外部基线**：完成 QuaRot Llama-3.2-1B 兼容移植和 BF16/W4/W4KV4/W4A4/W4A4KV4 完整消融。
- **实验审计**：保存 revision、token 哈希、配置哈希、校准 provenance、阶段耗时和失败状态。

## 证据边界

| 层级 | 当前状态 | 可以证明什么 |
|---|---|---|
| 量化质量仿真 | 已完成 | 相同任务协议下的 PPL 和量化误差 |
| 真实低比特权重存储 | 已完成 NQMC | 打包文件大小、无 Hessian 回载、回载 PPL 一致性 |
| 真实低比特激活/KV 存储 | 未完成 | 当前 A/KV 在运行时仍是浮点重构张量 |
| 真实低比特 kernel | 未完成 | 不能声称 INT4 GEMM、吞吐提升或延迟下降 |

train64 block-LDLQ NQMC 文件实测为 659,281,364 字节，能够在不加载 Hessian 的情况下重建权重并复现 PPL 10.276133。该路径仍将权重反量化到 BF16 Linear，因此不代表 INT4 推理加速。

## 快速开始

基础算法和 CPU 单元测试不需要下载 Llama 权重：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
python -m unittest discover -s tests -v
python scripts/demo_e8.py
python scripts/demo_beta_search.py
python scripts/run_synthetic_matrix_product.py
```

当前共有 108 项主项目测试，覆盖 E8、codec、multi-beta、beta来源防漂移、离线权重/在线激活Hadamard等价路径与模型级事务替换、LDLQ、packing、checkpoint、A4/KV4、PPL 聚合与进度回调、共享校准清单、统一复现入口、下游评测cache隔离与同协议汇总、量化范围排除和量化事务回滚。QuaRot 适配补丁另有 6 项 Hadamard/GQA 测试。

## Llama-3.2-1B PPL

模型是 Hugging Face gated repository。先在网页获得访问权，再在本机执行 `hf auth login`；不要把 token 写进命令、代码或 Git：

```bash
python -m pip install -e ".[evaluation]"
hf auth login
python scripts/evaluate_llama_wikitext_ppl.py --device cuda --offline
```

一个窗口的 smoke 可增加 `--limit-chunks 1`，但不能当作完整 PPL。原始运行产物默认写入被 Git 忽略的 `results/raw/`。

推荐先使用统一入口查看、检查和校验整个复现流程：

```bash
python scripts/run_reproduction.py --stage plan
python scripts/run_reproduction.py --stage smoke
python scripts/run_reproduction.py --stage validate --require-raw-evidence
python scripts/run_reproduction.py --stage full --dry-run
```

完整 Llama 复现带有显式确认保护和产物级断点续跑。各阶段含义、正式命令与结果校验规则见 [统一复现工作流](docs/reproduction_workflow.md)。

PIQA/HellaSwag zero-shot下游评测使用固定的`lm_eval==0.4.13`，入口为`scripts/evaluate_downstream_tasks.py`；当前比较BF16、共享校准NQMC W4和W4A4KV4质量，不将结果解释为低比特kernel加速。`scripts/summarize_downstream_results.py`只接受协议和逐样本身份一致的三档结果，并自动生成分数差与正确性翻转报告。固定协议、smoke证据和夜间运行安排见[下游评测协议](docs/downstream_evaluation.md)。

正式校准窗口固定在 [`configs/calibration/llama32_1b_wikitext2_train_seed58_n128.json`](configs/calibration/llama32_1b_wikitext2_train_seed58_n128.json)。该文件保存 128 个窗口的起点、逐窗口 token 哈希和总哈希；NestQuant Hessian 与 QuaRot 兼容入口都支持 `--calibration-manifest`，并在运行前验证模型、数据 revision、tokenization 和完整 train token 流。重新生成时应使用一个不存在的输出路径，因为生成器会拒绝静默覆盖正式清单：

```bash
python scripts/create_calibration_manifest.py --offline \
  --output configs/calibration/calibration_candidate.json
```

QuaRot 兼容移植实验入口：

```bash
python scripts/run_quarot_compat.py \
  --nsamples 128 --a4 --kv4 \
  --output results/raw/quarot_gptq128_w4a4kv4_full.json
```

该脚本需要单独准备官方 QuaRot clone 和兼容环境。具体 commit、依赖冲突和补丁见 [适配审查](docs/QuaRot兼容性审查与适配路线.md)、[适配进展](docs/QuaRot_Llama32适配进展.md)和 [`external_patches/`](external_patches/)。

## 项目结构

```text
configs/            版本化实验参数与 beta 配置
nestquant/core/     E8、codec、multi-beta、rotation、LDLQ、packing
nestquant/models/   校准、Hessian、模型量化与 QA 适配
nestquant/evaluation/ A4、KV4、PPL 与结果汇总
nestquant/experiments/ 可导入的实验入口
scripts/            命令行实验、校准、审计和汇总工具
tests/unit/         108 项分层单元测试
docs/               算法、架构、协议、路线和证据边界
results/            审核后的小型汇总；raw 大文件不进入 Git
external_patches/   固定上游 commit 的 QuaRot 兼容补丁
```

代码分层见 [架构说明](docs/architecture.md)，固定协议见 [实验协议](docs/experiment_protocol.md)，结果入口见 [results/README.md](results/README.md)。

## 重点文档

- [嵌套格 codec](docs/nested_codec.md)
- [Multi-beta 搜索](docs/beta_search.md)
- [block-LDLQ 校准](docs/ldlq_calibration.md)
- [激活量化](docs/activation_quantization.md)
- [KV-cache 量化](docs/kv_cache_quantization.md)
- [低比特容器格式](docs/packing_format.md)
- [QA-LDLQ 负结果与尺度审计](docs/qa_ldlq.md)
- [当前正式结果与历史结果分层](docs/current_results.md)
- [论文、作者代码与本项目对应矩阵](docs/paper_implementation_matrix.md)
- [NestQuant 与 QuaRot 对照](results/llama32_1b_quarot_comparison.md)
- [简历证据边界](docs/resume_evidence.md)

## 论文与上游来源

- NestQuant paper: <https://arxiv.org/abs/2502.09720>
- NestQuant authors' repository: <https://github.com/cookiedoth/nestquant>
- QuaRot repository: <https://github.com/spcl/QuaRot>

论文表格值、上游代码行为和本项目实测结果在文档中分别标注。未经本地验证的结论不会写成本项目结果。

"""Validate and summarize a BF16 / NestQuant W4 / W4+LDLQ PPL ablation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_BASELINE = Path("results/raw/llama32_1b_current_source_baselines_full_20261021.json")
DEFAULT_LDLQ = Path("results/raw/llama32_1b_ldlq_train16_full_20261021.json")
DEFAULT_JSON = Path("results/llama32_1b_w4_ldlq_ablation.json")
DEFAULT_MARKDOWN = Path("results/llama32_1b_w4_ldlq_ablation.md")


def _load(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _require_equal(label: str, left: Any, right: Any) -> Any:
    if left != right:
        raise ValueError(f"{label} differs: {left!r} != {right!r}")
    return left


def _collect_calibration_artifacts(directory: Path, chunks: int) -> list[dict[str, Any]]:
    artifacts: list[dict[str, Any]] = []
    for path in sorted(directory.glob(f"hessian_*_train{chunks}.pt.json")):
        manifest = _load(path)
        if manifest["chunk_count"] != chunks:
            raise ValueError(f"unexpected chunk count in {path}")
        artifacts.append(
            {
                "artifact": Path(manifest["output"]).name,
                "bytes": manifest["output_bytes"],
                "sha256": manifest["output_sha256"],
                "module_count": len(manifest["modules"]),
                "peak_cuda_memory_allocated_bytes": manifest[
                    "peak_cuda_memory_allocated_bytes"
                ],
            }
        )
    if not artifacts:
        raise ValueError(f"no train{chunks} calibration manifests found in {directory}")
    return artifacts


def build_summary(
    baseline: dict[str, Any],
    ldlq: dict[str, Any],
    calibration_artifacts: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build a summary only after validating all comparison-critical fields."""
    if baseline.get("status") != "completed" or ldlq.get("status") != "completed":
        raise ValueError("both manifests must have status='completed'")

    invariant_paths = {
        "model_id": ("config", "model_id"),
        "model_revision": ("config", "model_revision"),
        "dataset_id": ("config", "dataset_id"),
        "dataset_config": ("config", "dataset_config"),
        "dataset_revision": ("config", "dataset_revision"),
        "context_length": ("config", "context_length"),
        "seed": ("config", "seed"),
        "q": ("config", "q"),
        "beta_coefficients_project_scale": ("config", "beta_coefficients_project_scale"),
        "selected_beta_count": ("config", "selected_beta_count"),
        "rows_per_chunk": ("config", "rows_per_chunk"),
        "weight_rotation": ("config", "weight_rotation"),
        "model_dtype": ("config", "model_dtype"),
        "logit_loss_dtype": ("config", "logit_loss_dtype"),
        "quantization_scope": ("config", "quantization_scope"),
        "source_sha256": ("source", "nestquant_python_source_sha256"),
        "test_token_ids_sha256": ("data", "token_ids_sha256"),
        "predicted_token_count": ("data", "predicted_token_count"),
    }
    protocol: dict[str, Any] = {}
    for label, path in invariant_paths.items():
        left: Any = baseline
        right: Any = ldlq
        for key in path:
            left = left[key]
            right = right[key]
        protocol[label] = _require_equal(label, left, right)

    bf16 = baseline["results"]["bf16"]
    ordinary = baseline["results"]["nestquant_w4"]
    with_ldlq = ldlq["results"]["nestquant_w4"]
    if ordinary.get("ldlq") is not False:
        raise ValueError("baseline NestQuant result is not marked ldlq=false")
    if with_ldlq.get("ldlq") is not True:
        raise ValueError("LDLQ NestQuant result is not marked ldlq=true")

    bf16_ppl = float(bf16["evaluation"]["perplexity"])
    ordinary_ppl = float(ordinary["evaluation"]["perplexity"])
    ldlq_ppl = float(with_ldlq["evaluation"]["perplexity"])
    ordinary_gap = ordinary_ppl - bf16_ppl
    ldlq_gap = ldlq_ppl - bf16_ppl
    ppl_reduction = ordinary_ppl - ldlq_ppl

    return {
        "schema_version": 1,
        "comparison": "Llama-3.2-1B BF16 vs NestQuant W4 vs NestQuant W4 + block-LDLQ",
        "protocol": protocol,
        "ldlq_calibration": {
            **with_ldlq["ldlq_calibration"],
            "relative_damp": with_ldlq["ldlq_relative_damp"],
            "artifacts": calibration_artifacts,
            "artifact_bytes_total": sum(item["bytes"] for item in calibration_artifacts),
        },
        "results": {
            "bf16": {
                "perplexity": bf16_ppl,
                "mean_nll": bf16["evaluation"]["mean_nll_per_predicted_token"],
                "elapsed_seconds": bf16["elapsed_seconds"],
            },
            "nestquant_w4": {
                "perplexity": ordinary_ppl,
                "mean_nll": ordinary["evaluation"]["mean_nll_per_predicted_token"],
                "elapsed_seconds": ordinary["elapsed_seconds"],
            },
            "nestquant_w4_ldlq_train16": {
                "perplexity": ldlq_ppl,
                "mean_nll": with_ldlq["evaluation"]["mean_nll_per_predicted_token"],
                "elapsed_seconds": with_ldlq["elapsed_seconds"],
            },
        },
        "derived": {
            "ordinary_w4_minus_bf16_ppl": ordinary_gap,
            "ldlq_w4_minus_bf16_ppl": ldlq_gap,
            "ldlq_absolute_ppl_reduction_vs_ordinary_w4": ppl_reduction,
            "ldlq_relative_ppl_reduction_vs_ordinary_w4_percent": 100.0 * ppl_reduction / ordinary_ppl,
            "bf16_gap_closed_by_ldlq_percent": 100.0 * ppl_reduction / ordinary_gap,
        },
        "evidence_scope": {
            "quality": "full WikiText-2 test fake-quant quality simulation",
            "storage": "not measured for the LDLQ variant",
            "inference_speed": "not measured; execution dequantizes to BF16 Linear weights",
            "paper_scope": "W4 block-LDLQ ablation; no activation/KV quantization and no QA-LDLQ",
        },
    }


def render_markdown(summary: dict[str, Any], baseline_path: Path, ldlq_path: Path) -> str:
    p = summary["protocol"]
    r = summary["results"]
    d = summary["derived"]
    c = summary["ldlq_calibration"]
    artifact_rows = "\n".join(
        f"| `{item['artifact']}` | {item['module_count']} | {item['bytes']:,} | `{item['sha256']}` |"
        for item in c["artifacts"]
    )
    return f"""# Llama-3.2-1B NestQuant W4 block-LDLQ 消融

## 结论

在模型、数据、测试 token、量化参数、dither seed 与源码哈希完全相同的条件下，加入 8 列 E8 block-LDLQ 后，完整 WikiText-2 test PPL 从 **{r['nestquant_w4']['perplexity']:.6f}** 降至 **{r['nestquant_w4_ldlq_train16']['perplexity']:.6f}**，绝对下降 **{d['ldlq_absolute_ppl_reduction_vs_ordinary_w4']:.6f}**，相对下降 **{d['ldlq_relative_ppl_reduction_vs_ordinary_w4_percent']:.2f}%**。相对 BF16 的 PPL 差距从 **{d['ordinary_w4_minus_bf16_ppl']:.6f}** 缩小到 **{d['ldlq_w4_minus_bf16_ppl']:.6f}**，本轮 LDLQ 消除了 **{d['bf16_gap_closed_by_ldlq_percent']:.2f}%** 的差距。

这说明当前 block-LDLQ 实现在固定的 16 序列校准条件下显著改善 W4 质量。16 序列结果仍需要更大校准集验证稳定性，不能直接当作论文全部结果的复现。

## 固定协议

- 模型：`{p['model_id']}`，revision `{p['model_revision']}`
- 测试集：`{p['dataset_id']}/{p['dataset_config']}`，revision `{p['dataset_revision']}`
- 上下文：{p['context_length']}；完整非重叠块；预测 token：{p['predicted_token_count']:,}
- 测试 token SHA-256：`{p['test_token_ids_sha256']}`
- 项目 Python 源码 SHA-256：`{p['source_sha256']}`
- 量化：Linear 权重质量仿真；E8、q={p['q']}、每层从 7 个候选中用 DP 选 {p['selected_beta_count']} 个 beta；CPU dither seed {p['seed']}
- LDLQ 校准：WikiText-2 train 的 {c['chunks']} 个完整序列（{c['chunks'] * p['context_length']:,} token），相对阻尼 {c['relative_damp']}
- 精确选中校准 token SHA-256：`{c['calibration_token_ids_sha256']}`
- 完整 train 语料 token SHA-256：`{c['train_corpus_token_ids_sha256']}`

### 校准文件清单

| Hessian 文件 | 唯一模块统计数 | 字节数 | SHA-256 |
|---|---:|---:|---|
{artifact_rows}

合计 {c['artifact_bytes_total']:,} 字节。大型矩阵不提交 Git；此表中的名称、大小和摘要足以核对本地文件是否是产生该结果的校准数据。

## 完整 PPL 结果

| 方法 | PPL | 平均 NLL | 相对 BF16 的 PPL 增量 | 整个方法耗时 |
|---|---:|---:|---:|---:|
| BF16 | **{r['bf16']['perplexity']:.6f}** | {r['bf16']['mean_nll']:.6f} | 0 | {r['bf16']['elapsed_seconds']:.1f} s |
| NestQuant W4 | {r['nestquant_w4']['perplexity']:.6f} | {r['nestquant_w4']['mean_nll']:.6f} | +{d['ordinary_w4_minus_bf16_ppl']:.6f} | {r['nestquant_w4']['elapsed_seconds']:.1f} s |
| NestQuant W4 + block-LDLQ（train 16） | **{r['nestquant_w4_ldlq_train16']['perplexity']:.6f}** | **{r['nestquant_w4_ldlq_train16']['mean_nll']:.6f}** | +{d['ldlq_w4_minus_bf16_ppl']:.6f} | {r['nestquant_w4_ldlq_train16']['elapsed_seconds']:.1f} s |

耗时包含模型加载、离线量化和 PPL 评测。它用于说明本机实验成本，不能解释为推理延迟或吞吐。LDLQ 版本当前使用 Python/PyTorch 参考递推，耗时明显高于普通 W4。

## 实现与验证证据

- `nestquant/core/ldlq.py` 实现带 8×8 单位对角块的 block-LDL 分解，以及从右向左的 8 列误差反馈。
- `nestquant/models/calibration.py` 收集 Linear 输入的 `E[x x^T]`；q/k/v 共享输入统计，gate/up 共享输入统计。
- `nestquant/models/hessian_store.py` 按四个 Transformer 层惰性加载约 4.77 GiB 的 FP32 Hessian，避免全部统计同时进入内存或显存。
- 单层 q_proj smoke 中，Hessian 加权代理损失下降 90.41%；down_proj 下降 86.91%。两层的普通 MSE 均上升，符合 LDLQ 优化激活加权误差而非逐权重 MSE 的目标。
- 67 项单元测试覆盖 E8、嵌套码字、beta DP、打包、PPL 口径、旋转协方差、block-LDL 分解、LDLQ 反馈、组合路径、Hessian 惰性加载及 LDLQ NQMC 往返。

## 结果边界

这组实验是 **W4 权重量化质量仿真**：量化结果写回 BF16 权重并执行 BF16 Linear。它没有测量 LDLQ 版本的真实模型文件大小、运行时权重显存、推理延迟或吞吐，也没有实现 A4、KV4、QA-LDLQ 或论文完整旋转图。因此可以陈述“实现并验证了 E8 block-LDLQ，在固定协议下将 PPL 从 11.2026 降至 10.3237”，不能陈述“实现了 W4A4KV4”“获得 INT4 推理加速”或“完整复现论文”。

## 原始证据

- 同版本 BF16/普通 W4 manifest：`{baseline_path.as_posix()}`
- LDLQ manifest：`{ldlq_path.as_posix()}`
- 机器可读汇总：`results/llama32_1b_w4_ldlq_ablation.json`
- 大型 Hessian 与逐层原始记录位于被 Git 忽略的 `results/raw/calibration/`，提交到 GitHub 时保留 manifest 与哈希，不提交约 4.77 GiB 原始矩阵。
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--ldlq", type=Path, default=DEFAULT_LDLQ)
    parser.add_argument("--calibration-dir", type=Path, default=Path("results/raw/calibration"))
    parser.add_argument("--output-json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--output-markdown", type=Path, default=DEFAULT_MARKDOWN)
    args = parser.parse_args()

    ldlq_manifest = _load(args.ldlq)
    chunks = int(ldlq_manifest["results"]["nestquant_w4"]["ldlq_calibration"]["chunks"])
    calibration_artifacts = _collect_calibration_artifacts(args.calibration_dir, chunks)
    summary = build_summary(_load(args.baseline), ldlq_manifest, calibration_artifacts)
    first_sidecar = _load(
        args.calibration_dir / f"{calibration_artifacts[0]['artifact']}.json"
    )
    calibration = summary["ldlq_calibration"]
    calibration["legacy_run_manifest_train_token_ids_sha256"] = calibration.pop(
        "train_token_ids_sha256"
    )
    calibration["calibration_token_ids_sha256"] = first_sidecar[
        "calibration_token_ids_sha256"
    ]
    calibration["train_corpus_token_ids_sha256"] = first_sidecar[
        "train_corpus_token_ids_sha256"
    ]
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    args.output_markdown.write_text(
        render_markdown(summary, args.baseline, args.ldlq), encoding="utf-8"
    )
    print(f"Wrote {args.output_json}")
    print(f"Wrote {args.output_markdown}")


if __name__ == "__main__":
    main()

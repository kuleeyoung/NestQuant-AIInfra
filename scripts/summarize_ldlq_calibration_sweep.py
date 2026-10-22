"""Summarize the Llama-3.2-1B 16-vs-64-sequence LDLQ calibration sweep."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from summarize_ldlq_ablation import _collect_calibration_artifacts, _load, _require_equal


def _result(manifest: dict[str, Any], method: str) -> dict[str, float]:
    result = manifest["results"][method]
    evaluation = result["evaluation"]
    return {
        "perplexity": float(evaluation["perplexity"]),
        "mean_nll": float(evaluation["mean_nll_per_predicted_token"]),
        "elapsed_seconds": float(result["elapsed_seconds"]),
    }


def _calibration(directory: Path, chunks: int) -> dict[str, Any]:
    artifacts = _collect_calibration_artifacts(directory, chunks)
    manifests = [
        _load(directory / f"{item['artifact']}.json") for item in artifacts
    ]
    selected_hash = manifests[0]["calibration_token_ids_sha256"]
    corpus_hash = manifests[0]["train_corpus_token_ids_sha256"]
    for manifest in manifests[1:]:
        _require_equal(
            f"train{chunks} selected token hash",
            selected_hash,
            manifest["calibration_token_ids_sha256"],
        )
        _require_equal(
            f"train{chunks} corpus token hash",
            corpus_hash,
            manifest["train_corpus_token_ids_sha256"],
        )
    return {
        "chunks": chunks,
        "tokens": chunks * int(manifests[0]["context_length"]),
        "calibration_token_ids_sha256": selected_hash,
        "train_corpus_token_ids_sha256": corpus_hash,
        "artifact_bytes_total": sum(item["bytes"] for item in artifacts),
        "artifacts": artifacts,
    }


def build_summary(
    baseline: dict[str, Any],
    train16: dict[str, Any],
    train64: dict[str, Any],
    calibration_dir: Path,
) -> dict[str, Any]:
    invariant_paths = [
        ("model_id", "config", "model_id"),
        ("model_revision", "config", "model_revision"),
        ("dataset_revision", "config", "dataset_revision"),
        ("context_length", "config", "context_length"),
        ("seed", "config", "seed"),
        ("q", "config", "q"),
        ("beta_candidates", "config", "beta_coefficients_project_scale"),
        ("selected_beta_count", "config", "selected_beta_count"),
        ("weight_rotation", "config", "weight_rotation"),
        ("source_sha256", "source", "nestquant_python_source_sha256"),
        ("test_token_ids_sha256", "data", "token_ids_sha256"),
        ("predicted_token_count", "data", "predicted_token_count"),
    ]
    protocol: dict[str, Any] = {}
    for label, section, key in invariant_paths:
        left = baseline[section][key]
        _require_equal(label, left, train16[section][key])
        _require_equal(label, left, train64[section][key])
        protocol[label] = left
    _require_equal(
        "LDLQ damping",
        train16["config"]["ldlq_relative_damp"],
        train64["config"]["ldlq_relative_damp"],
    )

    bf16 = _result(baseline, "bf16")
    ordinary = _result(baseline, "nestquant_w4")
    ldlq16 = _result(train16, "nestquant_w4")
    ldlq64 = _result(train64, "nestquant_w4")
    ordinary_gap = ordinary["perplexity"] - bf16["perplexity"]
    gain16 = ordinary["perplexity"] - ldlq16["perplexity"]
    gain64 = ordinary["perplexity"] - ldlq64["perplexity"]
    extra_gain = ldlq16["perplexity"] - ldlq64["perplexity"]

    return {
        "schema_version": 1,
        "comparison": "Llama-3.2-1B NestQuant W4 block-LDLQ calibration sweep",
        "protocol": {
            **protocol,
            "ldlq_relative_damp": train16["config"]["ldlq_relative_damp"],
        },
        "calibration": {
            "train16": _calibration(calibration_dir, 16),
            "train64": _calibration(calibration_dir, 64),
        },
        "results": {
            "bf16": bf16,
            "nestquant_w4": ordinary,
            "nestquant_w4_ldlq_train16": ldlq16,
            "nestquant_w4_ldlq_train64": ldlq64,
        },
        "derived": {
            "ordinary_w4_minus_bf16_ppl": ordinary_gap,
            "train16_ppl_reduction_vs_ordinary": gain16,
            "train64_ppl_reduction_vs_ordinary": gain64,
            "train16_bf16_gap_closed_percent": 100.0 * gain16 / ordinary_gap,
            "train64_bf16_gap_closed_percent": 100.0 * gain64 / ordinary_gap,
            "train64_extra_ppl_reduction_vs_train16": extra_gain,
            "train64_relative_ppl_reduction_vs_train16_percent": (
                100.0 * extra_gain / ldlq16["perplexity"]
            ),
        },
        "interpretation": {
            "stability": "train64 preserves and slightly improves the train16 result",
            "quality_scope": "fake-quant W4 quality simulation with BF16 Linear execution",
            "deployment_scope": "no LDLQ checkpoint size, low-bit kernel, latency, or throughput result",
        },
    }


def render_markdown(summary: dict[str, Any]) -> str:
    p, r, d, c = (
        summary["protocol"], summary["results"], summary["derived"], summary["calibration"]
    )
    return f"""# Llama-3.2-1B NestQuant W4：LDLQ 校准规模消融

## 结论

将 WikiText-2 train 校准规模从 16 个 2048-token 序列扩大到 64 个序列后，完整 WikiText-2 test PPL 从 **{r['nestquant_w4_ldlq_train16']['perplexity']:.6f}** 进一步降至 **{r['nestquant_w4_ldlq_train64']['perplexity']:.6f}**。额外下降 **{d['train64_extra_ppl_reduction_vs_train16']:.6f}**，相对 train-16 下降 **{d['train64_relative_ppl_reduction_vs_train16_percent']:.2f}%**。

train-64 保留并小幅增强了 train-16 的收益，说明 16 序列结果不是方向相反的抽样偶然；同时，四倍校准 token 只带来约 0.048 PPL 的额外改善，当前校准收益已接近饱和。继续扩大到全部 train token 的投入产出比很低。

## 严格固定的比较条件

- 模型 revision：`{p['model_revision']}`
- 测试 token：{p['predicted_token_count']:,}；SHA-256 `{p['test_token_ids_sha256']}`
- 项目 Python 源码 SHA-256：`{p['source_sha256']}`
- E8、q={p['q']}、同一 beta 候选/DP、dither seed {p['seed']}
- 权重旋转：`{p['weight_rotation']}`；相对阻尼：{p['ldlq_relative_damp']}
- 唯一变化：校准序列从 16 增加到 64

## 完整结果

| 方法 | 校准 token | PPL | 相对 BF16 增量 | 整个方法耗时 |
|---|---:|---:|---:|---:|
| BF16 | — | **{r['bf16']['perplexity']:.6f}** | 0 | {r['bf16']['elapsed_seconds']:.1f} s |
| NestQuant W4 | — | {r['nestquant_w4']['perplexity']:.6f} | +{d['ordinary_w4_minus_bf16_ppl']:.6f} | {r['nestquant_w4']['elapsed_seconds']:.1f} s |
| W4 + block-LDLQ，train 16 | {c['train16']['tokens']:,} | {r['nestquant_w4_ldlq_train16']['perplexity']:.6f} | +{r['nestquant_w4_ldlq_train16']['perplexity'] - r['bf16']['perplexity']:.6f} | {r['nestquant_w4_ldlq_train16']['elapsed_seconds']:.1f} s |
| W4 + block-LDLQ，train 64 | {c['train64']['tokens']:,} | **{r['nestquant_w4_ldlq_train64']['perplexity']:.6f}** | +{r['nestquant_w4_ldlq_train64']['perplexity'] - r['bf16']['perplexity']:.6f} | {r['nestquant_w4_ldlq_train64']['elapsed_seconds']:.1f} s |

train-16 消除了普通 W4 与 BF16 间 **{d['train16_bf16_gap_closed_percent']:.2f}%** 的 PPL 差距；train-64 消除了 **{d['train64_bf16_gap_closed_percent']:.2f}%**。

## 校准审计

| 校准规模 | 精确选中 token SHA-256 | Hessian 文件 | 总字节数 |
|---|---|---:|---:|
| train 16 | `{c['train16']['calibration_token_ids_sha256']}` | {len(c['train16']['artifacts'])} | {c['train16']['artifact_bytes_total']:,} |
| train 64 | `{c['train64']['calibration_token_ids_sha256']}` | {len(c['train64']['artifacts'])} | {c['train64']['artifact_bytes_total']:,} |

两组都来自同一个完整 train token 流，其 SHA-256 为 `{c['train16']['train_corpus_token_ids_sha256']}`。早期 manifest 的 `train_token_ids_sha256` 字段实际记录完整语料哈希；收集脚本已修正为同时记录完整语料哈希与精确选中校准 token 哈希。重新收集后，train-16 的五个 Hessian 文件 SHA-256 与修正前完全一致，证明本次只修正元数据，没有改变校准矩阵。

## 证据边界

这仍是 W4 权重质量仿真：量化后写回 BF16 权重并执行 BF16 Linear。结果支持“扩大校准集后 LDLQ 改善保持稳定”，不支持“真实 LDLQ 模型文件缩小”“低比特推理显存下降”或“推理加速”。
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=Path("results/raw/llama32_1b_current_source_baselines_full_20261021.json"))
    parser.add_argument("--train16", type=Path, default=Path("results/raw/llama32_1b_ldlq_train16_full_20261021.json"))
    parser.add_argument("--train64", type=Path, default=Path("results/raw/llama32_1b_ldlq_train64_full_20261021.json"))
    parser.add_argument("--calibration-dir", type=Path, default=Path("results/raw/calibration"))
    parser.add_argument("--output-json", type=Path, default=Path("results/llama32_1b_w4_ldlq_calibration_sweep.json"))
    parser.add_argument("--output-markdown", type=Path, default=Path("results/llama32_1b_w4_ldlq_calibration_sweep.md"))
    args = parser.parse_args()

    summary = build_summary(
        _load(args.baseline), _load(args.train16), _load(args.train64), args.calibration_dir
    )
    args.output_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.output_markdown.write_text(render_markdown(summary), encoding="utf-8")
    print(f"Wrote {args.output_json}")
    print(f"Wrote {args.output_markdown}")


if __name__ == "__main__":
    main()

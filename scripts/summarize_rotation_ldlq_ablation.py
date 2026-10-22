"""Verify and summarize the same-source W4 rotation/LDLQ ablation."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def measurement(run: dict, method: str, source: Path) -> dict:
    result = run["results"][method]
    evaluation = result["evaluation"]
    return {
        "run_id": run["run_id"],
        "perplexity": evaluation["perplexity"],
        "mean_nll_per_predicted_token": evaluation["mean_nll_per_predicted_token"],
        "predicted_token_count": evaluation["predicted_token_count"],
        "elapsed_seconds": result["elapsed_seconds"],
        "source": str(source),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bf16", type=Path, default=Path(
        "results/raw/llama32_1b_bf16_rotation_ldlq_current_source_20261021.json"))
    parser.add_argument("--ordinary", type=Path, default=Path(
        "results/raw/llama32_1b_no_rotation_current_source_20261021.json"))
    parser.add_argument("--rotation-seed123", type=Path, default=Path(
        "results/raw/llama32_1b_rotation123_current_source_v2_20261021.json"))
    parser.add_argument("--rotation-seed124", type=Path, default=Path(
        "results/raw/llama32_1b_rotation124_current_source_v2_20261021.json"))
    parser.add_argument("--ldlq", type=Path, default=Path(
        "results/raw/llama32_1b_ldlq_train64_current_source_v2_20261021.json"))
    parser.add_argument("--combined-seed123", type=Path, default=Path(
        "results/raw/llama32_1b_rotation123_ldlq_train64_full_20261021.json"))
    parser.add_argument("--combined-seed124", type=Path, default=Path(
        "results/raw/llama32_1b_rotation124_ldlq_train64_full_20261021.json"))
    parser.add_argument("--layer-smoke", type=Path, default=Path(
        "results/raw/layer0_qproj_rotation123_ldlq_train64_smoke.json"))
    parser.add_argument("--output-json", type=Path, default=Path(
        "results/llama32_1b_w4_rotation_ldlq_ablation.json"))
    parser.add_argument("--output-markdown", type=Path, default=Path(
        "results/llama32_1b_w4_rotation_ldlq_ablation.md"))
    args = parser.parse_args()

    paths = {
        "bf16": args.bf16,
        "ordinary_w4": args.ordinary,
        "rotation_seed123": args.rotation_seed123,
        "rotation_seed124": args.rotation_seed124,
        "ldlq_only": args.ldlq,
        "rotation123_plus_ldlq": args.combined_seed123,
        "rotation124_plus_ldlq": args.combined_seed124,
    }
    runs = {name: read_json(path) for name, path in paths.items()}
    for name, run in runs.items():
        if run.get("status") != "completed" or run["config"].get("limit_chunks") is not None:
            raise ValueError(f"{name} must be a completed full evaluation")

    common_config_keys = (
        "model_id", "model_revision", "dataset_id", "dataset_config",
        "dataset_revision", "context_length", "seed", "q",
        "beta_coefficients_project_scale", "selected_beta_count",
        "rows_per_chunk", "model_dtype", "logit_loss_dtype", "quantization_scope",
    )
    reference = runs["ordinary_w4"]
    for name, run in runs.items():
        for key in common_config_keys:
            if run["config"][key] != reference["config"][key]:
                raise ValueError(f"{name} config mismatch: {key}")
        for key in ("text_sha256", "token_ids_sha256", "chunk_count",
                    "predicted_token_count"):
            if run["data"][key] != reference["data"][key]:
                raise ValueError(f"{name} data mismatch: {key}")

    source_hashes = {
        run["source"]["nestquant_python_source_sha256"] for run in runs.values()
    }
    if len(source_hashes) != 1:
        raise ValueError("all ablation runs must use the same NestQuant source hash")

    expected_setups = {
        "bf16": (None, False),
        "ordinary_w4": (None, False),
        "rotation_seed123": (123, False),
        "rotation_seed124": (124, False),
        "ldlq_only": (None, True),
        "rotation123_plus_ldlq": (123, True),
        "rotation124_plus_ldlq": (124, True),
    }
    for name, (rotation_seed, ldlq) in expected_setups.items():
        config = runs[name]["config"]
        if config["rotation_seed"] != rotation_seed or config["ldlq"] is not ldlq:
            raise ValueError(f"unexpected ablation setup for {name}")
    if set(runs["bf16"]["results"]) != {"bf16"}:
        raise ValueError("BF16 run must contain only the BF16 method")
    quantized_names = tuple(name for name in runs if name != "bf16")
    for name in quantized_names:
        if set(runs[name]["results"]) != {"nestquant_w4"}:
            raise ValueError(f"{name} must contain only NestQuant W4")

    if runs["ldlq_only"]["results"]["nestquant_w4"].get(
            "ldlq_hessian_coordinates") != "original_input":
        raise ValueError("LDLQ-only run has unexpected Hessian coordinates")
    for name in ("rotation123_plus_ldlq", "rotation124_plus_ldlq"):
        if runs[name]["results"]["nestquant_w4"].get(
                "ldlq_hessian_coordinates") != "per_linear_signed_hadamard_input":
            raise ValueError(f"{name} has unexpected Hessian coordinates")

    ldlq_calibration = runs["ldlq_only"]["results"]["nestquant_w4"]["ldlq_calibration"]
    calibration_keys = (
        "chunks", "calibration_tokens", "calibration_token_ids_sha256",
        "train_corpus_token_ids_sha256", "hessian_bytes_total",
    )
    for name in ("rotation123_plus_ldlq", "rotation124_plus_ldlq"):
        calibration = runs[name]["results"]["nestquant_w4"]["ldlq_calibration"]
        for key in calibration_keys:
            if calibration[key] != ldlq_calibration[key]:
                raise ValueError(f"LDLQ calibration mismatch for {name}: {key}")

    metrics = {
        "bf16": measurement(runs["bf16"], "bf16", paths["bf16"]),
        **{
            name: measurement(runs[name], "nestquant_w4", paths[name])
            for name in quantized_names
        },
    }
    p = {name: item["perplexity"] for name, item in metrics.items()}
    ldlq_gain = p["ordinary_w4"] - p["ldlq_only"]
    seed_metrics = {}
    for seed in (123, 124):
        rotation_name = f"rotation_seed{seed}"
        combined_name = f"rotation{seed}_plus_ldlq"
        rotation_gain = p["ordinary_w4"] - p[rotation_name]
        combined_gain = p["ordinary_w4"] - p[combined_name]
        additive_expected = p["ordinary_w4"] - rotation_gain - ldlq_gain
        seed_metrics[str(seed)] = {
            "rotation_ppl_reduction_without_ldlq": rotation_gain,
            "rotation_ppl_reduction_after_ldlq": p["ldlq_only"] - p[combined_name],
            "ldlq_ppl_reduction_after_rotation": p[rotation_name] - p[combined_name],
            "combined_ppl_reduction_vs_ordinary_w4": combined_gain,
            "combined_remaining_ppl_gap_to_bf16": p[combined_name] - p["bf16"],
            "combined_bf16_gap_closed_percent": (
                100 * combined_gain / (p["ordinary_w4"] - p["bf16"])
            ),
            "additive_expected_ppl": additive_expected,
            "interaction_ppl_above_additive_expectation": (
                p[combined_name] - additive_expected
            ),
        }

    combined_ppls = [p["rotation123_plus_ldlq"], p["rotation124_plus_ldlq"]]
    incremental_rotation_gains = [
        seed_metrics[str(seed)]["rotation_ppl_reduction_after_ldlq"]
        for seed in (123, 124)
    ]

    layer_smoke = read_json(args.layer_smoke)
    if (layer_smoke.get("status") != "smoke_completed"
            or layer_smoke.get("rotation_seed") != 123
            or layer_smoke.get("ldlq_hessian_coordinates") != "signed_hadamard_input"):
        raise ValueError("invalid rotation+LDLQ layer smoke")

    summary = {
        "schema_version": 2,
        "status": "verified_local_result",
        "comparison": "Llama-3.2-1B NestQuant W4 rotation x block-LDLQ ablation",
        "protocol": {
            key: reference["config"][key] for key in common_config_keys
        } | {
            "evaluation_split": "test",
            "test_token_ids_sha256": reference["data"]["token_ids_sha256"],
            "predicted_token_count": reference["data"]["predicted_token_count"],
            "nestquant_python_source_sha256": next(iter(source_hashes)),
            "rotation_seeds": [123, 124],
            "ldlq_calibration_chunks": ldlq_calibration["chunks"],
            "ldlq_calibration_token_ids_sha256": ldlq_calibration[
                "calibration_token_ids_sha256"],
            "ldlq_relative_damp": runs["ldlq_only"]["config"]["ldlq_relative_damp"],
        },
        "measurements": metrics,
        "derived": {
            "ldlq_ppl_reduction_without_rotation": ldlq_gain,
            "by_rotation_seed": seed_metrics,
            "two_seed_combined_ppl_mean": statistics.mean(combined_ppls),
            "two_seed_combined_ppl_sample_std": statistics.stdev(combined_ppls),
            "two_seed_rotation_gain_after_ldlq_mean": statistics.mean(
                incremental_rotation_gains),
            "two_seed_rotation_gain_after_ldlq_min": min(incremental_rotation_gains),
            "two_seed_rotation_gain_after_ldlq_max": max(incremental_rotation_gains),
        },
        "layer0_qproj_smoke": {
            "source": str(args.layer_smoke),
            "rotation_only_hessian_proxy_loss": layer_smoke["ordinary"]["hessian_proxy_loss"],
            "rotation_plus_ldlq_hessian_proxy_loss": layer_smoke["block_ldlq"][
                "hessian_proxy_loss"],
            "proxy_loss_reduction_fraction": layer_smoke["comparison"][
                "proxy_loss_reduction_fraction"],
            "weight_mse_ratio_ldlq_over_rotation_only": layer_smoke["comparison"][
                "weight_mse_ratio_ldlq_over_ordinary"],
        },
        "limitations": [
            "This combines the project's per-Linear signed-Hadamard weight-space ablation with block-LDLQ; it is not the paper's complete globally fused rotation graph.",
            "The combined result was repeated with only two preselected rotation seeds; this is a robustness check, not a broad statistical study.",
            "Beta subsets are selected before LDLQ feedback, while the paper discusses post-LDLQ beta optimization.",
            "Weights execute through BF16 Linear after inverse rotation and dequantization; no low-bit runtime speedup is measured.",
            "A4, KV4 and QA-LDLQ remain unimplemented.",
        ],
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    smoke = summary["layer0_qproj_smoke"]
    s123 = seed_metrics["123"]
    s124 = seed_metrics["124"]
    markdown = f"""# Llama-3.2-1B：NestQuant W4 旋转 × block-LDLQ 消融

## 结论

在相同源码、模型、测试 token、量化参数、dither seed 和 WikiText-2 test 协议下，signed-Hadamard 权重旋转与 train64 block-LDLQ 可以组合。两个旋转种子的组合 PPL 分别为 **{p['rotation123_plus_ldlq']:.6f}** 和 **{p['rotation124_plus_ldlq']:.6f}**，都优于仅 LDLQ 的 {p['ldlq_only']:.6f}，说明旋转在 LDLQ 之后仍提供了可重复的额外改善。

| 设置 | rotation seed | train64 LDLQ | PPL | 相对普通 W4 改善 |
|---|---:|---:|---:|---:|
| BF16 | — | 否 | **{p['bf16']:.6f}** | — |
| 普通 NestQuant W4 | — | 否 | {p['ordinary_w4']:.6f} | 0 |
| 仅 signed-Hadamard 旋转 | 123 | 否 | {p['rotation_seed123']:.6f} | {s123['rotation_ppl_reduction_without_ldlq']:.6f} |
| 仅 signed-Hadamard 旋转 | 124 | 否 | {p['rotation_seed124']:.6f} | {s124['rotation_ppl_reduction_without_ldlq']:.6f} |
| 仅 block-LDLQ | — | 是 | {p['ldlq_only']:.6f} | {ldlq_gain:.6f} |
| signed-Hadamard + block-LDLQ | 123 | 是 | **{p['rotation123_plus_ldlq']:.6f}** | **{s123['combined_ppl_reduction_vs_ordinary_w4']:.6f}** |
| signed-Hadamard + block-LDLQ | 124 | 是 | **{p['rotation124_plus_ldlq']:.6f}** | **{s124['combined_ppl_reduction_vs_ordinary_w4']:.6f}** |

两种子组合 PPL 均值为 **{statistics.mean(combined_ppls):.6f}**，样本标准差为 **{statistics.stdev(combined_ppls):.6f}**。旋转在 LDLQ 之后额外降低 PPL `{s123['rotation_ppl_reduction_after_ldlq']:.6f}`（seed 123）和 `{s124['rotation_ppl_reduction_after_ldlq']:.6f}`（seed 124）。

## 模块贡献与交互

- 不使用旋转时，LDLQ 将 PPL 降低 `{ldlq_gain:.6f}`。
- seed 123 中，旋转单独降低 `{s123['rotation_ppl_reduction_without_ldlq']:.6f}`，在 LDLQ 后额外降低 `{s123['rotation_ppl_reduction_after_ldlq']:.6f}`。
- seed 124 中，旋转单独降低 `{s124['rotation_ppl_reduction_without_ldlq']:.6f}`，在 LDLQ 后额外降低 `{s124['rotation_ppl_reduction_after_ldlq']:.6f}`。
- 两个种子都表现出部分收益重叠，而组合均优于任一单模块设置。

## 坐标系核验

对行向量旋转 `X -> X R` 和权重旋转 `W -> W R`，LDLQ 使用的协方差必须变为 `H_rot = R^T H R`。项目直接对 train64 Hessian 做双侧快速 Hadamard 变换，与重新对 `X R` 收集协方差数学等价；单元测试已用显式激活计算核对。

真实 `layers.0.self_attn.q_proj` smoke 中，仅旋转的 Hessian 代理损失为 `{smoke['rotation_only_hessian_proxy_loss']:.6f}`，旋转加 LDLQ 为 `{smoke['rotation_plus_ldlq_hessian_proxy_loss']:.6f}`，下降 `{100 * smoke['proxy_loss_reduction_fraction']:.2f}%`。普通权重 MSE 增加到 `{smoke['weight_mse_ratio_ldlq_over_rotation_only']:.2f}` 倍，符合 LDLQ 优化激活加权误差而非逐权重 MSE 的目标。

## 可复核协议

- NestQuant 源码 SHA-256：`{summary['protocol']['nestquant_python_source_sha256']}`。
- 测试 token SHA-256：`{summary['protocol']['test_token_ids_sha256']}`；预测 token：`{summary['protocol']['predicted_token_count']:,}`。
- train64 校准 token SHA-256：`{summary['protocol']['ldlq_calibration_token_ids_sha256']}`。
- 所有运行使用相同模型/数据 revision、q=14、dither seed 58、7 个 beta 候选中选 4 个、LDLQ 相对阻尼 0.01。

## 边界

本实验组合的是项目当前“每个 Linear 独立随机符号 Hadamard、量化后逆旋转回原权重坐标”的算法消融，与论文完整的全模型旋转融合仍有差距。组合结论只用两个预先测试的 rotation seed 做了稳健性核验，不构成大样本统计。当前也没有 A4、KV4、QA-LDLQ 或低比特 kernel；PPL 使用 BF16 Linear 执行，不能据此声称推理加速。
"""
    args.output_markdown.write_text(markdown, encoding="utf-8")
    print(f"Verified JSON: {args.output_json}")
    print(f"Verified Markdown: {args.output_markdown}")


if __name__ == "__main__":
    main()

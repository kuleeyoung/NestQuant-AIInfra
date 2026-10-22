"""Validate and summarize the full Llama-3.2-1B W4KV4 ablation runs."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUNS = {
    "w4_dynamic_cache": PROJECT_ROOT / "results/raw/llama32_1b_w4_ldlq_cache_control_current_full_20261021.json",
    "w4_qk_rotation": PROJECT_ROOT / "results/raw/llama32_1b_w4_ldlq_qkrot_control_current_full_20261021.json",
    "w4kv4_local_dp": PROJECT_ROOT / "results/raw/llama32_1b_w4kv4_perhead_qkrot_margin4_full_20261021.json",
    "w4kv4_upstream_beta": PROJECT_ROOT / "results/raw/llama32_1b_w4kv4_upstream_betas_full_20261021.json",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name, default in DEFAULT_RUNS.items():
        parser.add_argument(f"--{name.replace('_', '-')}", type=Path, default=default)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=PROJECT_ROOT / "results/llama32_1b_w4kv4_ablation.json",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=PROJECT_ROOT / "results/llama32_1b_w4kv4_ablation.md",
    )
    args = parser.parse_args()
    paths = {
        name: getattr(args, name)
        for name in DEFAULT_RUNS
    }
    payload = build_summary(paths)
    args.output_json.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    args.output_md.write_text(render_markdown(payload), encoding="utf-8")
    print(f"Saved {args.output_json}")
    print(f"Saved {args.output_md}")


def build_summary(paths: dict[str, Path]) -> dict:
    manifests = {
        name: json.loads(path.read_text(encoding="utf-8"))
        for name, path in paths.items()
    }
    for name, manifest in manifests.items():
        if manifest.get("status") != "completed":
            raise ValueError(f"{name} is not a completed full run")
        if manifest["config"].get("limit_chunks") is not None:
            raise ValueError(f"{name} is a smoke run")

    comparable_fields = {
        "model_id": lambda item: item["config"]["model_id"],
        "model_revision": lambda item: item["config"]["model_revision"],
        "dataset_revision": lambda item: item["config"]["dataset_revision"],
        "context_length": lambda item: item["config"]["context_length"],
        "token_ids_sha256": lambda item: item["data"]["token_ids_sha256"],
        "predicted_token_count": lambda item: item["data"]["predicted_token_count"],
        "source_sha256": lambda item: item["source"]["nestquant_python_source_sha256"],
        "checkpoint_sha256": lambda item: item["results"]["nestquant_w4"]["checkpoint_round_trip"]["file_sha256"],
    }
    comparability = {}
    for field, getter in comparable_fields.items():
        values = {name: getter(item) for name, item in manifests.items()}
        unique = set(values.values())
        if len(unique) != 1:
            raise ValueError(f"runs disagree on {field}: {values}")
        comparability[field] = next(iter(unique))

    rows = {}
    for name, manifest in manifests.items():
        method = manifest["results"]["nestquant_w4"]
        evaluation = method["evaluation"]
        kv_stats = method.get("kv_cache_quantization")
        rows[name] = {
            "label": {
                "w4_dynamic_cache": "W4 LDLQ + floating DynamicCache",
                "w4_qk_rotation": "W4 LDLQ + matched Q/K rotation",
                "w4kv4_local_dp": "W4KV4 + local six-chunk beta DP",
                "w4kv4_upstream_beta": "W4KV4 + public-upstream default betas",
            }[name],
            "manifest": str(paths[name].resolve()),
            "manifest_sha256": hashlib.sha256(paths[name].read_bytes()).hexdigest(),
            "config_sha256": manifest["config_sha256"],
            "perplexity": evaluation["perplexity"],
            "mean_nll": evaluation["mean_nll_per_predicted_token"],
            "predicted_tokens": evaluation["predicted_token_count"],
            "elapsed_seconds": method["elapsed_seconds"],
            "relative_squared_error": None if kv_stats is None else kv_stats["relative_squared_error"],
            "overload_fraction": None if kv_stats is None else _overload_fraction(kv_stats),
            "estimated_bits_per_kv_entry": None if kv_stats is None else kv_stats["storage_estimate"]["effective_bits_per_entry"],
            "kv_storage_status": None if kv_stats is None else kv_stats["storage_estimate"]["status"],
        }

    baseline = rows["w4_dynamic_cache"]["perplexity"]
    rotation = rows["w4_qk_rotation"]["perplexity"]
    for row in rows.values():
        row["ppl_delta_vs_w4"] = row["perplexity"] - baseline
        row["ppl_relative_change_vs_w4_percent"] = (row["perplexity"] / baseline - 1.0) * 100.0
        row["ppl_delta_vs_qk_rotation"] = row["perplexity"] - rotation
        row["ppl_relative_change_vs_qk_rotation_percent"] = (
            row["perplexity"] / rotation - 1.0
        ) * 100.0

    local = rows["w4kv4_local_dp"]["perplexity"]
    upstream = rows["w4kv4_upstream_beta"]["perplexity"]
    return {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "validated",
        "scope": "Llama-3.2-1B Base, NQMC W4 block-LDLQ weights, WikiText-2 test, KV4 quality simulation",
        "comparability": comparability,
        "results": rows,
        "findings": {
            "qk_rotation_relative_ppl_change_percent": (
                rotation / baseline - 1.0
            ) * 100.0,
            "upstream_beta_relative_ppl_change_vs_w4_percent": (
                upstream / baseline - 1.0
            ) * 100.0,
            "local_dp_relative_ppl_change_vs_w4_percent": (
                local / baseline - 1.0
            ) * 100.0,
            "upstream_beta_ppl_improvement_over_local_dp": local - upstream,
            "upstream_beta_relative_ppl_improvement_over_local_dp_percent": (
                1.0 - upstream / local
            ) * 100.0,
        },
        "evidence_boundary": {
            "quality": "full WikiText-2 fake-quant PPL measured",
            "storage": "4.75 bits/entry is an analytical estimate; runtime cache remains floating point",
            "speed": "not measured and not claimed; no packed KV cache or low-bit attention kernel",
            "paper_reproduction": "public defaults are mapped from the upstream repository; the upstream beta DP script is unavailable",
        },
    }


def _overload_fraction(kv_stats: dict) -> float:
    overloads = 0
    blocks = 0
    for layer in kv_stats["layers"].values():
        for kind in ("key", "value"):
            overloads += layer[kind]["overload_count"]
            blocks += layer[kind]["e8_block_count"]
    return overloads / blocks


def render_markdown(payload: dict) -> str:
    rows = payload["results"]
    finding = payload["findings"]
    lines = [
        "# Llama-3.2-1B W4KV4 消融实验",
        "",
        "四个完整运行已经通过模型、数据、token、代码和 NQMC 文件哈希一致性检查。",
        "",
        "| 设置 | PPL | 相对 W4 | KV 相对平方误差 | KV 过载率 | 估算 bit/entry |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name in ("w4_dynamic_cache", "w4_qk_rotation", "w4kv4_upstream_beta", "w4kv4_local_dp"):
        row = rows[name]
        rel_error = "—" if row["relative_squared_error"] is None else f"{row['relative_squared_error']:.6%}"
        overload = "—" if row["overload_fraction"] is None else f"{row['overload_fraction']:.6%}"
        bits = "—" if row["estimated_bits_per_kv_entry"] is None else f"{row['estimated_bits_per_kv_entry']:.2f}"
        lines.append(
            f"| {row['label']} | {row['perplexity']:.6f} | "
            f"{row['ppl_relative_change_vs_w4_percent']:+.3f}% | {rel_error} | {overload} | {bits} |"
        )
    lines += [
        "",
        "## 结论",
        "",
        f"- 匹配 Q/K Hadamard 旋转相对普通 DynamicCache 的 PPL 变化为 "
        f"{finding['qk_rotation_relative_ppl_change_percent']:+.3f}%，因此缓存适配与旋转本身只产生很小的 BF16 数值漂移。",
        f"- 公开实现默认 beta 的 W4KV4 PPL 相对 W4 增加 "
        f"{finding['upstream_beta_relative_ppl_change_vs_w4_percent']:.3f}%。",
        f"- 本地六段校准 DP 的 W4KV4 PPL 相对 W4 增加 "
        f"{finding['local_dp_relative_ppl_change_vs_w4_percent']:.3f}%；公开默认 beta 比它低 "
        f"{finding['upstream_beta_ppl_improvement_over_local_dp']:.6f} PPL。",
        "- 该反例说明在六段训练校准样本上按 E8 块重构误差选出的 beta 不保证测试集重构误差或端到端 PPL 最优；校准目标、样本规模与跨层敏感度仍需研究。",
        "",
        "## 证据边界",
        "",
        "当前实现把量化后的 K/V 重构为浮点张量再参与注意力计算，因此证明的是质量仿真。4.75 bit/entry 是包含 4-bit residue、每 8 维 2-bit beta index 和每 64 维 32-bit 行范数的理论估算。尚未分配真实 packed KV cache，也没有低比特 attention kernel，所以不能据此声称显存降低或推理加速。",
        "",
        "公开默认 beta 来自固定上游提交的 `parallel/ppl.py`，系数除以 `q` 的行为由 `coder.py` 核对。公开仓库未提供其 beta DP 脚本，本地六段校准属于项目自研消融，不应描述为作者原始校准流程。",
        "",
        "## 可复核信息",
        "",
        f"- 模型 revision：`{payload['comparability']['model_revision']}`",
        f"- 数据 revision：`{payload['comparability']['dataset_revision']}`",
        f"- token SHA-256：`{payload['comparability']['token_ids_sha256']}`",
        f"- NQMC SHA-256：`{payload['comparability']['checkpoint_sha256']}`",
        f"- 当前 Python 源码 SHA-256：`{payload['comparability']['source_sha256']}`",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    main()

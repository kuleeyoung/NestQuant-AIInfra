"""Verify LDLQ NQMC export/reload runs and write Git-safe JSON/Markdown evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--export-run", type=Path,
        default=Path("results/raw/llama32_1b_ldlq_train64_nqmc_full_20261021.json"),
    )
    parser.add_argument(
        "--load-run", type=Path,
        default=Path("results/raw/llama32_1b_ldlq_train64_nqmc_loadonly_full_20261021.json"),
    )
    parser.add_argument(
        "--accounting", type=Path,
        default=Path("results/raw/llama32_1b_nestquant_w4_ldlq_train64_accounting.json"),
    )
    parser.add_argument(
        "--quality-summary", type=Path,
        default=Path("results/llama32_1b_w4_ldlq_calibration_sweep.json"),
    )
    parser.add_argument(
        "--output-json", type=Path,
        default=Path("results/llama32_1b_w4_ldlq_nqmc.json"),
    )
    parser.add_argument(
        "--output-markdown", type=Path,
        default=Path("results/llama32_1b_w4_ldlq_nqmc.md"),
    )
    args = parser.parse_args()

    export_run = read_json(args.export_run)
    load_run = read_json(args.load_run)
    accounting = read_json(args.accounting)
    quality = read_json(args.quality_summary)
    for label, run in (("export", export_run), ("load", load_run)):
        if run.get("status") != "completed" or run["config"].get("limit_chunks") is not None:
            raise ValueError(f"{label} run must be a completed full evaluation")

    export_result = export_run["results"]["nestquant_w4"]
    load_result = load_run["results"]["nestquant_w4"]
    export_checkpoint = export_result["checkpoint_round_trip"]
    load_checkpoint = load_result["checkpoint_round_trip"]
    if export_checkpoint["encoder"] != "block_ldlq":
        raise ValueError("exported checkpoint is not marked as block-LDLQ")
    if export_checkpoint["file_sha256"] != load_checkpoint["file_sha256"]:
        raise ValueError("export and load-only runs used different checkpoint files")
    if export_checkpoint["file_sha256"] != accounting["file_sha256"]:
        raise ValueError("checkpoint accounting hash disagrees with evaluation")
    if export_checkpoint["file_bytes"] != accounting["file_bytes"]:
        raise ValueError("checkpoint accounting byte count disagrees with evaluation")
    checked_data = ("token_ids_sha256", "chunk_count", "predicted_token_count")
    for key in checked_data:
        if export_run["data"][key] != load_run["data"][key]:
            raise ValueError(f"evaluation data mismatch: {key}")
    export_ppl = export_result["evaluation"]["perplexity"]
    load_ppl = load_result["evaluation"]["perplexity"]
    in_memory_ppl = quality["results"]["nestquant_w4_ldlq_train64"]["perplexity"]
    if export_ppl != load_ppl or export_ppl != in_memory_ppl:
        raise ValueError("in-memory, export-roundtrip and load-only PPL do not match exactly")

    encoder = export_checkpoint["encoder_metadata"]
    summary = {
        "schema_version": 1,
        "status": "verified_local_result",
        "scope": "Llama-3.2-1B NestQuant W4 block-LDLQ NQMC export and reload",
        "model_id": export_run["config"]["model_id"],
        "model_revision": export_run["config"]["model_revision"],
        "dataset_id": export_run["config"]["dataset_id"],
        "dataset_revision": export_run["config"]["dataset_revision"],
        "test_token_ids_sha256": export_run["data"]["token_ids_sha256"],
        "predicted_token_count": export_run["data"]["predicted_token_count"],
        "calibration": encoder,
        "quality": {
            "bf16_perplexity": quality["results"]["bf16"]["perplexity"],
            "ordinary_nestquant_w4_perplexity": quality["results"]["nestquant_w4"]["perplexity"],
            "ldlq_in_memory_perplexity": in_memory_ppl,
            "ldlq_export_roundtrip_perplexity": export_ppl,
            "ldlq_load_only_perplexity": load_ppl,
            "exact_ppl_match": True,
        },
        "checkpoint": {
            key: accounting[key] for key in (
                "file_sha256", "file_bytes", "quantized_tensor_count",
                "quantized_parameter_count", "quantized_record_bytes",
                "effective_bits_per_quantized_parameter_including_record_metadata",
                "dense_tensor_count", "dense_payload_bytes", "tied_alias_count",
                "container_header_manifest_bytes", "raw_bf16_equivalent_payload_bytes",
                "file_to_raw_bf16_payload_ratio",
            )
        },
        "runs": {
            "export_roundtrip_manifest": str(args.export_run),
            "load_only_manifest": str(args.load_run),
            "export_roundtrip_elapsed_seconds": export_result["elapsed_seconds"],
            "load_only_full_run_elapsed_seconds": load_result["elapsed_seconds"],
        },
        "limitations": [
            "NQMC contains real packed 4-bit residue and beta-index streams, but reload dequantizes weights for BF16 Linear execution.",
            "The BF16 size denominator is a calculated unique-state payload, not a separately exported BF16 checkpoint file.",
            "The load-only elapsed time includes base-model loading, NQMC parsing/dequantization and PPL evaluation; it is not an isolated model-load benchmark.",
            "No low-bit matrix-multiplication kernel, latency speedup or throughput improvement is claimed.",
            "Rotation and LDLQ remain separate model-level ablations; A4, KV4 and QA-LDLQ are not implemented.",
        ],
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    checkpoint = summary["checkpoint"]
    mib = checkpoint["file_bytes"] / (1024 ** 2)
    gib_hessian = encoder["hessian_bytes_total"] / (1024 ** 3)
    bf16_ratio_percent = checkpoint["file_to_raw_bf16_payload_ratio"] * 100
    ordinary = summary["quality"]["ordinary_nestquant_w4_perplexity"]
    bf16 = summary["quality"]["bf16_perplexity"]
    gap_closed = 100 * (ordinary - export_ppl) / (ordinary - bf16)
    markdown = f"""# Llama-3.2-1B：block-LDLQ NQMC 导出与独立回载

## 结论

方向一已经完成。项目把 train64 校准得到的 E8 block-LDLQ 最终码字写入真实 NQMC 文件，并在不提供 Hessian 的独立进程中回载。内存量化、导出后立即回载、独立回载三条路径的完整 WikiText-2 PPL **精确一致**，均为 `{export_ppl:.12f}`。

| 路径 | 是否需要 Hessian | 完整 WikiText-2 PPL |
|---|---:|---:|
| train64 LDLQ 内存质量仿真 | 编码时需要 | {in_memory_ppl:.6f} |
| LDLQ 编码 → NQMC → 立即回载 | 编码时需要 | {export_ppl:.6f} |
| 已有 NQMC 独立回载 | **不需要** | {load_ppl:.6f} |

BF16 PPL 为 `{bf16:.6f}`，普通 NestQuant W4 为 `{ordinary:.6f}`。block-LDLQ 保留了此前的质量收益，消除了普通 W4 到 BF16 差距的 `{gap_closed:.2f}%`。

## 实际存储结果

| 指标 | 实测或核算值 |
|---|---:|
| NQMC 文件 | `{checkpoint['file_bytes']:,}` B（{mib:.2f} MiB） |
| 文件 SHA-256 | `{checkpoint['file_sha256']}` |
| 量化张量 | {checkpoint['quantized_tensor_count']} |
| 量化参数 | `{checkpoint['quantized_parameter_count']:,}` |
| 量化记录有效码率 | `{checkpoint['effective_bits_per_quantized_parameter_including_record_metadata']:.6f}` bit/量化参数 |
| 稠密张量载荷 | `{checkpoint['dense_payload_bytes']:,}` B |
| tied alias | {checkpoint['tied_alias_count']} |
| NQMC / BF16 唯一状态值逻辑载荷 | `{bf16_ratio_percent:.4f}%` |

BF16 分母 `{checkpoint['raw_bf16_equivalent_payload_bytes']:,}` B 是把同一批唯一状态值按 BF16 存储得到的逻辑载荷，并非另行导出的 BF16 文件。因此这里可以说“实测 NQMC 文件为 BF16 逻辑载荷的 26.67%”，不能说两个实际模型文件已经完成严格对照。

## 校准证据

- 校准集：`{encoder['dataset']}`，固定 revision `{encoder['dataset_revision']}`。
- 校准规模：{encoder['chunks']} × {encoder['context_length']} = `{encoder['calibration_tokens']:,}` token。
- 选中 token SHA-256：`{encoder['calibration_token_ids_sha256']}`。
- Hessian：{len(encoder['hessian_artifacts'])} 个文件、65 份唯一统计，共 `{encoder['hessian_bytes_total']:,}` B（{gib_hessian:.2f} GiB）。
- Hessian 只用于离线 LDLQ 编码；NQMC 解码不读取 Hessian。NQMC 内保存 residue code、beta index、行范数和可复现 dither RNG 状态。

## 工程边界

NQMC 是真实低比特**存储**格式，但当前加载后会解码为浮点权重并执行 BF16 Linear。该结果证明文件可独立保存、校验、回载并保持 PPL，尚未证明低比特常驻显存或推理加速。完整原始 manifest 和大模型文件位于 `results/raw/`，被 Git 忽略；本文件与同名 JSON 是可提交的核验摘要。
"""
    args.output_markdown.write_text(markdown, encoding="utf-8")
    print(f"Verified JSON: {args.output_json}")
    print(f"Verified Markdown: {args.output_markdown}")


if __name__ == "__main__":
    main()

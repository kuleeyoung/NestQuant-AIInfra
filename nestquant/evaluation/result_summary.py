"""Build a flat CSV summary from local WikiText-2 PPL run manifests."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


FIELDS = (
    "run_id",
    "status",
    "quality_stage",
    "method",
    "model_id",
    "model_revision",
    "dataset_id",
    "dataset_revision",
    "context_length",
    "seed",
    "predicted_tokens",
    "perplexity",
    "ppl_delta_vs_bf16",
    "elapsed_seconds",
    "peak_cuda_memory_allocated_bytes",
    "config_sha256",
    "manifest_path",
    "error",
)


def collect_rows(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(paths):
        payload = json.loads(path.read_text(encoding="utf-8"))
        config = payload.get("config", {})
        results = payload.get("results", {})
        bf16_result = results.get("bf16", {}).get("evaluation", {})
        bf16_ppl = bf16_result.get("perplexity")

        if results:
            for method, result in results.items():
                evaluation = result.get("evaluation", {})
                ppl = evaluation.get("perplexity")
                delta = None
                if method != "bf16" and isinstance(ppl, (int, float)) and isinstance(bf16_ppl, (int, float)):
                    delta = (ppl / bf16_ppl - 1.0) * 100.0
                rows.append(_row(payload, path, method, result, evaluation, delta))
        else:
            rows.append(_row(payload, path, payload.get("failed_method", ""), {}, {}, None))
    return rows


def _row(payload: dict[str, Any], path: Path, method: str,
         result: dict[str, Any], evaluation: dict[str, Any], delta: float | None) -> dict[str, Any]:
    config = payload.get("config", {})
    error = payload.get("error", {})
    return {
        "run_id": payload.get("run_id", ""),
        "status": payload.get("status", "unknown"),
        "quality_stage": result.get("quality_stage", ""),
        "method": method,
        "model_id": config.get("model_id", ""),
        "model_revision": config.get("model_revision", ""),
        "dataset_id": config.get("dataset_id", ""),
        "dataset_revision": config.get("dataset_revision", ""),
        "context_length": config.get("context_length", ""),
        "seed": config.get("seed", ""),
        "predicted_tokens": evaluation.get("predicted_token_count", ""),
        "perplexity": evaluation.get("perplexity", ""),
        "ppl_delta_vs_bf16": delta if delta is not None else "",
        "elapsed_seconds": result.get("elapsed_seconds", ""),
        "peak_cuda_memory_allocated_bytes": evaluation.get("peak_cuda_memory_allocated_bytes", ""),
        "config_sha256": payload.get("config_sha256", ""),
        "manifest_path": str(path),
        "error": error.get("message", "") if isinstance(error, dict) else str(error),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("results/raw"))
    parser.add_argument("--output", type=Path, default=Path("results/ppl_summary.csv"))
    args = parser.parse_args()
    manifests = sorted(args.input_dir.glob("*.json")) if args.input_dir.exists() else []
    rows = collect_rows(manifests)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Summarized {len(manifests)} manifest(s), {len(rows)} row(s): {args.output}")


if __name__ == "__main__":
    main()

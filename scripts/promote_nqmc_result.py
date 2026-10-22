"""Verify and promote local full-PPL manifests into a small Git-safe summary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path,
                        default=Path("results/raw/llama32_1b_bf16_rtn_w4_current_full_ppl.json"))
    parser.add_argument("--nestquant", type=Path,
                        default=Path("results/raw/llama32_1b_nestquant_w4_nqmc_full_ppl.json"))
    parser.add_argument("--accounting", type=Path,
                        default=Path("results/raw/llama32_1b_nestquant_w4_nqmc_accounting.json"))
    parser.add_argument("--output", type=Path,
                        default=Path("results/llama32_1b_nqmc_w4_full.json"))
    args = parser.parse_args()
    baseline, nestquant, accounting = map(read_json, (args.baseline, args.nestquant, args.accounting))
    if baseline.get("status") != "completed" or nestquant.get("status") != "completed":
        raise ValueError("only completed full-PPL runs can be promoted")
    checked_config_keys = ("model_id", "model_revision", "dataset_id", "dataset_config",
                           "dataset_revision", "context_length", "model_dtype", "logit_loss_dtype")
    for key in checked_config_keys:
        if baseline["config"][key] != nestquant["config"][key]:
            raise ValueError(f"evaluation config mismatch: {key}")
    checked_data_keys = ("text_sha256", "token_ids_sha256", "chunk_count", "predicted_token_count")
    for key in checked_data_keys:
        if baseline["data"][key] != nestquant["data"][key]:
            raise ValueError(f"evaluation data mismatch: {key}")
    if (baseline["config"]["limit_chunks"] is not None
            or nestquant["config"]["limit_chunks"] is not None):
        raise ValueError("smoke runs cannot be promoted")
    checkpoint = nestquant["results"]["nestquant_w4"]["checkpoint_round_trip"]
    if (checkpoint["file_sha256"] != accounting["file_sha256"]
            or checkpoint["file_bytes"] != accounting["file_bytes"]
            or not checkpoint["reloaded_before_perplexity"]):
        raise ValueError("checkpoint accounting disagrees with PPL manifest")
    methods = {}
    for key, source in (("bf16", baseline), ("rtn_w4", baseline), ("nestquant_w4", nestquant)):
        measurement = source["results"][key]["evaluation"]
        methods[key] = {"perplexity": measurement["perplexity"],
                        "mean_nll_per_predicted_token": measurement["mean_nll_per_predicted_token"],
                        "predicted_token_count": measurement["predicted_token_count"],
                        "run_id": source["run_id"]}
    summary = {
        "status": "verified_local_result",
        "model_id": baseline["config"]["model_id"],
        "model_revision": baseline["config"]["model_revision"],
        "dataset_id": baseline["config"]["dataset_id"],
        "dataset_config": baseline["config"]["dataset_config"],
        "dataset_revision": baseline["config"]["dataset_revision"],
        "context_length": baseline["config"]["context_length"],
        "token_ids_sha256": baseline["data"]["token_ids_sha256"],
        "chunk_count": baseline["data"]["chunk_count"],
        "methods": methods,
        "checkpoint": {key: accounting[key] for key in (
            "file_sha256", "file_bytes", "quantized_tensor_count", "quantized_parameter_count",
            "quantized_record_bytes", "effective_bits_per_quantized_parameter_including_record_metadata",
            "dense_tensor_count", "dense_payload_bytes", "tied_alias_count",
            "raw_bf16_equivalent_payload_bytes", "file_to_raw_bf16_payload_ratio")},
        "limitations": [
            "Paper Appendix G reports a search universe through 40.0/q and selected defaults through 25.0/q; the project adds 40.0 to selected candidates and extends the reported universe with 50.0.",
            "Rotation and LDLQ/QA-LDLQ are not implemented.",
            "PPL uses dequantized BF16 model execution; no low-bit inference speedup is measured.",
            "BF16 reference size is calculated raw payload, not a measured BF16 file.",
            "RTN W4 has only estimated effective bit rate; no packed RTN checkpoint is measured.",
            "The repository had no Git commit at measurement time.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Verified summary written to {args.output}")


if __name__ == "__main__":
    main()

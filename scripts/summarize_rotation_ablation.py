"""Promote completed same-protocol rotation runs into a small audited summary."""

import argparse
import json
from pathlib import Path
import statistics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--rotated", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, default=Path("results/rotation_w4_ablation.json"))
    args = parser.parse_args()
    paths = [args.baseline, *args.rotated]
    runs = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    base = runs[0]
    keys = ("model_id", "model_revision", "dataset_id", "dataset_config", "dataset_revision",
            "context_length", "model_dtype", "logit_loss_dtype", "seed", "q",
            "beta_coefficients_project_scale", "selected_beta_count", "dither_rng_device",
            "rows_per_chunk", "quantization_scope")
    if base["config"].get("weight_rotation", "none") != "none":
        raise ValueError("baseline must have no weight rotation")
    measurements = []
    source_hashes = set()
    for index, (path, run) in enumerate(zip(paths, runs)):
        if run["status"] != "completed" or run["config"]["limit_chunks"] is not None:
            raise ValueError(f"{path}: full evaluation required")
        for key in keys:
            if run["config"][key] != base["config"][key]:
                raise ValueError(f"{path}: protocol mismatch: {key}")
        if run["data"]["token_ids_sha256"] != base["data"]["token_ids_sha256"]:
            raise ValueError(f"{path}: token hash mismatch")
        if index and run["config"].get("weight_rotation") != "signed_hadamard_inverse_weight":
            raise ValueError("rotated inputs must use signed Hadamard")
        result = run["results"]["nestquant_w4"]
        evaluation = result["evaluation"]
        if evaluation["predicted_token_count"] != base["data"]["predicted_token_count"]:
            raise ValueError("evaluation does not cover all target tokens")
        source_hash = run["source"].get("nestquant_python_source_sha256")
        if not source_hash:
            raise ValueError(f"{path}: missing NestQuant source hash")
        source_hashes.add(source_hash)
        measurements.append({
            "source": path.as_posix(), "run_id": run["run_id"],
            "source_sha256": source_hash,
            "rotation_seed": run["config"].get("rotation_seed"),
            "perplexity": evaluation["perplexity"],
            "mean_nll": evaluation["mean_nll_per_predicted_token"],
            "elapsed_seconds": result["elapsed_seconds"],
        })
    ppls = [m["perplexity"] for m in measurements[1:]]
    if len({m["rotation_seed"] for m in measurements[1:]}) != len(ppls):
        raise ValueError("duplicate rotation seeds")
    if len(source_hashes) != 1:
        raise ValueError(f"source mismatch across runs: {sorted(source_hashes)}")
    baseline_ppl = measurements[0]["perplexity"]
    for measurement in measurements[1:]:
        measurement["ppl_change_vs_no_rotation"] = measurement["perplexity"] - baseline_ppl
    summary = {
        "protocol": {key: base["config"][key] for key in keys},
        "evaluation_split": "test", "predicted_tokens": base["data"]["predicted_token_count"],
        "token_ids_sha256": base["data"]["token_ids_sha256"],
        "nestquant_python_source_sha256": next(iter(source_hashes)),
        "measurements": measurements,
        "rotated_mean_ppl": statistics.mean(ppls),
        "rotated_population_std_ppl": statistics.pstdev(ppls),
        "rotated_mean_ppl_change_vs_no_rotation": statistics.mean(ppls) - baseline_ppl,
        "limitations": [
            "Weight-space rotation ablation, not complete paper rotation/LDLQ/QA-LDLQ.",
            "Reconstructed original-basis weights execute in BF16; no packed rotated artifact or speedup.",
            "Paper Appendix G says validation; public author ppl.py/ppl_utils.py use test. This run uses test.",
            "Rotation seeds are tested at one fixed dither seed; do not infer broad robustness.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

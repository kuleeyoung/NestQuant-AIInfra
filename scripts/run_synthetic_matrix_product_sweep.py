"""Run the synthetic matrix-product experiment over multiple random seeds."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from nestquant.experiments.matrix_product import run_synthetic_matrix_product


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=16)
    parser.add_argument("--m", type=int, default=16)
    parser.add_argument("--k", type=int, default=32)
    parser.add_argument("--q", type=int, default=14)
    parser.add_argument("--beta-count", type=int, default=4)
    parser.add_argument("--seeds", type=int, nargs="+", default=[58, 59, 60])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", type=Path, default=Path("results/raw"))
    args = parser.parse_args()

    runs = [run_synthetic_matrix_product(n=args.n, m=args.m, k=args.k, q=args.q,
        beta_count=args.beta_count, seed=seed, device=args.device) for seed in args.seeds]
    method_names = list(runs[0]["metrics"])
    aggregate = {}
    for method in method_names:
        values = [run["metrics"][method]["mean_squared_error_per_entry"] for run in runs]
        aggregate[method] = {"mean_product_mse_per_entry": statistics.mean(values),
                             "sample_std_product_mse_per_entry": statistics.stdev(values) if len(values) > 1 else 0.0}

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"matrix_product_sweep_n{args.n}_m{args.m}_k{args.k}"
    payload = {"schema_version": 1, "experiment": "synthetic_matrix_product_seed_sweep",
               "created_at_utc": datetime.now(timezone.utc).isoformat(),
               "sweep_config": {"n": args.n, "m": args.m, "k": args.k, "q": args.q,
                                "beta_count": args.beta_count, "seeds": args.seeds,
                                "device": args.device},
               "aggregate": aggregate, "runs": runs,
               "limitations": runs[0]["limitations"] + [
                   "three seeds are an initial variability check, not a confidence interval or a paper-scale result"]}
    json_path = args.output_dir / f"{stem}.json"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    csv_path = args.output_dir / f"{stem}.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=["seed", "method", "mse_per_entry", "rmse_per_entry",
            "relative_frobenius_error", "paper_formula_bits_per_entry", "shannon_entropy_bits_per_entry"])
        writer.writeheader()
        for run in runs:
            for method, metrics in run["metrics"].items():
                writer.writerow({"seed": run["config"]["seed"], "method": method,
                    "mse_per_entry": metrics["mean_squared_error_per_entry"],
                    "rmse_per_entry": metrics["root_mean_squared_error_per_entry"],
                    "relative_frobenius_error": metrics["relative_frobenius_error"],
                    "paper_formula_bits_per_entry": run["rate_estimate"]["paper_formula_bits_per_entry"],
                    "shannon_entropy_bits_per_entry": run["rate_estimate"]["shannon_side_information_bits_per_entry"]})
    print(f"Saved: {json_path}")
    print(f"Saved: {csv_path}")
    for method, stats in aggregate.items():
        print(f"{method}: product MSE/entry={stats['mean_product_mse_per_entry']:.6g} "
              f"+/- {stats['sample_std_product_mse_per_entry']:.6g} (sample std across {len(runs)} seeds)")
    print("Scope: initial toy variability check; not a paper-scale or matched-rate conclusion.")


if __name__ == "__main__":
    main()

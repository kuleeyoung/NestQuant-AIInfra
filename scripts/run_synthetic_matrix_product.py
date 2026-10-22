"""Run a configurable toy matrix-product benchmark and save raw JSON."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from nestquant.experiments.matrix_product import run_synthetic_matrix_product


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=16, help="A rows (toy default: 16)")
    parser.add_argument("--m", type=int, default=16, help="B rows (toy default: 16)")
    parser.add_argument("--k", type=int, default=32, help="shared feature dimension; must be divisible by 8")
    parser.add_argument("--q", type=int, default=14)
    parser.add_argument("--beta-count", type=int, default=4)
    parser.add_argument("--seed", type=int, default=58)
    parser.add_argument("--candidate-step", type=float, default=0.5,
                        help="beta grid step; paper §5.1 uses 0.5 * {1,...,50}")
    parser.add_argument("--device", default="cpu", help="cpu or cuda")
    parser.add_argument("--output", type=Path, default=None, help="output JSON (default results/raw/...) ")
    args = parser.parse_args()

    result = run_synthetic_matrix_product(n=args.n, m=args.m, k=args.k, q=args.q,
        beta_count=args.beta_count, seed=args.seed, device=args.device,
        candidate_step=args.candidate_step)
    result["created_at_utc"] = datetime.now(timezone.utc).isoformat()
    output = args.output or Path("results/raw") / (
        f"matrix_product_toy_n{args.n}_m{args.m}_k{args.k}_seed{args.seed}.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Saved: {output}")
    for name, metrics in result["metrics"].items():
        print(f"{name}: product RMSE/entry={metrics['root_mean_squared_error_per_entry']:.6g}, "
              f"relative Frobenius error={metrics['relative_frobenius_error']:.6g}")
    print("Scope: toy quality simulation; not paper-scale reproduction or a speed/storage result.")


if __name__ == "__main__":
    main()

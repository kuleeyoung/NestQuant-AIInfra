"""Create a per-layer KV beta config from the public NestQuant defaults."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
UPSTREAM_COMMIT = "6b269f84264afad990a09248d8eeb45d5f6248ff"
KEY_COEFFICIENTS = (3.50, 4.58, 6.47, 17.06)
VALUE_COEFFICIENTS = (3.53, 5.59, 9.62, 29.03)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--q", type=int, default=14)
    parser.add_argument("--layer-count", type=int, default=16)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "configs/kv_betas_upstream_defaults_q14.json",
    )
    args = parser.parse_args()
    if args.q < 2 or args.layer_count < 1:
        raise ValueError("q must be at least 2 and layer count must be positive")

    key_betas = [value / args.q for value in KEY_COEFFICIENTS]
    value_betas = [value / args.q for value in VALUE_COEFFICIENTS]
    payload = {
        "schema_version": 1,
        "status": "completed",
        "model_id": "meta-llama/Llama-3.2-1B",
        "model_revision": "4e20de362430cd3b72f300e6b0f18e50e7166e08",
        "q": args.q,
        "selected_beta_count": 4,
        "selection_method": "public_upstream_global_defaults",
        "upstream_repository": "https://github.com/cookiedoth/nestquant",
        "upstream_commit": UPSTREAM_COMMIT,
        "upstream_file": "parallel/ppl.py",
        "scaling_evidence_file": "coder.py",
        "key_coefficients_project_scale": list(KEY_COEFFICIENTS),
        "value_coefficients_project_scale": list(VALUE_COEFFICIENTS),
        "vector_layout": "one_row_per_attention_head",
        "key_rotation": "normalized_hadamard_after_rope",
        "value_rotation": "none",
        "layers": {
            str(index): {
                "key": {"beta_values": key_betas},
                "value": {"beta_values": value_betas},
            }
            for index in range(args.layer_count)
        },
        "limitations": [
            "The public repository exposes these global defaults but not the beta DP script.",
            "This file maps the public defaults to every Llama-3.2-1B layer for a controlled ablation.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Saved public-upstream KV beta config: {args.output}")


if __name__ == "__main__":
    main()

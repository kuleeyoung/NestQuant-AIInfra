"""Audit the Hessian normalization used by local and public QA-LDLQ paths."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration-dir", type=Path, default=Path("results/raw/calibration"))
    parser.add_argument("--chunks", type=int, default=64)
    parser.add_argument(
        "--variance-file", type=Path,
        default=Path("results/raw/calibration/activation_variance_train20.json"),
    )
    parser.add_argument("--upstream-epsilon-constant", type=float, default=155.0)
    parser.add_argument("--output", type=Path, default=Path("results/raw/qa_ldlq_scale_audit.json"))
    args = parser.parse_args()

    variance = json.loads(args.variance_file.read_text(encoding="utf-8"))
    context_length = int(variance["context_length"])
    equivalent_local_constant = args.upstream_epsilon_constant * context_length
    manifests = sorted(args.calibration_dir.glob(f"hessian_*_train{args.chunks}.pt.json"))
    if not manifests:
        raise FileNotFoundError("no matching Hessian manifests")
    records = []
    for manifest_path in manifests:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["context_length"] != context_length:
            raise ValueError(f"{manifest_path}: context length mismatch")
        if manifest["accumulation"] != "sum(x.T @ x) / activation_rows":
            raise ValueError(f"{manifest_path}: unexpected Hessian normalization")
        if manifest["activation_rows_per_module"] != args.chunks * context_length:
            raise ValueError(f"{manifest_path}: activation-row count mismatch")
        records.append({
            "manifest": str(manifest_path.resolve()),
            "chunk_count": manifest["chunk_count"],
            "activation_rows_per_module": manifest["activation_rows_per_module"],
            "accumulation": manifest["accumulation"],
        })

    payload = {
        "schema_version": 1,
        "status": "completed",
        "local_hessian": "sum(x.T @ x) / (sequence_count * context_length)",
        "public_hessian": "sum(x.T @ x) / sequence_count",
        "public_to_local_hessian_scale": context_length,
        "x_var_normalization_both_paths": "sum(x**2) / activation_element_count",
        "upstream_epsilon_constant": args.upstream_epsilon_constant,
        "equivalent_local_epsilon_constant": equivalent_local_constant,
        "derivation": (
            "H_public=L*H_local. Holding eps^2/H fixed requires "
            "eps_local^2=eps_public^2/L=x_var/(C_public*L)."
        ),
        "context_length": context_length,
        "hessian_manifests": records,
        "variance_file": str(args.variance_file.resolve()),
        "limitations": [
            "This establishes scalar normalization equivalence, not end-to-end paper equivalence.",
            "The local block-LDLQ path additionally applies relative damping 0.01.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

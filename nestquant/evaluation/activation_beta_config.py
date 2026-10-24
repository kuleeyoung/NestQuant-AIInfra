"""Validation and profile resolution for calibrated A4 beta subsets."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from .activation_quantization import projection_kind


SCHEMA_VERSION = 1
KIND = "nestquant_activation_beta_calibration"


def load_activation_beta_config(
    path: Path | str,
    *,
    model_id: str,
    model_revision: str,
    q: int,
    selected_beta_count: int,
    profile: str,
) -> tuple[dict[str, tuple[float, ...]], dict]:
    """Load one calibrated profile and return explicit per-module beta values."""
    config_path = Path(path)
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("activation beta config root must be an object")
    if payload.get("schema_version") != SCHEMA_VERSION or payload.get("kind") != KIND:
        raise ValueError("unsupported activation beta config schema")
    if payload.get("status") != "completed":
        raise ValueError("activation beta calibration is not completed")
    if payload.get("model_id") != model_id or payload.get("model_revision") != model_revision:
        raise ValueError("activation beta config model identity mismatch")
    if payload.get("q") != q or payload.get("selected_beta_count") != selected_beta_count:
        raise ValueError("activation beta config q or beta count mismatch")
    if profile not in {"global", "projection_type", "module"}:
        raise ValueError("activation beta profile must be global, projection_type, or module")
    profiles = payload.get("profiles")
    if not isinstance(profiles, dict) or not isinstance(profiles.get("module"), dict):
        raise ValueError("activation beta config has no module profile")
    module_names = sorted(profiles["module"])
    if not module_names:
        raise ValueError("activation beta config has no calibrated modules")

    def values(record: object) -> tuple[float, ...]:
        if not isinstance(record, dict) or not isinstance(record.get("beta_values"), list):
            raise ValueError("invalid activation beta profile record")
        betas = tuple(float(value) for value in record["beta_values"])
        if len(betas) != selected_beta_count:
            raise ValueError("activation beta profile has wrong beta count")
        if any(not math.isfinite(value) or value <= 0 for value in betas):
            raise ValueError("activation beta values must be positive and finite")
        if any(left >= right for left, right in zip(betas, betas[1:])):
            raise ValueError("activation beta values must be strictly increasing")
        return betas

    resolved: dict[str, tuple[float, ...]] = {}
    if profile == "global":
        shared = values(profiles.get("global", {}).get("global"))
        resolved = {name: shared for name in module_names}
    elif profile == "projection_type":
        typed = profiles.get("projection_type")
        if not isinstance(typed, dict):
            raise ValueError("activation beta config has no projection_type profile")
        resolved = {name: values(typed.get(projection_kind(name))) for name in module_names}
    else:
        resolved = {name: values(profiles["module"].get(name)) for name in module_names}

    metadata = {
        "status": "calibrated_train_split",
        "path": str(config_path.resolve()),
        "sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "profile": profile,
        "module_count": len(resolved),
        "calibration_token_ids_sha256": payload.get("calibration_token_ids_sha256"),
        "candidate_coefficients_project_scale": payload.get(
            "candidate_coefficients_project_scale"
        ),
    }
    return resolved, metadata

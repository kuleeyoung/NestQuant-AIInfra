"""Pure helpers for leakage-safe held-out activation-beta validation."""

from __future__ import annotations

from collections.abc import Sequence

from ..evaluation.calibration_manifest import token_ids_sha256


def select_holdout_windows(
    windows: Sequence[Sequence[int]],
    *,
    calibration_window_count: int,
    holdout_start: int,
    holdout_count: int,
) -> tuple[tuple[tuple[int, ...], ...], dict]:
    """Select a disjoint slice from a validated train-window manifest."""
    if calibration_window_count < 1:
        raise ValueError("calibration_window_count must be positive")
    if holdout_start < calibration_window_count:
        raise ValueError("holdout windows overlap activation-beta calibration windows")
    if holdout_count < 1:
        raise ValueError("holdout_count must be positive")
    end = holdout_start + holdout_count
    if end > len(windows):
        raise ValueError("calibration manifest has too few held-out windows")
    selected = tuple(tuple(int(token) for token in window) for window in windows[holdout_start:end])
    flat = tuple(token for window in selected for token in window)
    metadata = {
        "window_indices": list(range(holdout_start, end)),
        "window_count": len(selected),
        "token_count": len(flat),
        "token_ids_sha256": token_ids_sha256(flat),
        "window_token_ids_sha256": [token_ids_sha256(window) for window in selected],
        "disjoint_from_calibration_window_indices": list(range(calibration_window_count)),
    }
    return selected, metadata


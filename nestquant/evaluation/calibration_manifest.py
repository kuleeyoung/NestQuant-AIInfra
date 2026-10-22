"""Versioned calibration-window manifests shared by quantization backends."""

from __future__ import annotations

import hashlib
import json
import random
import struct
from pathlib import Path
from typing import Mapping, Sequence


SCHEMA_VERSION = 1
MANIFEST_KIND = "llm_calibration_windows"
SELECTION_METHOD = "python_random_randint_with_replacement"
TOKEN_HASH_ENCODING = "ordered_signed_int64_little_endian"


def token_ids_sha256(token_ids: Sequence[int]) -> str:
    """Hash ordered token IDs using a stable cross-platform encoding."""
    digest = hashlib.sha256()
    for token_id in token_ids:
        digest.update(struct.pack("<q", int(token_id)))
    return digest.hexdigest()


def select_calibration_windows(
    token_ids: Sequence[int], start_indices: Sequence[int], context_length: int
) -> tuple[tuple[int, ...], ...]:
    """Return exact manifest-selected windows after validating all bounds."""
    if context_length < 2:
        raise ValueError("context_length must be at least 2")
    if not start_indices:
        raise ValueError("start_indices must not be empty")
    windows: list[tuple[int, ...]] = []
    for index, start in enumerate(start_indices):
        if isinstance(start, bool) or not isinstance(start, int):
            raise TypeError(f"start_indices[{index}] must be an integer")
        if start < 0 or start + context_length > len(token_ids):
            raise ValueError(f"start_indices[{index}] is outside the token corpus")
        windows.append(tuple(int(value) for value in token_ids[start : start + context_length]))
    return tuple(windows)


def build_calibration_manifest(
    token_ids: Sequence[int],
    *,
    model_id: str,
    model_revision: str,
    dataset_id: str,
    dataset_config: str,
    dataset_revision: str,
    context_length: int,
    sample_count: int,
    seed: int,
) -> dict:
    """Build the shared random-with-replacement calibration protocol."""
    if context_length < 2 or sample_count < 1:
        raise ValueError("context_length and sample_count must be positive")
    maximum_start = len(token_ids) - context_length - 1
    if maximum_start < 0:
        raise ValueError("token corpus is too short for one calibration window")
    # Match the single Python RNG stream used by the QuaRot data loader.
    rng = random.Random(seed)
    starts = [rng.randint(0, maximum_start) for _ in range(sample_count)]
    windows = select_calibration_windows(token_ids, starts, context_length)
    selected_ids = tuple(value for window in windows for value in window)
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": MANIFEST_KIND,
        "model": {"id": model_id, "revision": model_revision},
        "dataset": {
            "id": dataset_id,
            "config": dataset_config,
            "split": "train",
            "revision": dataset_revision,
        },
        "tokenization": {
            "row_joiner": "\\n\\n",
            "add_special_tokens": False,
            "boundary_tokens": "none",
            "token_hash_encoding": TOKEN_HASH_ENCODING,
        },
        "selection": {
            "method": SELECTION_METHOD,
            "seed": seed,
            "context_length": context_length,
            "sample_count": sample_count,
            "with_replacement": True,
            "start_indices": starts,
        },
        "train_corpus": {
            "token_count": len(token_ids),
            "token_ids_sha256": token_ids_sha256(token_ids),
        },
        "selected_windows": {
            "token_count": len(selected_ids),
            "token_ids_sha256": token_ids_sha256(selected_ids),
            "window_token_ids_sha256": [token_ids_sha256(window) for window in windows],
        },
    }


def validate_calibration_manifest(
    manifest: Mapping,
    token_ids: Sequence[int],
    *,
    model_id: str | None = None,
    model_revision: str | None = None,
    dataset_id: str | None = None,
    dataset_config: str | None = None,
    dataset_revision: str | None = None,
) -> tuple[tuple[int, ...], ...]:
    """Validate provenance and return the exact selected token windows."""
    if manifest.get("schema_version") != SCHEMA_VERSION or manifest.get("kind") != MANIFEST_KIND:
        raise ValueError("unsupported calibration manifest schema")
    expected = {
        ("model", "id"): model_id,
        ("model", "revision"): model_revision,
        ("dataset", "id"): dataset_id,
        ("dataset", "config"): dataset_config,
        ("dataset", "revision"): dataset_revision,
    }
    for (section, key), value in expected.items():
        if value is not None and manifest.get(section, {}).get(key) != value:
            raise ValueError(f"calibration manifest {section}.{key} mismatch")
    tokenization = manifest.get("tokenization", {})
    if tokenization.get("row_joiner") != "\\n\\n" or tokenization.get("add_special_tokens") is not False:
        raise ValueError("unsupported calibration tokenization protocol")
    if tokenization.get("boundary_tokens") != "none" or tokenization.get("token_hash_encoding") != TOKEN_HASH_ENCODING:
        raise ValueError("unsupported calibration token boundary or hash protocol")
    selection = manifest.get("selection", {})
    if selection.get("method") != SELECTION_METHOD or selection.get("with_replacement") is not True:
        raise ValueError("unsupported calibration selection protocol")
    starts = selection.get("start_indices")
    if not isinstance(starts, list) or selection.get("sample_count") != len(starts):
        raise ValueError("calibration sample count does not match start indices")
    context_length = selection.get("context_length")
    if isinstance(context_length, bool) or not isinstance(context_length, int):
        raise ValueError("invalid calibration context length")
    if manifest.get("train_corpus", {}).get("token_count") != len(token_ids):
        raise ValueError("calibration train corpus token count mismatch")
    if manifest.get("train_corpus", {}).get("token_ids_sha256") != token_ids_sha256(token_ids):
        raise ValueError("calibration train corpus token hash mismatch")
    windows = select_calibration_windows(token_ids, starts, context_length)
    selected_ids = tuple(value for window in windows for value in window)
    selected = manifest.get("selected_windows", {})
    if selected.get("token_count") != len(selected_ids):
        raise ValueError("calibration selected token count mismatch")
    if selected.get("token_ids_sha256") != token_ids_sha256(selected_ids):
        raise ValueError("calibration selected token hash mismatch")
    window_hashes = selected.get("window_token_ids_sha256")
    expected_window_hashes = [token_ids_sha256(window) for window in windows]
    if window_hashes != expected_window_hashes:
        raise ValueError("calibration per-window token hashes mismatch")
    return windows


def load_calibration_manifest(path: Path | str) -> dict:
    """Load a manifest without weakening validation error messages."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("calibration manifest root must be an object")
    return payload


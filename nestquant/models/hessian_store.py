"""Lazy access to grouped Llama calibration Hessians."""

from __future__ import annotations

import gc
import json
from pathlib import Path
import re

import torch


class LlamaHessianStore:
    """Load one four-layer calibration group at a time and map shared inputs."""

    _LAYER_PATTERN = re.compile(r"^model\.layers\.(\d+)\.(.+)$")

    def __init__(
        self,
        directory: Path | str,
        *,
        calibration_chunks: int,
        expected_model_id: str,
        expected_model_revision: str,
        expected_token_hash: str | None = None,
    ) -> None:
        self.directory = Path(directory)
        self.calibration_chunks = calibration_chunks
        self.expected_model_id = expected_model_id
        self.expected_model_revision = expected_model_revision
        self.expected_token_hash = expected_token_hash
        self._loaded_path: Path | None = None
        self._loaded: dict[str, torch.Tensor] | None = None

    def resolve(self, module_name: str) -> tuple[str, torch.Tensor]:
        key, path = self._source_key_and_path(module_name)
        if path != self._loaded_path:
            self._load(path)
        assert self._loaded is not None
        if key not in self._loaded:
            raise KeyError(f"{path} does not contain Hessian {key}")
        return key, self._loaded[key]

    def close(self) -> None:
        self._loaded = None
        self._loaded_path = None
        gc.collect()

    def _source_key_and_path(self, module_name: str) -> tuple[str, Path]:
        if module_name == "lm_head":
            return module_name, self.directory / f"hessian_lm_head_train{self.calibration_chunks}.pt"
        match = self._LAYER_PATTERN.match(module_name)
        if not match:
            raise KeyError(f"unsupported calibrated module: {module_name}")
        layer = int(match.group(1))
        suffix = match.group(2)
        if suffix in {"self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"}:
            source_suffix = "self_attn.q_proj"
        elif suffix == "self_attn.o_proj":
            source_suffix = suffix
        elif suffix in {"mlp.gate_proj", "mlp.up_proj"}:
            source_suffix = "mlp.gate_proj"
        elif suffix == "mlp.down_proj":
            source_suffix = suffix
        else:
            raise KeyError(f"unsupported calibrated module: {module_name}")
        group_start = (layer // 4) * 4
        group_end = group_start + 3
        path = self.directory / (
            f"hessian_layers{group_start:02d}_{group_end:02d}_train{self.calibration_chunks}.pt"
        )
        return f"model.layers.{layer}.{source_suffix}", path

    def _load(self, path: Path) -> None:
        if not path.is_file():
            raise FileNotFoundError(path)
        manifest_path = path.with_suffix(path.suffix + ".json")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["model_id"] != self.expected_model_id:
            raise ValueError(f"{path}: model ID mismatch")
        if manifest["model_revision"] != self.expected_model_revision:
            raise ValueError(f"{path}: model revision mismatch")
        if manifest["chunk_count"] != self.calibration_chunks:
            raise ValueError(f"{path}: calibration chunk mismatch")
        if self.expected_token_hash is None:
            self.expected_token_hash = manifest["train_token_ids_sha256"]
        elif manifest["train_token_ids_sha256"] != self.expected_token_hash:
            raise ValueError(f"{path}: train token hash mismatch")
        self._loaded = None
        gc.collect()
        loaded = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(loaded, dict) or not all(
            isinstance(value, torch.Tensor) for value in loaded.values()
        ):
            raise ValueError(f"{path}: invalid Hessian mapping")
        self._loaded = loaded
        self._loaded_path = path

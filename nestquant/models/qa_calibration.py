"""Validated activation-variance artifacts for QA-LDLQ."""

from __future__ import annotations

import json
from pathlib import Path


class ActivationVarianceStore:
    def __init__(
        self,
        path: Path,
        *,
        expected_model_id: str,
        expected_model_revision: str,
        expected_chunks: int,
    ) -> None:
        self.path = Path(path)
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        if payload.get("status") != "completed":
            raise ValueError("activation variance artifact is not completed")
        if payload.get("model_id") != expected_model_id:
            raise ValueError("activation variance model ID mismatch")
        if payload.get("model_revision") != expected_model_revision:
            raise ValueError("activation variance model revision mismatch")
        if payload.get("calibration_chunks") != expected_chunks:
            raise ValueError("activation variance calibration chunk mismatch")
        records = payload.get("modules")
        if not isinstance(records, dict) or not records:
            raise ValueError("activation variance artifact has no module records")
        self.payload = payload
        self.records = records

    def resolve(self, module_name: str) -> tuple[str, float] | None:
        if module_name == "lm_head":
            return None
        record = self.records.get(module_name)
        if not isinstance(record, dict) or "x_var" not in record:
            raise KeyError(f"activation variance missing for {module_name}")
        value = float(record["x_var"])
        if value < 0:
            raise ValueError(f"negative activation variance for {module_name}")
        return str(record.get("source_module", module_name)), value

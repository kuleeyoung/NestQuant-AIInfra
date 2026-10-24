import json
import tempfile
import unittest
from pathlib import Path

from nestquant.evaluation.activation_beta_config import (
    KIND,
    SCHEMA_VERSION,
    load_activation_beta_config,
)


class TestActivationBetaConfig(unittest.TestCase):
    def write_config(self, root: Path) -> Path:
        modules = {
            "model.layers.0.self_attn.q_proj": {"beta_values": [0.2, 0.4]},
            "model.layers.0.mlp.down_proj": {"beta_values": [0.3, 0.7]},
        }
        payload = {
            "schema_version": SCHEMA_VERSION,
            "kind": KIND,
            "status": "completed",
            "model_id": "model",
            "model_revision": "revision",
            "q": 14,
            "selected_beta_count": 2,
            "calibration_token_ids_sha256": "abc",
            "candidate_coefficients_project_scale": [1, 2, 3],
            "profiles": {
                "global": {"global": {"beta_values": [0.25, 0.5]}},
                "projection_type": {
                    "q_proj": {"beta_values": [0.2, 0.4]},
                    "down_proj": {"beta_values": [0.3, 0.7]},
                },
                "module": modules,
            },
        }
        path = root / "a4.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def test_resolves_each_profile_to_explicit_module_mapping(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_config(Path(directory))
            global_values, metadata = load_activation_beta_config(
                path, model_id="model", model_revision="revision", q=14,
                selected_beta_count=2, profile="global",
            )
            typed_values, _ = load_activation_beta_config(
                path, model_id="model", model_revision="revision", q=14,
                selected_beta_count=2, profile="projection_type",
            )
            module_values, _ = load_activation_beta_config(
                path, model_id="model", model_revision="revision", q=14,
                selected_beta_count=2, profile="module",
            )
        self.assertEqual(set(global_values.values()), {(0.25, 0.5)})
        self.assertEqual(typed_values, module_values)
        self.assertEqual(metadata["status"], "calibrated_train_split")
        self.assertEqual(metadata["module_count"], 2)

    def test_rejects_model_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_config(Path(directory))
            with self.assertRaisesRegex(ValueError, "model identity"):
                load_activation_beta_config(
                    path, model_id="other", model_revision="revision", q=14,
                    selected_beta_count=2, profile="module",
                )


if __name__ == "__main__":
    unittest.main()

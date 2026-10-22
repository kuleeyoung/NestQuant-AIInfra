import json
from pathlib import Path
import tempfile
import unittest

from nestquant.models.qa_calibration import ActivationVarianceStore


class TestActivationVarianceStore(unittest.TestCase):
    def test_validates_and_resolves_module_variance(self):
        payload = {
            "status": "completed", "model_id": "model/test",
            "model_revision": "revision", "calibration_chunks": 20,
            "modules": {"model.layers.0.self_attn.q_proj": {
                "source_module": "model.layers.0.self_attn.q_proj", "x_var": 0.25,
            }},
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "variance.json"
            path.write_text(json.dumps(payload), "utf-8")
            store = ActivationVarianceStore(
                path, expected_model_id="model/test",
                expected_model_revision="revision", expected_chunks=20,
            )
            self.assertEqual(
                store.resolve("model.layers.0.self_attn.q_proj"),
                ("model.layers.0.self_attn.q_proj", 0.25),
            )
            self.assertIsNone(store.resolve("lm_head"))

    def test_rejects_protocol_mismatch(self):
        payload = {
            "status": "completed", "model_id": "model/test",
            "model_revision": "revision", "calibration_chunks": 6,
            "modules": {"x": {"x_var": 1.0}},
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "variance.json"
            path.write_text(json.dumps(payload), "utf-8")
            with self.assertRaisesRegex(ValueError, "chunk mismatch"):
                ActivationVarianceStore(
                    path, expected_model_id="model/test",
                    expected_model_revision="revision", expected_chunks=20,
                )


if __name__ == "__main__":
    unittest.main()

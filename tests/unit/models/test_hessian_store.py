import json
from pathlib import Path
import tempfile
import unittest

import torch

from nestquant.models.hessian_store import LlamaHessianStore


class TestHessianStore(unittest.TestCase):
    def test_maps_shared_inputs_and_validates_manifest(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path = root / "hessian_layers00_03_train16.pt"
            tensor = torch.eye(8)
            torch.save({
                "model.layers.0.self_attn.q_proj": tensor,
                "model.layers.0.mlp.gate_proj": tensor * 2,
            }, path)
            path.with_suffix(".pt.json").write_text(json.dumps({
                "model_id": "model", "model_revision": "rev", "chunk_count": 16,
                "train_token_ids_sha256": "hash",
            }), encoding="utf-8")
            store = LlamaHessianStore(
                root, calibration_chunks=16, expected_model_id="model",
                expected_model_revision="rev", expected_token_hash="hash",
            )
            key, value = store.resolve("model.layers.0.self_attn.k_proj")
            self.assertEqual(key, "model.layers.0.self_attn.q_proj")
            torch.testing.assert_close(value, tensor)
            key, value = store.resolve("model.layers.0.mlp.up_proj")
            self.assertEqual(key, "model.layers.0.mlp.gate_proj")
            torch.testing.assert_close(value, tensor * 2)

    def test_rejects_unknown_module(self):
        store = LlamaHessianStore(
            ".", calibration_chunks=16, expected_model_id="m", expected_model_revision="r"
        )
        with self.assertRaises(KeyError):
            store.resolve("model.layers.0.unknown")


if __name__ == "__main__":
    unittest.main()

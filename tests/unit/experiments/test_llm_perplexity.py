"""Focused tests for the Llama PPL experiment's audit metadata."""

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from nestquant.experiments.llm_perplexity import (
    _ldlq_calibration_metadata,
    _load_kv_beta_config,
)


class TestLdlqCalibrationMetadata(unittest.TestCase):
    def test_collects_consistent_group_manifests_without_loading_hessians(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            names_and_counts = [
                ("hessian_layers00_03_train64.pt", 16),
                ("hessian_layers04_07_train64.pt", 16),
                ("hessian_layers08_11_train64.pt", 16),
                ("hessian_layers12_15_train64.pt", 16),
                ("hessian_lm_head_train64.pt", 1),
            ]
            for name, module_count in names_and_counts:
                tensor_path = directory / name
                tensor_path.write_bytes(b"test-hessian")
                manifest = {
                    "model_id": "model/test",
                    "model_revision": "revision-1",
                    "dataset": "WikiText-2 train",
                    "dataset_revision": "dataset-revision-1",
                    "context_length": 32,
                    "chunk_count": 64,
                    "calibration_token_ids_sha256": "a" * 64,
                    "train_corpus_token_ids_sha256": "b" * 64,
                    "output_bytes": tensor_path.stat().st_size,
                    "output_sha256": hashlib.sha256(tensor_path.read_bytes()).hexdigest(),
                    "modules": {f"module-{index}": {} for index in range(module_count)},
                }
                tensor_path.with_suffix(".pt.json").write_text(
                    json.dumps(manifest), encoding="utf-8"
                )

            metadata = _ldlq_calibration_metadata(
                directory, 64, "model/test", "revision-1", 0.01
            )

            self.assertEqual(metadata["algorithm"], "8-column E8 block-LDLQ")
            self.assertEqual(metadata["dataset_revision"], "dataset-revision-1")
            self.assertEqual(metadata["context_length"], 32)
            self.assertEqual(metadata["calibration_tokens"], 2048)
            self.assertEqual(metadata["calibration_token_ids_sha256"], "a" * 64)
            self.assertEqual(len(metadata["hessian_artifacts"]), 5)
            self.assertEqual(sum(a["module_count"] for a in metadata["hessian_artifacts"]), 65)

    def test_rejects_inconsistent_context_length(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            names_and_counts = [
                ("hessian_layers00_03_train1.pt", 16),
                ("hessian_layers04_07_train1.pt", 16),
                ("hessian_layers08_11_train1.pt", 16),
                ("hessian_layers12_15_train1.pt", 16),
                ("hessian_lm_head_train1.pt", 1),
            ]
            for file_index, (name, module_count) in enumerate(names_and_counts):
                tensor_path = directory / name
                tensor_path.write_bytes(b"x")
                manifest = {
                    "model_id": "model/test", "model_revision": "revision-1",
                    "dataset": "WikiText-2 train", "dataset_revision": "dataset-revision-1",
                    "context_length": 16 if file_index == 4 else 32,
                    "chunk_count": 1, "calibration_token_ids_sha256": "a" * 64,
                    "train_corpus_token_ids_sha256": "b" * 64,
                    "output_bytes": 1, "output_sha256": "c" * 64,
                    "modules": {f"module-{index}": {} for index in range(module_count)},
                }
                tensor_path.with_suffix(".pt.json").write_text(
                    json.dumps(manifest), encoding="utf-8"
                )

            with self.assertRaisesRegex(ValueError, "calibration protocol metadata disagrees"):
                _ldlq_calibration_metadata(
                    directory, 1, "model/test", "revision-1", 0.01
                )


class TestKvBetaConfig(unittest.TestCase):
    def test_loads_separate_key_and_value_subsets(self):
        payload = {
            "status": "completed",
            "model_id": "model/test",
            "model_revision": "revision-1",
            "q": 14,
            "selected_beta_count": 2,
            "dataset": "dataset/test",
            "dataset_config": "raw",
            "dataset_revision": "data-revision",
            "split": "train",
            "context_length": 16,
            "calibration_chunks": 2,
            "token_ids_sha256": "a" * 64,
            "candidate_beta_values": [0.25, 0.5, 1.0],
            "sample_limit_per_tensor": 128,
            "layers": {
                "0": {
                    "key": {"beta_values": [0.25, 1.0]},
                    "value": {"beta_values": [0.5, 1.0]},
                }
            },
        }
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "kv-betas.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            layers, metadata = _load_kv_beta_config(
                path, model_id="model/test", model_revision="revision-1",
                q=14, selected_beta_count=2,
            )
        self.assertEqual(layers[0]["key"], (0.25, 1.0))
        self.assertEqual(layers[0]["value"], (0.5, 1.0))
        self.assertEqual(metadata["layer_count"], 1)
        self.assertEqual(len(metadata["sha256"]), 64)

    def test_rejects_wrong_model(self):
        payload = {
            "status": "completed", "model_id": "wrong", "model_revision": "r",
            "q": 14, "selected_beta_count": 1,
            "layers": {"0": {"key": {"beta_values": [1.0]},
                               "value": {"beta_values": [1.0]}}},
        }
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "kv-betas.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "model identity"):
                _load_kv_beta_config(
                    path, model_id="model/test", model_revision="r",
                    q=14, selected_beta_count=1,
                )


if __name__ == "__main__":
    unittest.main()

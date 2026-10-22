import copy
import unittest

from nestquant.evaluation.calibration_manifest import (
    build_calibration_manifest,
    select_calibration_windows,
    token_ids_sha256,
    validate_calibration_manifest,
)


class TestCalibrationManifest(unittest.TestCase):
    def setUp(self):
        self.tokens = tuple(range(100))
        self.metadata = dict(
            model_id="model",
            model_revision="model-rev",
            dataset_id="dataset",
            dataset_config="config",
            dataset_revision="data-rev",
        )

    def test_build_is_reproducible_and_validation_returns_exact_windows(self):
        first = build_calibration_manifest(
            self.tokens, context_length=8, sample_count=5, seed=58, **self.metadata
        )
        second = build_calibration_manifest(
            self.tokens, context_length=8, sample_count=5, seed=58, **self.metadata
        )
        self.assertEqual(first, second)
        windows = validate_calibration_manifest(first, self.tokens, **self.metadata)
        expected = select_calibration_windows(
            self.tokens, first["selection"]["start_indices"], 8
        )
        self.assertEqual(windows, expected)
        flattened = tuple(value for window in windows for value in window)
        self.assertEqual(
            first["selected_windows"]["token_ids_sha256"], token_ids_sha256(flattened)
        )
        self.assertEqual(
            first["selected_windows"]["window_token_ids_sha256"],
            [token_ids_sha256(window) for window in windows],
        )

    def test_rejects_corpus_provenance_and_selected_hash_changes(self):
        manifest = build_calibration_manifest(
            self.tokens, context_length=8, sample_count=5, seed=58, **self.metadata
        )
        with self.assertRaisesRegex(ValueError, "corpus token hash"):
            validate_calibration_manifest(manifest, self.tokens[:-1] + (999,), **self.metadata)
        changed = copy.deepcopy(manifest)
        changed["selected_windows"]["token_ids_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "selected token hash"):
            validate_calibration_manifest(changed, self.tokens, **self.metadata)
        changed = copy.deepcopy(manifest)
        changed["selected_windows"]["window_token_ids_sha256"][0] = "0" * 64
        with self.assertRaisesRegex(ValueError, "per-window token hashes"):
            validate_calibration_manifest(changed, self.tokens, **self.metadata)

    def test_rejects_wrong_metadata_and_out_of_bounds_start(self):
        manifest = build_calibration_manifest(
            self.tokens, context_length=8, sample_count=5, seed=58, **self.metadata
        )
        with self.assertRaisesRegex(ValueError, "model.id mismatch"):
            validate_calibration_manifest(manifest, self.tokens, model_id="other")
        changed = copy.deepcopy(manifest)
        changed["selection"]["start_indices"][0] = len(self.tokens)
        with self.assertRaisesRegex(ValueError, "outside"):
            validate_calibration_manifest(changed, self.tokens, **self.metadata)


if __name__ == "__main__":
    unittest.main()

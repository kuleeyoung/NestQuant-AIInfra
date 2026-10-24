import unittest

from nestquant.experiments.activation_beta_validation import select_holdout_windows


class ActivationBetaValidationTests(unittest.TestCase):
    def test_selects_exact_disjoint_slice_with_stable_metadata(self):
        windows = tuple((index, index + 10) for index in range(6))
        selected, metadata = select_holdout_windows(
            windows, calibration_window_count=2, holdout_start=3, holdout_count=2
        )
        self.assertEqual(selected, ((3, 13), (4, 14)))
        self.assertEqual(metadata["window_indices"], [3, 4])
        self.assertEqual(metadata["token_count"], 4)
        self.assertEqual(len(metadata["token_ids_sha256"]), 64)
        self.assertEqual(len(metadata["window_token_ids_sha256"]), 2)

    def test_rejects_overlap_and_out_of_range(self):
        windows = ((1, 2),) * 4
        with self.assertRaisesRegex(ValueError, "overlap"):
            select_holdout_windows(
                windows, calibration_window_count=2, holdout_start=1, holdout_count=1
            )
        with self.assertRaisesRegex(ValueError, "too few"):
            select_holdout_windows(
                windows, calibration_window_count=2, holdout_start=3, holdout_count=2
            )


if __name__ == "__main__":
    unittest.main()


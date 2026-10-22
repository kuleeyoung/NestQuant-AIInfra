"""Tests for deterministic synthetic matrix-product quality measurements."""

import unittest
import math

from nestquant.experiments.matrix_product import run_synthetic_matrix_product


class TestSyntheticMatrixProduct(unittest.TestCase):
    def test_run_is_reproducible_and_reports_scope(self):
        kwargs = dict(n=8, m=8, k=8, q=14, beta_count=2, seed=58, device="cpu")
        first = run_synthetic_matrix_product(**kwargs)
        second = run_synthetic_matrix_product(**kwargs)
        self.assertEqual(first["config"], second["config"])
        self.assertEqual(first["metrics"], second["metrics"])
        self.assertIn("rtn_w4_per_row", first["metrics"])
        self.assertIn("uniform_linf_w4_block8", first["metrics"])
        self.assertIn("nestquant_w_quality_simulation", first["metrics"])
        self.assertTrue(any("toy dimensions" in item for item in first["limitations"]))
        self.assertAlmostEqual(first["config"]["beta_candidate_codec_scales"][0],
                               first["config"]["beta_candidate_coefficients_paper_scale"][0] / kwargs["q"])
        self.assertAlmostEqual(first["rate_estimate"]["shannon_side_information_bits_per_entry"],
                               first["rate_estimate"]["paper_formula_bits_per_entry"] +
                               (-sum(p * math.log2(p) for p in
                                     first["rate_estimate"]["beta_use_probabilities"] if p > 0) / 4))

    def test_requires_eight_dimensional_blocks(self):
        with self.assertRaises(ValueError):
            run_synthetic_matrix_product(n=8, m=8, k=10, beta_count=2)


if __name__ == "__main__":
    unittest.main()

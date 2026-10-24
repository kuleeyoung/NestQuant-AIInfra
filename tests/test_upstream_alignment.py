import unittest

import torch

from nestquant.evaluation.upstream_alignment import tensor_alignment


class UpstreamAlignmentTests(unittest.TestCase):
    def test_reports_exact_and_known_difference(self):
        reference = torch.tensor([[3.0, 4.0]], dtype=torch.float32)
        exact = tensor_alignment(reference, reference.clone())
        self.assertEqual(exact["max_abs_error"], 0.0)
        self.assertEqual(exact["relative_l2_error"], 0.0)
        changed = tensor_alignment(
            reference, torch.tensor([[3.0, 5.0]], dtype=torch.float32)
        )
        self.assertEqual(changed["max_abs_error"], 1.0)
        self.assertAlmostEqual(changed["rmse"], 2 ** -0.5)
        self.assertAlmostEqual(changed["relative_l2_error"], 0.2)

    def test_rejects_invalid_comparisons(self):
        with self.assertRaisesRegex(ValueError, "shapes"):
            tensor_alignment(torch.ones(2), torch.ones(3))
        with self.assertRaisesRegex(TypeError, "floating"):
            tensor_alignment(torch.ones(2, dtype=torch.int64), torch.ones(2))
        with self.assertRaisesRegex(ValueError, "finite"):
            tensor_alignment(torch.tensor([float("nan")]), torch.ones(1))


if __name__ == "__main__":
    unittest.main()


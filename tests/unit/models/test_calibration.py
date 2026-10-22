import unittest

import torch
from torch import nn

from nestquant.models.calibration import LinearInputCovarianceCollector


class TestCalibrationCollector(unittest.TestCase):
    def test_collects_exact_linear_inputs_and_removes_hooks(self):
        model = nn.Sequential(nn.Linear(3, 4), nn.ReLU(), nn.Linear(4, 2)).double()
        values = torch.tensor([[1.0, 2.0, 3.0], [-1.0, 0.5, 2.0]], dtype=torch.float64)
        first_output = model[1](model[0](values))
        with LinearInputCovarianceCollector(
            model, ("0", "2"), accumulator_dtype=torch.float64
        ) as collector:
            model(values)
            covariances = collector.covariances()
            metadata = collector.metadata()
        torch.testing.assert_close(covariances["0"], values.T @ values / 2)
        torch.testing.assert_close(covariances["2"], first_output.T @ first_output / 2)
        self.assertEqual(metadata["0"]["sample_count"], 2)
        self.assertEqual(metadata["2"]["feature_count"], 4)
        before = collector.accumulators["0"].sample_count
        model(values)
        self.assertEqual(collector.accumulators["0"].sample_count, before)

    def test_rejects_unknown_duplicate_and_non_linear_modules(self):
        model = nn.Sequential(nn.Linear(3, 4), nn.ReLU())
        with self.assertRaisesRegex(ValueError, "unknown"):
            LinearInputCovarianceCollector(model, ("missing",))
        with self.assertRaisesRegex(ValueError, "unique"):
            LinearInputCovarianceCollector(model, ("0", "0"))
        with self.assertRaisesRegex(ValueError, "not nn.Linear"):
            LinearInputCovarianceCollector(model, ("1",))


if __name__ == "__main__":
    unittest.main()

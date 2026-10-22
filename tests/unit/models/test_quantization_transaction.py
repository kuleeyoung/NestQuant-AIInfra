"""Model adapters must not leave partially quantized models on failure."""

import unittest

import torch
from torch import nn

from nestquant.models.quantization import quantize_nestquant_w4_, quantize_rtn_w4_


class TestQuantizationTransaction(unittest.TestCase):
    def _assert_failure_preserves_weights(self, model, call):
        before = [module.weight.detach().clone() for module in model.modules()
                  if isinstance(module, nn.Linear)]
        with self.assertRaises(ValueError):
            call(model)
        after = [module.weight.detach() for module in model.modules()
                 if isinstance(module, nn.Linear)]
        self.assertEqual(len(before), len(after))
        for expected, actual in zip(before, after):
            self.assertTrue(torch.equal(expected, actual))

    def test_rtn_failure_does_not_modify_earlier_layers(self):
        model = nn.Sequential(nn.Linear(4, 4), nn.Linear(3, 4))
        self._assert_failure_preserves_weights(
            model, lambda candidate: quantize_rtn_w4_(candidate, group_size=2))

    def test_nestquant_failure_does_not_modify_earlier_layers(self):
        model = nn.Sequential(nn.Linear(8, 8), nn.Linear(7, 8))
        self._assert_failure_preserves_weights(
            model,
            lambda candidate: quantize_nestquant_w4_(
                candidate, q=14, beta_coefficients=(3.5, 4.5, 6.0, 14.5, 25.0),
                selected_beta_count=4, seed=58, rows_per_chunk=4),
        )


if __name__ == "__main__":
    unittest.main()

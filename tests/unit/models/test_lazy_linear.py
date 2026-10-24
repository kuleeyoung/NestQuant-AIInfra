"""Tests for the single-layer lazy NQMC inference prototype."""

from pathlib import Path
import tempfile
import unittest

import torch
from torch.nn import functional as F

from nestquant.core.model_checkpoint import PackedNestQuantModel
from nestquant.core.model_checkpoint_file import NQMCFileReader
from nestquant.core.quantizer import quantize_rows
from nestquant.core.rotation import hadamard
from nestquant.core.tensor_checkpoint import PackedNestQuantTensor
from nestquant.models.lazy_linear import LazyNQMCLinear


class TestLazyNQMCLinear(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.temporary = tempfile.TemporaryDirectory()
        self.weight = torch.randn(5, 8)
        rows = quantize_rows(self.weight, q=14, betas=(0.5, 0.75, 1.0, 1.25))
        record = PackedNestQuantTensor.from_quantized_rows("proj.weight", rows)
        state = {
            "proj.weight": self.weight,
            "dense.weight": torch.randn(3, 8),
        }
        path = Path(self.temporary.name) / "tiny.nqmc"
        PackedNestQuantModel.from_state_dict(
            state, {"proj.weight": record}
        ).write_to_file(path)
        self.reader = NQMCFileReader(path)

    def tearDown(self):
        self.temporary.cleanup()

    def test_no_cache_matches_explicit_decode_and_preserves_input_dtype(self):
        inputs = torch.randn(3, 8)
        bias = torch.randn(5)
        layer = LazyNQMCLinear(
            self.reader, "proj.weight", bias, compute_dtype=torch.bfloat16,
            rows_per_chunk=2, cache_policy="none",
        )
        expected_weight = self.reader.decode_tensor(
            "proj.weight", output_dtype=torch.bfloat16, rows_per_chunk=2
        )
        expected = F.linear(inputs.to(torch.bfloat16), expected_weight,
                            bias.to(torch.bfloat16)).float()
        actual = layer(inputs)
        self.assertTrue(torch.equal(actual, expected))
        self.assertEqual(actual.dtype, inputs.dtype)
        self.assertFalse(layer.cached)
        self.assertEqual(layer.cached_weight_bytes, 0)

    def test_persistent_cache_reuses_weight_and_can_be_cleared(self):
        inputs = torch.randn(2, 8, dtype=torch.bfloat16)
        layer = LazyNQMCLinear(
            self.reader, "proj.weight", compute_dtype=torch.bfloat16,
            rows_per_chunk=1, cache_policy="persistent",
        )
        first = layer(inputs)
        pointer = layer._cached_weight.data_ptr()
        second = layer(inputs)
        self.assertTrue(torch.equal(first, second))
        self.assertTrue(layer.cached)
        self.assertEqual(layer._cached_weight.data_ptr(), pointer)
        self.assertEqual(layer.cached_weight_bytes, 5 * 8 * 2)
        self.assertNotIn("_cached_weight", layer.state_dict())
        layer.clear_cache()
        self.assertFalse(layer.cached)

    def test_optional_hadamard_matches_explicit_reference(self):
        inputs = torch.randn(4, 8, dtype=torch.bfloat16)
        layer = LazyNQMCLinear(
            self.reader, "proj.weight", input_hadamard=True,
            compute_dtype=torch.bfloat16, output_dtype=torch.float32,
            rows_per_chunk=2,
        )
        weight = self.reader.decode_tensor(
            "proj.weight", output_dtype=torch.bfloat16, rows_per_chunk=2
        )
        expected = F.linear(hadamard(inputs), weight).float()
        self.assertTrue(torch.equal(layer(inputs), expected))

    def test_rejects_dense_weight_invalid_options_and_bad_input(self):
        with self.assertRaisesRegex(ValueError, "packed NestQuant"):
            LazyNQMCLinear(self.reader, "dense.weight")
        with self.assertRaisesRegex(ValueError, "cache_policy"):
            LazyNQMCLinear(self.reader, "proj.weight", cache_policy="layer")
        with self.assertRaisesRegex(ValueError, "rows_per_chunk"):
            LazyNQMCLinear(self.reader, "proj.weight", rows_per_chunk=0)
        with self.assertRaisesRegex(ValueError, "bias"):
            LazyNQMCLinear(self.reader, "proj.weight", torch.ones(4))
        layer = LazyNQMCLinear(self.reader, "proj.weight")
        with self.assertRaisesRegex(ValueError, "feature dimension"):
            layer(torch.randn(2, 7))


if __name__ == "__main__":
    unittest.main()

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
from nestquant.models.lazy_linear import LazyNQMCLinear, LazyNQMCWeightCache


class TestLazyNQMCLinear(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.temporary = tempfile.TemporaryDirectory()
        self.weight = torch.randn(5, 8)
        rows = quantize_rows(self.weight, q=14, betas=(0.5, 0.75, 1.0, 1.25))
        record = PackedNestQuantTensor.from_quantized_rows("proj.weight", rows)
        state = {
            "proj.weight": self.weight,
            "proj_b.weight": torch.randn(5, 8),
            "proj_c.weight": torch.randn(5, 8),
            "dense.weight": torch.randn(3, 8),
        }
        records = {"proj.weight": record}
        for name in ("proj_b.weight", "proj_c.weight"):
            records[name] = PackedNestQuantTensor.from_quantized_rows(
                name, quantize_rows(
                    state[name], q=14, betas=(0.5, 0.75, 1.0, 1.25)
                )
            )
        path = Path(self.temporary.name) / "tiny.nqmc"
        PackedNestQuantModel.from_state_dict(
            state, records
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
        with self.assertRaisesRegex(ValueError, "requires"):
            LazyNQMCLinear(self.reader, "proj.weight", cache_policy="lru")
        with self.assertRaisesRegex(ValueError, "only valid"):
            LazyNQMCLinear(
                self.reader, "proj.weight", weight_cache=LazyNQMCWeightCache(80)
            )
        with self.assertRaisesRegex(ValueError, "rows_per_chunk"):
            LazyNQMCLinear(self.reader, "proj.weight", rows_per_chunk=0)
        with self.assertRaisesRegex(ValueError, "bias"):
            LazyNQMCLinear(self.reader, "proj.weight", torch.ones(4))
        layer = LazyNQMCLinear(self.reader, "proj.weight")
        with self.assertRaisesRegex(ValueError, "feature dimension"):
            layer(torch.randn(2, 7))

    def test_shared_lru_obeys_capacity_and_updates_recency(self):
        cache = LazyNQMCWeightCache(160)  # exactly two 5 x 8 BF16 weights
        layers = {
            name: LazyNQMCLinear(
                self.reader, f"{name}.weight", compute_dtype=torch.bfloat16,
                rows_per_chunk=2, cache_policy="lru", weight_cache=cache,
            )
            for name in ("proj", "proj_b", "proj_c")
        }
        inputs = torch.randn(2, 8, dtype=torch.bfloat16)
        outputs = [layers[name](inputs) for name in ("proj", "proj_b", "proj", "proj_c")]
        for output, name in zip(outputs, ("proj", "proj_b", "proj", "proj_c")):
            weight = self.reader.decode_tensor(
                f"{name}.weight", output_dtype=torch.bfloat16, rows_per_chunk=2
            )
            self.assertTrue(torch.equal(output, F.linear(inputs, weight)))
        stats = cache.snapshot()
        self.assertEqual(stats["hits"], 1)
        self.assertEqual(stats["misses"], 3)
        self.assertEqual(stats["decodes"], 3)
        self.assertEqual(stats["evictions"], 1)
        self.assertEqual(stats["resident_bytes"], 160)
        self.assertEqual(stats["peak_resident_bytes"], 160)
        self.assertTrue(layers["proj"].cached)
        self.assertFalse(layers["proj_b"].cached)
        self.assertTrue(layers["proj_c"].cached)

    def test_oversized_weight_bypasses_lru_without_exceeding_budget(self):
        cache = LazyNQMCWeightCache(79)
        layer = LazyNQMCLinear(
            self.reader, "proj.weight", compute_dtype=torch.bfloat16,
            cache_policy="lru", weight_cache=cache,
        )
        inputs = torch.randn(1, 8, dtype=torch.bfloat16)
        first = layer(inputs)
        second = layer(inputs)
        self.assertTrue(torch.equal(first, second))
        self.assertFalse(layer.cached)
        self.assertEqual(cache.snapshot(), {
            "capacity_bytes": 79, "resident_bytes": 0,
            "peak_resident_bytes": 0, "resident_tensor_count": 0,
            "hits": 0, "misses": 2, "decodes": 2,
            "evictions": 0, "bypasses": 2, "hit_rate": 0.0,
        })

    def test_cache_validates_budget_and_can_reset_statistics(self):
        with self.assertRaisesRegex(TypeError, "integer"):
            LazyNQMCWeightCache(True)
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            LazyNQMCWeightCache(-1)
        cache = LazyNQMCWeightCache(80)
        layer = LazyNQMCLinear(
            self.reader, "proj.weight", compute_dtype=torch.bfloat16,
            cache_policy="lru", weight_cache=cache,
        )
        layer(torch.randn(1, 8, dtype=torch.bfloat16))
        cache.clear(reset_statistics=True)
        self.assertEqual(cache.snapshot()["resident_bytes"], 0)
        self.assertEqual(cache.snapshot()["misses"], 0)

    def test_profile_callback_splits_read_parse_decode_transform_and_gemm(self):
        events = []
        layer = LazyNQMCLinear(
            self.reader, "proj.weight", input_hadamard=True,
            compute_dtype=torch.bfloat16, rows_per_chunk=2,
            profile_callback=events.append, synchronize_profile=True,
        )
        layer(torch.randn(2, 8, dtype=torch.bfloat16))
        self.assertEqual([event["phase"] for event in events], [
            "input_transform", "record_read", "record_parse",
            "unpack_dequant", "gemm_output_cast",
        ])
        self.assertTrue(all(event["seconds"] >= 0 for event in events))
        self.assertTrue(all(event["weight_name"] == "proj.weight" for event in events))

    def test_synchronized_profile_requires_callback(self):
        with self.assertRaisesRegex(ValueError, "requires"):
            LazyNQMCLinear(
                self.reader, "proj.weight", synchronize_profile=True
            )


if __name__ == "__main__":
    unittest.main()

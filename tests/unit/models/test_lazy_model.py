"""Tests for transactionally converting an entire model to lazy NQMC layers."""

from pathlib import Path
import tempfile
import unittest

import torch
from torch import nn

from nestquant.core.model_checkpoint import PackedNestQuantModel
from nestquant.core.model_checkpoint_file import NQMCFileReader
from nestquant.core.quantizer import quantize_rows
from nestquant.core.rotation import hadamard
from nestquant.core.tensor_checkpoint import PackedNestQuantTensor
from nestquant.models.lazy_linear import LazyNQMCLinear, LazyNQMCWeightCache
from nestquant.models.lazy_model import convert_model_to_lazy_nqmc_
from nestquant.models.rotated_linear import PAPER_HADAMARD_FORMAT


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(8, 4, bias=True)
        self.norm = nn.LayerNorm(4)

    def forward(self, inputs):
        return self.norm(self.proj(inputs))


class TestLazyModelConversion(unittest.TestCase):
    def _checkpoint(self, source: TinyModel, directory: str, *, use_hadamard=True):
        state = source.state_dict()
        stored_weight = (
            hadamard(state["proj.weight"]) if use_hadamard else state["proj.weight"]
        )
        rows = quantize_rows(
            stored_weight, q=14, betas=(0.5, 0.75, 1.0, 1.25)
        )
        record = PackedNestQuantTensor.from_quantized_rows("proj.weight", rows)
        checkpoint_state = dict(state)
        checkpoint_state["proj.weight"] = stored_weight
        metadata = {}
        if use_hadamard:
            metadata["hadamard_rotation"] = {
                "format": PAPER_HADAMARD_FORMAT,
                "converted_linear_names": ["proj"],
                "skipped_tied_linear_names": [],
                "skipped_non_power_of_two_linear_names": [],
                "weight_storage": "source_model_dtype",
            }
        path = Path(directory) / "tiny.nqmc"
        PackedNestQuantModel.from_state_dict(
            checkpoint_state, {"proj.weight": record}, model_metadata=metadata
        ).write_to_file(path)
        return NQMCFileReader(path), record

    def test_converts_quantized_linear_loads_dense_state_and_matches_reference(self):
        torch.manual_seed(11)
        source = TinyModel().eval()
        with tempfile.TemporaryDirectory() as directory:
            reader, record = self._checkpoint(source, directory)
            target = TinyModel().eval()
            with torch.no_grad():
                target.norm.weight.zero_()
                target.norm.bias.fill_(9)
                target.proj.bias.fill_(7)
            cache = LazyNQMCWeightCache(64)
            report = convert_model_to_lazy_nqmc_(target, reader, cache, rows_per_chunk=2)

            self.assertIsInstance(target.proj, LazyNQMCLinear)
            self.assertEqual(report.replaced_linear_names, ("proj",))
            self.assertEqual(report.input_hadamard_names, ("proj",))
            self.assertTrue(torch.equal(target.norm.weight, source.norm.weight))
            self.assertTrue(torch.equal(target.proj.bias, source.proj.bias.to(torch.bfloat16)))
            inputs = torch.randn(3, 8, dtype=torch.bfloat16)
            restored = record.dequantize().to(torch.bfloat16)
            expected = torch.nn.functional.linear(
                hadamard(inputs), restored, source.proj.bias.to(torch.bfloat16)
            ).float()
            expected = source.norm(expected)
            self.assertTrue(torch.equal(target(inputs), expected))

    def test_validation_failure_does_not_replace_modules_or_load_dense_state(self):
        source = TinyModel().eval()
        with tempfile.TemporaryDirectory() as directory:
            reader, _ = self._checkpoint(source, directory)
            target = nn.Sequential(nn.Linear(8, 4)).eval()
            before = target[0].weight.detach().clone()
            with self.assertRaisesRegex(ValueError, "state-dict mismatch"):
                convert_model_to_lazy_nqmc_(
                    target, reader, LazyNQMCWeightCache(64)
                )
            self.assertIsInstance(target[0], nn.Linear)
            self.assertTrue(torch.equal(target[0].weight, before))

    def test_rejects_hadamard_metadata_that_disagrees_with_quantized_modules(self):
        source = TinyModel().eval()
        with tempfile.TemporaryDirectory() as directory:
            reader, _ = self._checkpoint(source, directory)
            reader.model_metadata["hadamard_rotation"]["converted_linear_names"] = []
            target = TinyModel().eval()
            # Empty converted names mean an explicitly unrotated checkpoint and
            # are allowed; use a different name to exercise disagreement.
            reader.model_metadata["hadamard_rotation"]["converted_linear_names"] = ["norm"]
            with self.assertRaisesRegex(ValueError, "disagree"):
                convert_model_to_lazy_nqmc_(
                    target, reader, LazyNQMCWeightCache(64)
                )
            self.assertIsInstance(target.proj, nn.Linear)

    def test_rejects_runtime_options_before_mutating_model(self):
        source = TinyModel().eval()
        with tempfile.TemporaryDirectory() as directory:
            reader, _ = self._checkpoint(source, directory)
            target = TinyModel().eval()
            before = target.norm.weight.detach().clone()
            with self.assertRaisesRegex(ValueError, "rows_per_chunk"):
                convert_model_to_lazy_nqmc_(
                    target, reader, LazyNQMCWeightCache(64), rows_per_chunk=0
                )
            self.assertIsInstance(target.proj, nn.Linear)
            self.assertTrue(torch.equal(target.norm.weight, before))


if __name__ == "__main__":
    unittest.main()

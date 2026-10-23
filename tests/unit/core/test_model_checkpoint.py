"""Tests for model-wide NQMC serialization and tied-weight restoration."""

import unittest
from pathlib import Path
import tempfile

import torch
from torch import nn

from nestquant.core.model_checkpoint import PackedNestQuantModel
from nestquant.core.model_checkpoint_file import NQMCFileReader
from nestquant.core.quantizer import quantize_rows
from nestquant.core.tensor_checkpoint import PackedNestQuantTensor
from nestquant.core.tensor_checkpoint import _tensor_to_bytes
from nestquant.models.quantization import quantize_nestquant_w4_
from nestquant.models.rotated_linear import (
    HadamardRotatedLinear,
    convert_linears_to_hadamard_coordinates_,
    restore_hadamard_structure_from_metadata_,
)


class TinyTiedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.Embedding(12, 8)
        self.lm_head = nn.Linear(8, 12, bias=False)
        self.lm_head.weight = self.embed_tokens.weight
        self.norm = nn.LayerNorm(8)
        self.register_buffer("position_ids", torch.arange(8, dtype=torch.int64))


class TestModelCheckpoint(unittest.TestCase):
    def test_tensor_bytes_cover_only_contiguous_tensor_values(self):
        base = torch.arange(24, dtype=torch.float32).reshape(4, 6)
        view = base[:, ::2]
        expected = view.contiguous().view(torch.uint8).numpy().tobytes(order="C")
        self.assertEqual(_tensor_to_bytes(view), expected)
        self.assertEqual(len(_tensor_to_bytes(view)), view.numel() * view.element_size())

    def _write_checkpoint(self, checkpoint: PackedNestQuantModel, directory: str) -> Path:
        path = Path(directory) / "model.nqmc"
        path.write_bytes(checkpoint.to_bytes())
        return path

    def test_adapter_artifact_names_match_state_dict_and_reload(self):
        model = nn.Sequential(nn.Linear(8, 2, bias=False)).to(torch.bfloat16)
        artifacts = {}
        progress = []
        quantize_nestquant_w4_(
            model, q=14, beta_coefficients=(3.5, 4.5, 6.0, 14.5),
            selected_beta_count=2, rows_per_chunk=1, output_artifacts=artifacts,
            progress_callback=progress.append,
        )
        self.assertEqual(set(artifacts), set(model.state_dict()))
        self.assertEqual(progress[0]["completed_layers"], 1)
        restored = PackedNestQuantModel.from_bytes(
            PackedNestQuantModel.from_state_dict(model.state_dict(), artifacts).to_bytes()
        ).state_dict()
        self.assertTrue(torch.equal(restored["0.weight"].to(model[0].weight.dtype),
                                    model[0].weight))

    def test_round_trip_covers_dense_quantized_and_tied_state(self):
        model = TinyTiedModel().to(torch.bfloat16)
        original_state = model.state_dict()
        quantized_rows = quantize_rows(
            original_state["lm_head.weight"], q=14,
            betas=(0.5, 0.75, 1.0, 1.25),
        )
        quantized = PackedNestQuantTensor.from_quantized_rows("lm_head.weight", quantized_rows)
        artifact = PackedNestQuantModel.from_state_dict(
            original_state, {"lm_head.weight": quantized},
            model_metadata={"model_id": "tiny-test", "architecture": "TinyTiedModel"},
        )
        self.assertEqual(artifact.tensor_count, len(original_state))
        self.assertEqual(artifact.aliases, {"embed_tokens.weight": "lm_head.weight"})
        self.assertIn("norm.weight", artifact.dense_tensors)
        self.assertIn("position_ids", artifact.dense_tensors)
        self.assertIn("lm_head.weight", artifact.quantized_tensors)

        loaded = PackedNestQuantModel.from_bytes(artifact.to_bytes())
        restored = loaded.state_dict()
        self.assertEqual(set(restored), set(original_state))
        self.assertIs(restored["embed_tokens.weight"], restored["lm_head.weight"])
        self.assertTrue(torch.equal(restored["norm.weight"], original_state["norm.weight"]))
        self.assertTrue(torch.equal(restored["position_ids"], original_state["position_ids"]))
        self.assertTrue(torch.equal(restored["lm_head.weight"], quantized.dequantize()))

        model.load_state_dict(restored)
        self.assertTrue(model.lm_head.weight.data_ptr() == model.embed_tokens.weight.data_ptr())
        self.assertTrue(torch.equal(model.lm_head.weight, quantized.dequantize()))

    def test_rejects_payload_corruption(self):
        model = TinyTiedModel()
        artifact = PackedNestQuantModel.from_state_dict(model.state_dict(), {})
        blob = bytearray(artifact.to_bytes())
        blob[-1] ^= 0x01
        with self.assertRaisesRegex(ValueError, "checksum"):
            PackedNestQuantModel.from_bytes(blob)

    def test_file_reader_streams_dense_quantized_and_tied_state(self):
        source = TinyTiedModel().to(torch.bfloat16)
        rows = quantize_rows(
            source.state_dict()["lm_head.weight"], q=14,
            betas=(0.5, 0.75, 1.0, 1.25),
        )
        record = PackedNestQuantTensor.from_quantized_rows("lm_head.weight", rows)
        checkpoint = PackedNestQuantModel.from_state_dict(
            source.state_dict(), {"lm_head.weight": record},
            model_metadata={"model_id": "tiny-test"},
        )
        target = TinyTiedModel().to(torch.bfloat16)
        with tempfile.TemporaryDirectory() as directory:
            reader = NQMCFileReader(self._write_checkpoint(checkpoint, directory), chunk_bytes=7)
            self.assertEqual(reader.tensor_count, len(source.state_dict()))
            self.assertEqual(reader.model_metadata["model_id"], "tiny-test")
            reader.load_into_model(target, device="cpu")

        expected = checkpoint.state_dict()
        self.assertEqual(set(target.state_dict()), set(expected))
        for name, tensor in target.state_dict().items():
            self.assertTrue(torch.equal(tensor, expected[name]), name)
        self.assertEqual(target.embed_tokens.weight.data_ptr(), target.lm_head.weight.data_ptr())

    def test_streaming_writer_matches_in_memory_format(self):
        model = TinyTiedModel().to(torch.bfloat16)
        checkpoint = PackedNestQuantModel.from_state_dict(
            model.state_dict(), {}, model_metadata={"writer": "streaming"}
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nested" / "model.nqmc"
            checkpoint.write_to_file(path)
            self.assertEqual(path.read_bytes(), checkpoint.to_bytes())
            loaded = NQMCFileReader(path)
            self.assertEqual(loaded.model_metadata["writer"], "streaming")

    def test_file_reader_casts_floating_checkpoint_to_model_dtype(self):
        source = TinyTiedModel().to(torch.float32)
        checkpoint = PackedNestQuantModel.from_state_dict(source.state_dict(), {})
        target = TinyTiedModel().to(torch.bfloat16)
        with tempfile.TemporaryDirectory() as directory:
            reader = NQMCFileReader(self._write_checkpoint(checkpoint, directory))
            reader.load_into_model(target, device="cpu")
        self.assertEqual(target.norm.weight.dtype, torch.bfloat16)
        self.assertTrue(torch.equal(target.norm.weight, source.norm.weight.to(torch.bfloat16)))

    def test_file_reader_rejects_corruption_before_loading(self):
        source = TinyTiedModel()
        checkpoint = PackedNestQuantModel.from_state_dict(source.state_dict(), {})
        target = TinyTiedModel()
        before = {name: tensor.clone() for name, tensor in target.state_dict().items()}
        with tempfile.TemporaryDirectory() as directory:
            path = self._write_checkpoint(checkpoint, directory)
            blob = bytearray(path.read_bytes())
            blob[-1] ^= 1
            path.write_bytes(blob)
            with self.assertRaisesRegex(ValueError, "checksum"):
                NQMCFileReader(path).load_into_model(target, device="cpu")
        for name, tensor in target.state_dict().items():
            self.assertTrue(torch.equal(tensor, before[name]), name)

    def test_file_reader_rejects_state_mismatch_before_loading(self):
        source = TinyTiedModel()
        checkpoint = PackedNestQuantModel.from_state_dict(source.state_dict(), {})
        target = nn.Sequential(nn.Linear(8, 2))
        before = {name: tensor.clone() for name, tensor in target.state_dict().items()}
        with tempfile.TemporaryDirectory() as directory:
            reader = NQMCFileReader(self._write_checkpoint(checkpoint, directory))
            with self.assertRaisesRegex(ValueError, "state-dict mismatch"):
                reader.load_into_model(target, device="cpu")
        for name, tensor in target.state_dict().items():
            self.assertTrue(torch.equal(tensor, before[name]), name)

    def test_file_reader_runs_custom_validation_before_loading(self):
        source = TinyTiedModel()
        checkpoint = PackedNestQuantModel.from_state_dict(source.state_dict(), {})
        target = TinyTiedModel()
        before = {name: tensor.clone() for name, tensor in target.state_dict().items()}

        def reject_norm(entry, _record):
            if entry.name == "norm.weight":
                raise ValueError("policy rejected record")

        with tempfile.TemporaryDirectory() as directory:
            reader = NQMCFileReader(self._write_checkpoint(checkpoint, directory))
            with self.assertRaisesRegex(ValueError, "policy rejected"):
                reader.load_into_model(target, device="cpu", record_validator=reject_norm)
        for name, tensor in target.state_dict().items():
            self.assertTrue(torch.equal(tensor, before[name]), name)

    def test_block_ldlq_encoder_metadata_survives_model_round_trip(self):
        model = nn.Sequential(nn.Linear(16, 3, bias=False))
        artifacts = {}

        def hessian_provider(_name):
            hessian = torch.eye(16)
            hessian[:8, 8:] = 0.1 * torch.eye(8)
            hessian[8:, :8] = 0.1 * torch.eye(8)
            return "calibration-input", hessian

        quantize_nestquant_w4_(
            model, q=14, beta_coefficients=(3.0, 4.5, 6.0, 10.0, 15.0),
            selected_beta_count=4, seed=58, rows_per_chunk=2,
            output_artifacts=artifacts, hessian_provider=hessian_provider,
        )
        encoder_metadata = {
            "algorithm": "8-column E8 block-LDLQ",
            "calibration_token_ids_sha256": "a" * 64,
        }
        packed = PackedNestQuantModel.from_state_dict(
            model.state_dict(), artifacts,
            model_metadata={"encoder": "block_ldlq",
                            "encoder_metadata": encoder_metadata},
        )
        restored = PackedNestQuantModel.from_bytes(packed.to_bytes())

        self.assertEqual(restored.model_metadata["encoder"], "block_ldlq")
        self.assertEqual(restored.model_metadata["encoder_metadata"], encoder_metadata)
        self.assertTrue(torch.equal(restored.state_dict()["0.weight"], model[0].weight))

    def test_hadamard_w4_checkpoint_rebuilds_online_rotation_model(self):
        torch.manual_seed(58)
        source = nn.Sequential(nn.Linear(8, 8, bias=True), nn.GELU(), nn.Linear(8, 3))
        conversion = convert_linears_to_hadamard_coordinates_(source)
        artifacts = {}
        quantize_nestquant_w4_(
            source, q=14, beta_coefficients=(3.0, 4.5, 6.0, 10.0, 15.0),
            selected_beta_count=4, rows_per_chunk=2, output_artifacts=artifacts,
        )
        checkpoint = PackedNestQuantModel.from_state_dict(
            source.state_dict(), artifacts,
            model_metadata={"hadamard_rotation": conversion.checkpoint_metadata()},
        )
        target = nn.Sequential(nn.Linear(8, 8, bias=True), nn.GELU(), nn.Linear(8, 3))
        inputs = torch.randn(4, 8)
        with tempfile.TemporaryDirectory() as directory:
            reader = NQMCFileReader(self._write_checkpoint(checkpoint, directory))
            restore_hadamard_structure_from_metadata_(
                target, reader.model_metadata["hadamard_rotation"]
            )
            reader.load_into_model(target, device="cpu")
        self.assertTrue(all(
            isinstance(target[index], HadamardRotatedLinear) for index in (0, 2)
        ))
        torch.testing.assert_close(target(inputs), source(inputs), atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main()

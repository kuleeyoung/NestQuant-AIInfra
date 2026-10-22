"""Tests for model-wide NQMC serialization and tied-weight restoration."""

import unittest

import torch
from torch import nn

from nestquant.core.model_checkpoint import PackedNestQuantModel
from nestquant.core.quantizer import quantize_rows
from nestquant.core.tensor_checkpoint import PackedNestQuantTensor
from nestquant.models.quantization import quantize_nestquant_w4_


class TinyTiedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.Embedding(12, 8)
        self.lm_head = nn.Linear(8, 12, bias=False)
        self.lm_head.weight = self.embed_tokens.weight
        self.norm = nn.LayerNorm(8)
        self.register_buffer("position_ids", torch.arange(8, dtype=torch.int64))


class TestModelCheckpoint(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()

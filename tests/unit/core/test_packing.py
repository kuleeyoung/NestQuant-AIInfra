"""Tests for the versioned fixed-width NestQuant index container."""

import json
import struct

import unittest

import torch

from nestquant.core.packing import PackedNestQuantIndices, pack_nestquant_indices
from nestquant.core.codec import generate_e8_dither
from nestquant.core.quantizer import quantize_rows
from nestquant.core.tensor_checkpoint import PackedNestQuantTensor
from nestquant.core.codec import CURRENT_CODEC_SEMANTICS, LEGACY_CODEC_SEMANTICS


class TestNestQuantPacking(unittest.TestCase):
    def test_numpy_packing_matches_reference_bit_order_for_all_widths(self):
        def reference(values, width):
            result = bytearray((len(values) * width + 7) // 8)
            for index, value in enumerate(values):
                result[index * width // 8] |= int(value) << (index * width % 8)
            return bytes(result)

        for cardinality in (2, 4, 14, 256):
            codes = torch.arange(24, dtype=torch.int64).remainder(cardinality).reshape(1, 3, 8)
            beta = torch.tensor([[0, 1 % cardinality, 2 % cardinality]], dtype=torch.int64)
            packed = pack_nestquant_indices(codes, beta, q=cardinality, beta_count=cardinality)
            width = packed.code_bit_width
            self.assertEqual(packed.code_payload, reference(codes.flatten().tolist(), width))
            self.assertEqual(packed.beta_payload, reference(beta.flatten().tolist(), width))
            restored_codes, restored_beta = packed.unpack()
            self.assertTrue(torch.equal(restored_codes, codes))
            self.assertTrue(torch.equal(restored_beta, beta))

    def test_q14_codes_and_four_way_beta_ids_round_trip(self):
        codes = torch.arange(16, dtype=torch.int64).remainder(14).reshape(1, 2, 8)
        beta_indices = torch.tensor([[0, 3]], dtype=torch.int64)
        packed = pack_nestquant_indices(codes, beta_indices, q=14, beta_count=4)

        self.assertEqual(packed.code_bit_width, 4)
        self.assertEqual(packed.beta_index_bit_width, 2)
        self.assertEqual(len(packed.code_payload), 8)
        self.assertEqual(len(packed.beta_payload), 1)
        self.assertEqual(packed.beta_payload, bytes([0b00001100]))

        decoded_container = PackedNestQuantIndices.from_bytes(packed.to_bytes())
        decoded_codes, decoded_beta = decoded_container.unpack()
        self.assertTrue(torch.equal(decoded_codes, codes))
        self.assertTrue(torch.equal(decoded_beta, beta_indices))
        self.assertEqual(decoded_container.serialized_bytes, len(packed.to_bytes()))

    def test_non_byte_aligned_value_count_has_zero_padding(self):
        codes = torch.tensor([[[1, 2, 3, 4, 5, 6, 7, 8]]])
        beta_indices = torch.tensor([[2]])
        packed = pack_nestquant_indices(codes, beta_indices, q=14, beta_count=4)
        self.assertEqual(len(packed.beta_payload), 1)
        decoded_codes, decoded_beta = packed.unpack()
        self.assertTrue(torch.equal(decoded_codes, codes.to(torch.int64)))
        self.assertTrue(torch.equal(decoded_beta, beta_indices.to(torch.int64)))

    def test_rejects_out_of_range_values_and_shape_mismatch(self):
        with self.assertRaises(ValueError):
            pack_nestquant_indices(torch.full((1, 1, 8), 14), torch.zeros((1, 1), dtype=torch.int64),
                                   q=14, beta_count=4)
        with self.assertRaises(ValueError):
            pack_nestquant_indices(torch.zeros((1, 1, 8), dtype=torch.int64),
                                   torch.zeros((1, 2), dtype=torch.int64), q=14, beta_count=4)

    def test_rejects_invalid_or_truncated_container(self):
        packed = pack_nestquant_indices(torch.zeros((1, 1, 8), dtype=torch.int64),
                                        torch.zeros((1, 1), dtype=torch.int64), q=14, beta_count=4)
        blob = packed.to_bytes()
        with self.assertRaisesRegex(ValueError, "magic"):
            PackedNestQuantIndices.from_bytes(b"BAD!" + blob[4:])
        with self.assertRaisesRegex(ValueError, "truncated"):
            PackedNestQuantIndices.from_bytes(blob[:-1])

    def test_rejects_shape_without_a_row_dimension(self):
        with self.assertRaisesRegex(ValueError, "2 to 255"):
            PackedNestQuantIndices((8,), 14, 4, bytes(4), bytes(1)).to_bytes()

    def test_full_tensor_checkpoint_round_trip_without_dither(self):
        weight = torch.randn((3, 24), generator=torch.Generator().manual_seed(17))
        quantized = quantize_rows(weight, q=14, betas=(0.5, 0.75, 1.0, 1.25))
        packed = PackedNestQuantTensor.from_quantized_rows("layer.proj.weight", quantized)
        restored = PackedNestQuantTensor.from_bytes(packed.to_bytes())

        self.assertEqual(restored.tensor_name, "layer.proj.weight")
        self.assertEqual(restored.codec_semantics, CURRENT_CODEC_SEMANTICS)
        self.assertEqual(restored.serialized_bytes, len(packed.to_bytes()))
        self.assertIsNone(restored.dither)
        self.assertTrue(torch.equal(restored.to_quantized_rows().row_norms, quantized.row_norms))
        self.assertTrue(torch.equal(restored.dequantize(), quantized.dequantize()))

    def test_v2_tensor_checkpoint_defaults_to_legacy_codec_semantics(self):
        weight = torch.randn((2, 8), generator=torch.Generator().manual_seed(117))
        quantized = quantize_rows(weight, q=14, betas=(0.5, 1.0))
        current = PackedNestQuantTensor.from_quantized_rows("legacy.weight", quantized).to_bytes()
        header = struct.Struct("<4sBBIQQQ")
        magic, _version, mode, metadata_len, indices_len, norms_len, dither_len = (
            header.unpack_from(current)
        )
        offset = header.size
        metadata = json.loads(current[offset:offset + metadata_len].decode("utf-8"))
        metadata.pop("codec_semantics")
        legacy_metadata = json.dumps(
            metadata, ensure_ascii=False, separators=(",", ":")
        ).encode("utf-8")
        payload = current[offset + metadata_len:]
        legacy = header.pack(
            magic, 2, mode, len(legacy_metadata), indices_len, norms_len, dither_len
        ) + legacy_metadata + payload
        restored = PackedNestQuantTensor.from_bytes(legacy)
        self.assertEqual(restored.codec_semantics, LEGACY_CODEC_SEMANTICS)

        bf16_weight = weight.to(torch.bfloat16)
        bf16_quantized = quantize_rows(bf16_weight, q=14, betas=(0.5, 0.75, 1.0, 1.25))
        bf16_restored = PackedNestQuantTensor.from_bytes(
            PackedNestQuantTensor.from_quantized_rows("bf16.weight", bf16_quantized).to_bytes()
        )
        self.assertEqual(bf16_restored.output_dtype, torch.bfloat16)
        self.assertTrue(torch.equal(bf16_restored.dequantize(), bf16_quantized.dequantize()))

    def test_full_tensor_checkpoint_preserves_dithered_reconstruction(self):
        weight = torch.randn((2, 16), generator=torch.Generator().manual_seed(19))
        blocks = weight.reshape(2, 2, 8)
        dither = generate_e8_dither(
            4, generator=torch.Generator().manual_seed(23), device="cpu"
        ).reshape(blocks.shape)
        quantized = quantize_rows(weight, q=14, betas=(0.6, 0.9), dither=dither)
        restored = PackedNestQuantTensor.from_bytes(
            PackedNestQuantTensor.from_quantized_rows("dither.weight", quantized).to_bytes()
        )
        self.assertTrue(torch.equal(restored.dither, dither))
        self.assertTrue(torch.equal(restored.dequantize(), quantized.dequantize()))

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_generator_state_artifact_reconstructs_on_cuda(self):
        weight = torch.randn((2, 16), generator=torch.Generator().manual_seed(29), dtype=torch.float32)
        generator = torch.Generator(device="cpu").manual_seed(31)
        state = bytes(generator.get_state().untyped_storage())
        dither = generate_e8_dither(4, generator, device="cuda").reshape(2, 2, 8)
        quantized = quantize_rows(weight.cuda(), q=14, betas=(0.6, 0.9), dither=dither)
        packed = PackedNestQuantTensor.from_quantized_rows(
            "cuda.weight", quantized, dither_generator_state=state,
        )
        restored = PackedNestQuantTensor.from_bytes(packed.to_bytes())
        self.assertIsNone(restored.dither)
        self.assertEqual(restored.dither_generator_device, "cpu")
        self.assertTrue(torch.equal(restored.dequantize(device="cuda"), quantized.dequantize()))


if __name__ == "__main__":
    unittest.main()

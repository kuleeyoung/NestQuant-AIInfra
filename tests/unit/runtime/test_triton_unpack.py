"""CUDA correctness tests for the experimental Triton NQPK unpacker."""

import unittest

import torch

from nestquant.core.codec import generate_e8_dither, generate_e8_dither_stateless
from nestquant.core.packing import pack_nestquant_indices
from nestquant.core.quantizer import quantize_rows
from nestquant.core.tensor_checkpoint import PackedNestQuantTensor
from nestquant.runtime.triton_decode import dequantize_with_triton_unpack
from nestquant.runtime.triton_unpack import triton_available, unpack_indices_triton


@unittest.skipUnless(triton_available(), "Triton CUDA runtime is unavailable")
class TestTritonUnpack(unittest.TestCase):
    def test_matches_reference_for_supported_fixed_widths_and_padding(self):
        generator = torch.Generator().manual_seed(19)
        cases = ((14, 4), (4, 3), (2, 1), (256, 16))
        for q, beta_count in cases:
            with self.subTest(q=q, beta_count=beta_count):
                codes = torch.randint(q, (3, 5, 8), generator=generator)
                betas = torch.randint(beta_count, (3, 5), generator=generator)
                packed = pack_nestquant_indices(
                    codes, betas, q=q, beta_count=beta_count
                )
                expected_codes, expected_betas = packed.unpack(device="cuda")
                actual_codes, actual_betas = unpack_indices_triton(
                    packed, output_dtype=torch.int64
                )
                self.assertTrue(torch.equal(actual_codes, expected_codes))
                self.assertTrue(torch.equal(actual_betas, expected_betas))

    def test_compact_output_dtypes_preserve_values(self):
        codes = torch.arange(80).reshape(2, 5, 8) % 14
        betas = torch.arange(10).reshape(2, 5) % 4
        packed = pack_nestquant_indices(codes, betas, q=14, beta_count=4)
        for dtype in (torch.int16, torch.int32):
            actual_codes, actual_betas = unpack_indices_triton(
                packed, output_dtype=dtype
            )
            self.assertEqual(actual_codes.dtype, dtype)
            self.assertTrue(torch.equal(actual_codes.cpu().long(), codes))
            self.assertTrue(torch.equal(actual_betas.cpu().long(), betas))

    def test_hybrid_decoder_matches_chunked_reference(self):
        generator = torch.Generator().manual_seed(23)
        matrix = torch.randn(7, 16, generator=generator)
        dither_generator = torch.Generator().manual_seed(29)
        dither_state = bytes(dither_generator.get_state().untyped_storage())
        dither = generate_e8_dither(
            14, generator=dither_generator, device="cuda"
        ).reshape(7, 2, 8)
        rows = quantize_rows(
            matrix.cuda(), q=14, betas=(0.25, 0.5, 0.75, 1.0), dither=dither
        )
        record = PackedNestQuantTensor.from_quantized_rows(
            "weight", rows, dither_generator_state=dither_state
        )
        expected = record.dequantize_chunked(
            device="cuda", output_dtype=torch.bfloat16, rows_per_chunk=3
        )
        actual = dequantize_with_triton_unpack(
            record, device="cuda", output_dtype=torch.bfloat16, rows_per_chunk=3
        )
        self.assertTrue(torch.equal(actual, expected))

    def test_hybrid_decoder_supports_v4_random_access_dither(self):
        matrix = torch.randn(
            9, 24, generator=torch.Generator().manual_seed(31)
        ).cuda()
        seed = 20261025
        dither = generate_e8_dither_stateless(
            27, seed, device="cuda"
        ).reshape(9, 3, 8)
        rows = quantize_rows(
            matrix, q=14, betas=(0.25, 0.5, 0.75, 1.0), dither=dither
        )
        record = PackedNestQuantTensor.from_quantized_rows(
            "weight", rows, dither_seed=seed
        )
        expected = record.dequantize_chunked(
            device="cuda", output_dtype=torch.bfloat16, rows_per_chunk=4
        )
        actual = dequantize_with_triton_unpack(
            record, device="cuda", output_dtype=torch.bfloat16, rows_per_chunk=2
        )
        self.assertTrue(torch.equal(actual, expected))


if __name__ == "__main__":
    unittest.main()

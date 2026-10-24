"""Round-trip, residue and overload tests for the nested E8 code."""

import itertools
import unittest

import torch

from nestquant.core.codec import (
    LEGACY_CODEC_SEMANTICS, decode_nested, encode_nested, generate_e8_dither,
)
from nestquant.core.lattice import nearest_e8


class TestNestedE8Codec(unittest.TestCase):
    def test_all_binary_residues_round_trip_to_same_coset(self):
        codes = torch.tensor(list(itertools.product(range(2), repeat=8)), dtype=torch.int64)
        representatives = decode_nested(codes, q=2, dtype=torch.float64)
        recovered, overloaded = encode_nested(representatives, q=2)
        self.assertTrue(torch.equal(recovered, codes))
        self.assertTrue(torch.equal(decode_nested(recovered, q=2, dtype=torch.float64), representatives))

        # Every reconstruction is itself an E8 point and lies in the q-scaled
        # Voronoi region (up to the deterministic tie policy at the boundary).
        closest, _ = nearest_e8(representatives)
        self.assertTrue(torch.equal(closest, representatives))
        scaled_closest, _ = nearest_e8(representatives / 2)
        self.assertTrue(torch.equal((scaled_closest != 0).any(dim=-1), overloaded))

        # NQRT v1/v2 used the older boundary policy and must remain readable.
        legacy_representatives = decode_nested(
            codes, q=2, dtype=torch.float64,
            codec_semantics=LEGACY_CODEC_SEMANTICS,
        )
        legacy_recovered, legacy_overloaded = encode_nested(
            legacy_representatives, q=2,
            codec_semantics=LEGACY_CODEC_SEMANTICS,
        )
        self.assertTrue(torch.equal(legacy_recovered, codes))
        self.assertFalse(legacy_overloaded.any().item())

    def test_lattice_points_separated_by_q_lattice_share_a_code(self):
        # The rows below are E8 points; adding q times another E8 point must
        # leave their coset ID unchanged.
        x = torch.tensor(
            [[1, 1, 0, 0, 0, 0, 0, 0], [2, 0, 0, 0, 0, 0, 0, 0]],
            dtype=torch.float64,
        )
        q = 3
        q_lambda = torch.tensor([2, -2, 0, 0, 0, 0, 0, 0], dtype=x.dtype) * q
        code_x, _ = encode_nested(x, q)
        code_shifted, _ = encode_nested(x + q_lambda, q)
        self.assertTrue(torch.equal(code_x, code_shifted))

    def test_overload_flag_distinguishes_large_lattice_points(self):
        small = torch.zeros(1, 8, dtype=torch.float64)
        large = torch.tensor([[8, 0, 0, 0, 0, 0, 0, 0]], dtype=torch.float64)
        _, small_overload = encode_nested(small, q=2)
        _, large_overload = encode_nested(large, q=2)
        self.assertFalse(small_overload.item())
        self.assertTrue(large_overload.item())

    def test_decoder_validates_residues_and_q(self):
        with self.assertRaises(ValueError):
            decode_nested(torch.tensor([2, 0, 0, 0, 0, 0, 0, 0]), q=2)
        with self.assertRaises(ValueError):
            decode_nested(torch.zeros(8, dtype=torch.int64), q=1)
        with self.assertRaises(TypeError):
            decode_nested(torch.zeros(8, dtype=torch.float32), q=2)

    def test_subtractive_dither_round_trip_with_shared_realization(self):
        generator = torch.Generator().manual_seed(91)
        x = torch.randn((32, 8), generator=generator, dtype=torch.float64)
        dither_generator = torch.Generator().manual_seed(17)
        dither = generate_e8_dither(32, dither_generator, device="cpu", dtype=torch.float64)
        codes, _ = encode_nested(x, q=14, dither=dither)
        reconstructed = decode_nested(codes, q=14, dtype=torch.float64, dither=dither)
        recovered, _ = encode_nested(reconstructed, q=14, dither=dither)
        self.assertTrue(torch.equal(codes, recovered))

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_cpu_rng_dither_is_identical_on_cpu_and_cuda(self):
        generator = torch.Generator(device="cpu").manual_seed(123)
        state = generator.get_state().clone()
        cuda_dither = generate_e8_dither(4096, generator, device="cuda").cpu()
        replay = torch.Generator(device="cpu")
        replay.set_state(state)
        cpu_dither = generate_e8_dither(4096, replay, device="cpu")
        self.assertTrue(torch.equal(cuda_dither, cpu_dither))


if __name__ == "__main__":
    unittest.main()

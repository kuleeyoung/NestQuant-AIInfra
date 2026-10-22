"""Tests for E8 membership and nearest-point correctness."""

from itertools import product
import unittest

import torch

from nestquant.core.lattice import E8_DIMENSION, nearest_e8


def brute_force_distance(x: torch.Tensor) -> float:
    """Independent finite-neighborhood search, used only for tiny test cases."""
    best = float("inf")
    values = [float(value) for value in x]
    for offset in (0.0, 0.5):
        shifted = [value - offset for value in values]
        floors = [int(value // 1) for value in shifted]
        axes = [range(floor - 1, floor + 3) for floor in floors]
        for candidate in product(*axes):
            if sum(candidate) % 2:
                continue
            distance = sum((value - (integer + offset)) ** 2
                           for value, integer in zip(values, candidate))
            best = min(best, distance)
    return best


class TestNearestE8(unittest.TestCase):
    def test_exact_points_from_both_cosets_are_preserved(self):
        x = torch.tensor(
            [[1, 1, 0, 0, 0, 0, 0, 0],
             [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]],
            dtype=torch.float64,
        )
        point, distance = nearest_e8(x)
        self.assertTrue(torch.equal(point, x))
        self.assertTrue(torch.equal(distance, torch.zeros(2, dtype=x.dtype)))

    def test_agrees_with_small_brute_force_search(self):
        generator = torch.Generator().manual_seed(58)
        vectors = torch.randn(4, E8_DIMENSION, generator=generator, dtype=torch.float64)
        vectors *= 1.7
        points, distances = nearest_e8(vectors)
        for vector, point, distance in zip(vectors, points, distances):
            self.assertAlmostEqual(float(distance), brute_force_distance(vector), places=12)
            self.assertAlmostEqual(float((vector - point).square().sum()), float(distance), places=12)

    def test_preserves_batch_shape_dtype_and_device(self):
        x = torch.randn(2, 3, E8_DIMENSION, dtype=torch.float32)
        point, distance = nearest_e8(x)
        self.assertEqual(point.shape, x.shape)
        self.assertEqual(distance.shape, x.shape[:-1])
        self.assertEqual(point.dtype, x.dtype)
        self.assertEqual(point.device, x.device)

    def test_rejects_wrong_dimension_integer_and_nonfinite_values(self):
        with self.assertRaises(ValueError):
            nearest_e8(torch.zeros(7))
        with self.assertRaises(TypeError):
            nearest_e8(torch.zeros(8, dtype=torch.int64))
        invalid = torch.tensor([float("nan")] + [0.0] * 7)
        with self.assertRaises(ValueError):
            nearest_e8(invalid)


if __name__ == "__main__":
    unittest.main()

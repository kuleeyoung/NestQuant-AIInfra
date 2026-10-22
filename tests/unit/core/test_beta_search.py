"""Dynamic-programming beta search compared with exhaustive subsets."""

from itertools import combinations
import unittest

import torch

from nestquant.core.beta_search import build_candidate_tables, select_beta_subset


def exhaustive_first_beta(betas, mse, overloaded, k):
    best = None
    for subset in combinations(range(len(betas)), k):
        subset_overload = overloaded[:, list(subset)]
        fits = ~subset_overload
        if not fits.any(dim=1).all().item():
            continue
        assignment_position = fits.to(torch.int64).argmax(dim=1)
        assignment = torch.tensor(subset)[assignment_position]
        total = sum(float(mse[row, assignment[row]]) for row in range(mse.shape[0]))
        if best is None or total < best[0]:
            best = (total, subset, assignment)
    return best


class TestBetaSearch(unittest.TestCase):
    def test_dynamic_program_matches_exhaustive_subset_search(self):
        generator = torch.Generator().manual_seed(58)
        betas = (0.5, 0.8, 1.2, 2.0, 3.0)
        thresholds = torch.tensor([0, 1, 3, 4, 2, 4])
        overload = torch.arange(len(betas)).unsqueeze(0) < thresholds.unsqueeze(1)
        mse = torch.rand(6, len(betas), generator=generator, dtype=torch.float64)

        expected = exhaustive_first_beta(betas, mse, overload, k=3)
        actual = select_beta_subset(betas, mse, overload, k=3)

        self.assertAlmostEqual(actual.total_mse, expected[0], places=12)
        self.assertEqual(actual.beta_indices, expected[1])
        self.assertTrue(torch.equal(actual.assignments.cpu(), expected[2]))

    def test_builds_real_candidate_tables_and_selects_covering_set(self):
        generator = torch.Generator().manual_seed(17)
        vectors = torch.randn(12, 8, generator=generator, dtype=torch.float64) * 2
        betas = (0.5, 1.0, 2.0, 4.0)
        mse, overload = build_candidate_tables(vectors, q=4, betas=betas)
        self.assertEqual(mse.shape, (12, 4))
        self.assertEqual(overload.shape, (12, 4))
        result = select_beta_subset(betas, mse, overload, k=2)
        self.assertEqual(len(result.beta_indices), 2)
        self.assertTrue(torch.isfinite(torch.tensor(result.total_mse)).item())
        self.assertFalse(overload[torch.arange(12), result.assignments].any().item())
        measured_total = mse[torch.arange(12), result.assignments].sum().item()
        self.assertAlmostEqual(result.total_mse, measured_total, places=10)

    def test_rejects_nonmonotone_overload_and_uncovered_data(self):
        betas = (0.5, 1.0, 2.0)
        mse = torch.ones(1, 3)
        nonmonotone = torch.tensor([[False, True, False]])
        with self.assertRaises(ValueError):
            select_beta_subset(betas, mse, nonmonotone, k=1)

        never_fits = torch.ones(2, 3, dtype=torch.bool)
        with self.assertRaises(ValueError):
            select_beta_subset(betas, torch.ones(2, 3), never_fits, k=2)


if __name__ == "__main__":
    unittest.main()

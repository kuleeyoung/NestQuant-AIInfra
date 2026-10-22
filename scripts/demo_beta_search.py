"""Toy demonstration of Appendix F beta subset search."""

import argparse
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nestquant.core.beta_search import build_candidate_tables, select_beta_subset
from nestquant.core.quantizer import quantize_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    generator = torch.Generator(device=device).manual_seed(58)
    rows = torch.randn(32, 64, generator=generator, device=device, dtype=torch.float32)
    row_norms = torch.linalg.vector_norm(rows, dim=-1, keepdim=True)
    normalized = rows * (64**0.5 / row_norms)
    vectors = normalized.reshape(-1, 8)
    betas = tuple(0.25 * (index + 1) for index in range(16))

    mse, overloaded = build_candidate_tables(vectors, q=14, betas=betas)
    result = select_beta_subset(betas, mse, overloaded, k=4)
    quantized = quantize_rows(rows, q=14, betas=result.beta_values)
    reconstruction = quantized.dequantize()
    rmse = (rows - reconstruction).square().mean().sqrt()
    print(f"device: {device}")
    print(f"calibration toy vectors: {vectors.shape[0]}")
    print(f"selected beta subset (toy grid): {result.beta_values}")
    print(f"First-beta assigned counts: {torch.bincount(result.assignments, minlength=len(betas)).tolist()}")
    print(f"sum of selected First-beta MSE: {result.total_mse:.6f}")
    print(f"Opt-beta per-block counts on selected set: {torch.bincount(quantized.beta_indices.flatten(), minlength=4).tolist()}")
    print(f"row quantizer toy RMSE: {rmse.item():.6f}")
    print("Toy demonstration only; this is not a model calibration result.")


if __name__ == "__main__":
    main()

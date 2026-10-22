"""Toy numerical sanity check for the fixed-candidate row quantizer."""

import argparse
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nestquant.core.quantizer import quantize_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    generator = torch.Generator(device=device).manual_seed(58)
    matrix = torch.randn(4, 32, generator=generator, device=device, dtype=torch.float32)
    quantized = quantize_rows(matrix, q=14, betas=(0.5, 1.0, 2.0))
    reconstructed = quantized.dequantize()
    rmse = (matrix - reconstructed).square().mean().sqrt()

    print(f"device: {device}")
    print(f"input shape: {tuple(matrix.shape)}")
    print(f"integer code tensor shape: {tuple(quantized.codes.shape)}")
    print(f"selected beta indices:\n{quantized.beta_indices}")
    print(f"toy reconstruction RMSE: {rmse.item():.6f}")
    print("This is a numerical toy check, not an LLM quality or compression result.")


if __name__ == "__main__":
    main()

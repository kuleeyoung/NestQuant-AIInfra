"""Run a tiny CPU/GPU demonstration of the E8 nearest-point oracle."""

import argparse
from pathlib import Path
import sys

import torch

# Allow running directly from a source checkout without an editable install.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nestquant.core.lattice import nearest_e8


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu", help="cpu, cuda, or an explicit torch device")
    args = parser.parse_args()

    device = torch.device(args.device)
    x = torch.tensor(
        [[0.1, 0.9, 1.2, -0.4, 0.7, -1.1, 0.2, 0.6]],
        dtype=torch.float64,
        device=device,
    )
    point, distance = nearest_e8(x)
    print(f"device: {device}")
    print(f"input: {x}")
    print(f"nearest E8 point: {point}")
    print(f"squared distance: {distance.item():.12f}")


if __name__ == "__main__":
    main()

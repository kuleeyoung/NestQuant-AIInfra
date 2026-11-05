"""Compare scalar-oracle and tiled packed-W4A4 logits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scalar", type=Path)
    parser.add_argument("tiled", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    scalar = torch.load(args.scalar, map_location="cpu", weights_only=True).float()
    tiled = torch.load(args.tiled, map_location="cpu", weights_only=True).float()
    if scalar.shape != tiled.shape:
        raise ValueError("logit tensor shapes differ")
    difference = tiled - scalar
    scalar_top = scalar.argmax(dim=-1)
    tiled_top = tiled.argmax(dim=-1)
    payload = {
        "protocol": "packed-w4a4-scalar-vs-tiled-logits-v1",
        "shape": list(scalar.shape),
        "max_abs_error": difference.abs().max().item(),
        "mean_abs_error": difference.abs().mean().item(),
        "relative_l2_error": (
            torch.linalg.vector_norm(difference) /
            torch.linalg.vector_norm(scalar)
        ).item(),
        "cosine_similarity": torch.nn.functional.cosine_similarity(
            scalar, tiled, dim=-1
        ).tolist(),
        "scalar_argmax": scalar_top.tolist(),
        "tiled_argmax": tiled_top.tolist(),
        "argmax_equal": bool(torch.equal(scalar_top, tiled_top)),
    }
    rendered = json.dumps(payload, indent=2)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

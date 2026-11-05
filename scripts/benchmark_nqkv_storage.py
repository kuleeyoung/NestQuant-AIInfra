"""Measure NQKV v1 page payload and round-trip quality on synthetic KV rows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from nestquant.core.kv_format import PackedNQKVPage
from nestquant.core.runtime_quantization import fake_quantize_nestquant_rows


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--page-size", type=int, default=16)
    parser.add_argument("--valid-tokens", type=int, default=16)
    parser.add_argument("--kv-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--q", type=int, default=14)
    parser.add_argument("--betas", type=float, nargs="+", default=(0.5, 0.75, 1.0, 1.25))
    parser.add_argument("--seed", type=int, default=20261027)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    generator = torch.Generator().manual_seed(args.seed)
    shape = (args.valid_tokens, args.kv_heads, args.head_dim)
    key = torch.randn(shape, generator=generator, dtype=torch.float32).to(torch.bfloat16)
    value = torch.randn(shape, generator=generator, dtype=torch.float32).to(torch.bfloat16)
    page = PackedNQKVPage.from_tensors(
        key, value, page_size=args.page_size, q=args.q, betas=args.betas
    )
    restored = PackedNQKVPage.from_bytes(page.to_bytes())
    restored_key, restored_value = restored.dequantize()
    oracle_key, _ = fake_quantize_nestquant_rows(key, q=args.q, betas=args.betas)
    oracle_value, _ = fake_quantize_nestquant_rows(value, q=args.q, betas=args.betas)

    key_error = (restored_key.float() - key.float()).square()
    value_error = (restored_value.float() - value.float()).square()
    result = {
        "status": "completed",
        "protocol": "nqkv-storage-v1",
        "seed": args.seed,
        "shape": list(shape),
        "q": args.q,
        "betas": list(args.betas),
        **page.storage_report(),
        "oracle_exact": bool(
            torch.equal(restored_key, oracle_key)
            and torch.equal(restored_value, oracle_value)
        ),
        "key_relative_squared_error": float(
            key_error.sum() / key.float().square().sum()
        ),
        "value_relative_squared_error": float(
            value_error.sum() / value.float().square().sum()
        ),
        "key_max_abs_error": float((restored_key.float() - key.float()).abs().max()),
        "value_max_abs_error": float((restored_value.float() - value.float()).abs().max()),
    }
    encoded = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")


if __name__ == "__main__":
    main()

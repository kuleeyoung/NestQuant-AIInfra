"""Compare the Python E8 reference with the CUDA C++ implementation."""

import argparse
import json
import time

import torch

from nestquant.core.lattice import nearest_e8
from nestquant.runtime import e8_nearest_cuda


def timed(fn, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1e6 / iters


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", nargs="+", type=int, default=[1024, 16384, 262144])
    parser.add_argument("--iters", type=int, default=100)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    rows = []
    for n in args.sizes:
        x = torch.randn(n, 8, device="cuda", dtype=torch.float32)
        ref, _ = nearest_e8(x)
        cuda = e8_nearest_cuda(x)
        assert torch.equal(ref, cuda), f"mismatch at n={n}"
        ref_us = timed(lambda: nearest_e8(x), warmup=10, iters=args.iters)
        cuda_us = timed(lambda: e8_nearest_cuda(x), warmup=10, iters=args.iters)
        rows.append({
            "vectors": n,
            "python_us": ref_us,
            "cuda_us": cuda_us,
            "speedup": ref_us / cuda_us,
            "cuda_vectors_per_s": n / (cuda_us * 1e-6),
        })
    print(json.dumps({"device": torch.cuda.get_device_name(), "rows": rows}, indent=2))


if __name__ == "__main__":
    main()

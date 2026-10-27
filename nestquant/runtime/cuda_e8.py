"""Lazy loader for the reference CUDA E8 projection kernel."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

_EXTENSION: Any | None = None


def cuda_e8_available() -> bool:
    """Return whether CUDA is available for compiling/running the extension."""
    return bool(torch.cuda.is_available())


def _load_extension() -> Any:
    global _EXTENSION
    if _EXTENSION is not None:
        return _EXTENSION
    if not cuda_e8_available():
        raise RuntimeError("CUDA E8 extension requires a CUDA-enabled PyTorch")
    from torch.utils.cpp_extension import load

    root = Path(__file__).resolve().parents[2]
    _EXTENSION = load(
        name="nestquant_cuda_e8",
        sources=[str(root / "nestquant/runtime/cuda_e8.cpp"), str(root / "nestquant/runtime/cuda_e8.cu")],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3"],
        verbose=False,
    )
    return _EXTENSION


def e8_nearest_cuda(x: torch.Tensor) -> torch.Tensor:
    """Project a ``[N, 8]`` float32 CUDA tensor onto E8."""
    if not x.is_cuda:
        raise ValueError("x must be a CUDA tensor")
    if x.dtype != torch.float32 or x.ndim != 2 or x.shape[1] != 8:
        raise ValueError("x must have dtype float32 and shape [N, 8]")
    return _load_extension().e8_nearest(x)

"""Small synthetic matrix-product quality benchmark (not a paper-scale run)."""

from __future__ import annotations

import math
import hashlib
import json
import platform
import subprocess
import sys
from typing import Any

import torch

from ..core.beta_search import build_candidate_tables, select_beta_subset
from ..core.codec import generate_e8_dither
from ..core.quantizer import quantize_rows


def _symmetric_rtn_rows(matrix: torch.Tensor, bits: int = 4) -> torch.Tensor:
    """Per-row symmetric signed uniform RTN, used as an engineering baseline."""
    qmax = 2 ** (bits - 1) - 1
    scale = matrix.abs().amax(dim=-1, keepdim=True).clamp_min(torch.finfo(matrix.dtype).tiny) / qmax
    return torch.round(matrix / scale).clamp(-qmax, qmax) * scale


def _uniform_linf_blocks(matrix: torch.Tensor, bits: int = 4, block_size: int = 8) -> torch.Tensor:
    """Uniform symmetric quantization with an L-infinity scale per 8D block."""
    qmax = 2 ** (bits - 1) - 1
    blocks = matrix.reshape(matrix.shape[0], -1, block_size)
    scale = blocks.abs().amax(dim=-1, keepdim=True).clamp_min(torch.finfo(matrix.dtype).tiny) / qmax
    reconstructed = torch.round(blocks / scale).clamp(-qmax, qmax) * scale
    return reconstructed.reshape_as(matrix)


def run_synthetic_matrix_product(
    *, n: int = 64, m: int = 64, k: int = 64, q: int = 14,
    beta_count: int = 4, seed: int = 58, device: str = "cpu",
    candidate_step: float = 0.5,
) -> dict[str, Any]:
    """Compare FP32, per-row RTN-W4, and NestQuant-style W quantization.

    The NestQuant beta subset is fitted jointly on normalized 8D blocks from
    both matrices using Appendix F DP. Final blocks use the current Opt-beta
    quantizer. This is a project-stage experiment: it omits rotations,
    LDLQ/QA-LDLQ, activation/KV quantization, and packed storage.
    """
    if min(n, m, k, q, beta_count) < 1:
        raise ValueError("dimensions, q and beta_count must be positive")
    if k % 8:
        raise ValueError("k must be divisible by 8 for E8 blocks")
    if q < 2:
        raise ValueError("q must be at least 2")
    if not math.isfinite(candidate_step) or candidate_step <= 0:
        raise ValueError("candidate_step must be positive and finite")

    target = torch.device(device)
    generator = torch.Generator(device=target).manual_seed(seed)
    a = torch.randn((n, k), generator=generator, device=target, dtype=torch.float32)
    b = torch.randn((m, k), generator=generator, device=target, dtype=torch.float32)
    # Paper §5.1 searches scaling coefficients in (1/2) * {1,...,50}.
    candidates = tuple(candidate_step * index for index in range(1, 51))
    if beta_count > len(candidates):
        raise ValueError("beta_count cannot exceed 50 candidate values")
    # The paper's coefficients are passed as beta_0 in the reference code,
    # which uses beta = beta_0 / q inside encode/decode (coder.py). Keep both
    # units explicit so the grid cannot be silently confused with codec scale.
    codec_scales = tuple(beta / q for beta in candidates)

    def normalized_blocks(matrix: torch.Tensor) -> torch.Tensor:
        norms = torch.linalg.vector_norm(matrix, dim=-1, keepdim=True).clamp_min(torch.finfo(matrix.dtype).tiny)
        return (matrix * (math.sqrt(k) / norms)).reshape(-1, 8)

    vectors = torch.cat((normalized_blocks(a), normalized_blocks(b)), dim=0)
    dither_generator = torch.Generator(device=target).manual_seed(seed + 1)
    flat_dither = generate_e8_dither(vectors.shape[0], dither_generator,
                                     device=target, dtype=vectors.dtype)
    mse, overloaded = build_candidate_tables(vectors, q, codec_scales, dither=flat_dither)
    selected = select_beta_subset(codec_scales, mse, overloaded, beta_count)
    a_block_count = a.shape[0] * (k // 8)
    a_dither = flat_dither[:a_block_count].reshape(a.shape[0], k // 8, 8)
    b_dither = flat_dither[a_block_count:].reshape(b.shape[0], k // 8, 8)
    aq_result = quantize_rows(a, q, selected.beta_values, dither=a_dither)
    bq_result = quantize_rows(b, q, selected.beta_values, dither=b_dither)
    aq, bq = aq_result.dequantize(), bq_result.dequantize()
    artn, brtn = _symmetric_rtn_rows(a), _symmetric_rtn_rows(b)
    auniform, buniform = _uniform_linf_blocks(a), _uniform_linf_blocks(b)

    reference = a @ b.T
    products = {
        "rtn_w4_per_row": artn @ brtn.T,
        "uniform_linf_w4_block8": auniform @ buniform.T,
        "nestquant_w_quality_simulation": aq @ bq.T,
    }
    beta_indices = torch.cat((aq_result.beta_indices.reshape(-1), bq_result.beta_indices.reshape(-1)))
    frequencies = torch.bincount(beta_indices, minlength=beta_count).to(torch.float64)
    probabilities_tensor = frequencies / frequencies.sum()
    nonzero_probabilities = probabilities_tensor[probabilities_tensor > 0]
    sum_p_log2_p = float((nonzero_probabilities * torch.log2(nonzero_probabilities)).sum().item())
    shannon_entropy = -sum_p_log2_p
    paper_rate = math.log2(q) + sum_p_log2_p / 8.0
    entropy_rate = math.log2(q) + shannon_entropy / 8.0
    config = {"n": n, "m": m, "k": k, "q": q, "beta_count": beta_count,
              "seed": seed, "device": str(target), "dtype": "float32",
              "beta_candidate_coefficients_paper_scale": list(candidates),
              "beta_candidate_codec_scales": list(codec_scales),
              "beta_subset_paper_scale": [candidates[index] for index in selected.beta_indices],
              "beta_subset_codec_scales": list(selected.beta_values),
              "subtractive_e8_dither": True, "dither_seed": seed + 1}
    config_hash = hashlib.sha256(json.dumps(config, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    try:
        git_revision = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                      check=True, text=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        git_revision = None
    environment = {"python": sys.version.split()[0], "platform": platform.platform(),
                   "torch": torch.__version__, "torch_cuda": torch.version.cuda,
                   "device_name": (torch.cuda.get_device_name(target) if target.type == "cuda" else "CPU")}
    result: dict[str, Any] = {
        "schema_version": 1,
        "experiment": "synthetic_matrix_product_toy",
        "paper_reference": "NestQuant §5.1, p.9; Appendix F, pp.19-21",
        "config": config,
        "config_sha256": config_hash,
        "git_revision": git_revision,
        "environment": environment,
        "metrics": {},
        "rate_estimate": {
            "beta_use_probabilities": probabilities_tensor.tolist(),
            "paper_formula_bits_per_entry": paper_rate,
            "shannon_side_information_bits_per_entry": entropy_rate,
            "formula_note": "The paper p.9 prints log2(q) + (1/8) sum p log2(p). The conventional entropy side-information accounting is log2(q) - (1/8) sum p log2(p); both are reported because the printed sign yields a lower rate when beta entropy increases.",
        },
        "limitations": [
            "toy dimensions; paper §5.1 uses n=k=m=4096",
            "RTN is a project reference baseline, not the paper's SpinQuant L-infinity baseline",
            "NestQuant path is fake-quant quality simulation with int64 codes, not packed low-bit storage",
            "no rotations, LDLQ/QA-LDLQ, activation/KV quantization, or inference kernels",
            "the same fixed dither realization is reused across candidate beta scales; upstream try-all may draw separate dithers per scale",
            "the beta subset is fitted on these synthetic matrices; no held-out generalization is measured",
            "the two methods do not have a matched effective bit budget",
            "beta assignment frequencies use the current Opt-beta quality simulator; paper rate/distortion analysis and final packed representation are not independently reproduced",
        ],
    }
    for name, product in products.items():
        difference = reference - product
        result["metrics"][name] = {
            "mean_squared_error_per_entry": float(difference.square().mean().item()),
            "root_mean_squared_error_per_entry": float(difference.square().mean().sqrt().item()),
            "relative_frobenius_error": float((torch.linalg.vector_norm(difference) /
                                                torch.linalg.vector_norm(reference).clamp_min(1e-30)).item()),
        }
    result["beta_search"] = {"first_beta_subset_total_squared_error": selected.total_mse,
                             "normalized_8d_vector_count": int(vectors.shape[0])}
    return result

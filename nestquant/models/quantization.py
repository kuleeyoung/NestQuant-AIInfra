"""Floating-point quality-simulation adapters for causal LM linear layers."""

from __future__ import annotations

import math
import time
from typing import Callable, Sequence

import torch
from torch import nn

from ..core.beta_search import build_candidate_tables, select_beta_subset
from ..core.codec import generate_e8_dither
from ..core.quantizer import QuantizedRows, quantize_rows, quantize_rows_ldlq
from ..core.ldlq import block_unit_lower_ldl
from ..core.tensor_checkpoint import PackedNestQuantTensor
from ..core.rotation import rotate_covariance, signed_hadamard
from .rotated_linear import HadamardRotatedLinear


def quantize_rtn_w4_(model: nn.Module, *, group_size: int = 128) -> dict:
    """In-place per-row/group symmetric RTN W4 fake quantization of Linear weights."""
    input_embedding_weight = _input_embedding_weight(model)
    stats = {"method": "rtn_w4_fake_quant", "group_size": group_size,
             "layers": 0, "parameters": 0, "scale_count": 0, "row_count": 0,
             "input_embedding_weight_shared_with_linear": False}
    replacements = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        weight = module.weight.data
        if group_size < 1:
            raise ValueError("group_size must be positive")
        if weight.shape[1] % group_size:
            raise ValueError(f"{name}: in_features must be divisible by group_size={group_size}")
        values = weight.float().reshape(weight.shape[0], -1, group_size)
        qmax = 7
        scales = values.abs().amax(dim=-1, keepdim=True).clamp_min(torch.finfo(values.dtype).tiny) / qmax
        restored = (torch.round(values / scales).clamp(-qmax, qmax) * scales).reshape_as(weight)
        replacements.append((weight, restored.to(weight.dtype)))
        if input_embedding_weight is not None and weight.data_ptr() == input_embedding_weight.data_ptr():
            stats["input_embedding_weight_shared_with_linear"] = True
        stats["layers"] += 1
        stats["parameters"] += weight.numel()
        stats["scale_count"] += scales.numel()
        stats["row_count"] += weight.shape[0]
    if stats["layers"] == 0:
        raise ValueError("model has no torch.nn.Linear modules to quantize")
    # Commit only after all layers have passed validation and quantization.
    with torch.no_grad():
        for weight, restored in replacements:
            weight.copy_(restored)
    stats["storage_estimate"] = _estimate_rtn_w4_storage(stats)
    stats["storage_estimate"]["input_embedding_weight_is_included"] = stats["input_embedding_weight_shared_with_linear"]
    return stats


def quantize_nestquant_w4_(
    model: nn.Module,
    *,
    q: int,
    beta_coefficients: Sequence[float],
    selected_beta_count: int,
    seed: int = 58,
    rows_per_chunk: int = 512,
    output_artifacts: dict[str, PackedNestQuantTensor] | None = None,
    progress_callback: Callable[[dict], None] | None = None,
    rotation_seed: int | None = None,
    hessian_provider: Callable[[str], tuple[str, torch.Tensor]] | None = None,
    activation_variance_provider: Callable[[str], tuple[str, float] | None] | None = None,
    qa_epsilon_constant: float = 155.0,
    ldlq_relative_damp: float = 0.01,
    excluded_linear_names: Sequence[str] = (),
) -> dict:
    """In-place NestQuant-style W4 quality simulation for every Linear weight.

    For each matrix this selects a beta subset by Appendix F First-beta DP on
    normalized 8D weight blocks, then applies Algorithm 3's Opt-beta block
    assignment. Optional signed-Hadamard rotation is a weight-space ablation:
    reconstructed weights are inverse-rotated before execution. When LDLQ is
    enabled, its calibration Hessian is transformed to the same coordinates.
    No whole-model rotation fusion, QA-LDLQ, or low-bit kernel is claimed.
    """
    if q < 2:
        raise ValueError("q must be at least 2")
    coefficients = tuple(float(value) for value in beta_coefficients)
    if not coefficients or any(not math.isfinite(value) or value <= 0 for value in coefficients):
        raise ValueError("beta_coefficients must be positive finite values")
    if any(left >= right for left, right in zip(coefficients, coefficients[1:])):
        raise ValueError("beta_coefficients must be strictly increasing")
    if selected_beta_count < 1 or selected_beta_count > len(coefficients):
        raise ValueError("selected_beta_count must fit the candidate set")
    if rows_per_chunk < 1:
        raise ValueError("rows_per_chunk must be positive")
    excluded_names = frozenset(excluded_linear_names)
    if any(not isinstance(name, str) or not name for name in excluded_names):
        raise ValueError("excluded_linear_names must contain non-empty strings")
    if rotation_seed is not None:
        if output_artifacts is not None:
            raise ValueError("rotated weights require rotation metadata; NQRT export is not supported yet")
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) and module.in_features & (module.in_features - 1):
                raise ValueError(f"{name}: signed Hadamard requires power-of-two in_features")
    if hessian_provider is not None:
        if ldlq_relative_damp < 0:
            raise ValueError("ldlq_relative_damp must be nonnegative")
    if activation_variance_provider is not None:
        if hessian_provider is None:
            raise ValueError("QA-LDLQ activation variance requires a Hessian provider")
        if not math.isfinite(qa_epsilon_constant) or qa_epsilon_constant <= 0:
            raise ValueError("qa_epsilon_constant must be positive and finite")

    stats = {"method": "nestquant_w4_quality_simulation", "q": q,
             "beta_coefficients_project_scale": list(coefficients),
             "candidate_codec_scales": [value / q for value in coefficients],
             "selected_beta_count": selected_beta_count, "seed": seed,
             "dither_rng_device": "cpu",
             "dither_state_captured": output_artifacts is not None,
             "weight_rotation": "none" if rotation_seed is None else "signed_hadamard_inverse_weight",
             "rotation_seed": rotation_seed,
             "ldlq": hessian_provider is not None,
             "ldlq_hessian_coordinates": (
                 None if hessian_provider is None else
                 "original_input" if rotation_seed is None else
                 "per_linear_signed_hadamard_input"
             ),
             "ldlq_relative_damp": ldlq_relative_damp if hessian_provider is not None else None,
             "qa_ldlq": activation_variance_provider is not None,
             "qa_epsilon_constant": qa_epsilon_constant if activation_variance_provider is not None else None,
             "rows_per_chunk": rows_per_chunk, "layers": 0, "parameters": 0,
             "row_count": 0, "e8_block_count": 0, "layer_beta_subsets": {},
             "layer_timings": {},
             "excluded_linear_names": sorted(excluded_names),
             "input_embedding_weight_shared_with_linear": False}
    input_embedding_weight = _input_embedding_weight(model)
    replacements = []
    staged_artifacts: dict[str, PackedNestQuantTensor] = {}
    generator = torch.Generator(device="cpu").manual_seed(seed)
    rotation_generator = torch.Generator(device="cpu").manual_seed(rotation_seed or 0)
    cached_hessian_key = None
    cached_lower = None
    cached_damping = None
    cached_qa_key = None
    cached_qa_cholesky = None
    cached_qa_hessian = None
    quantizable_types = (nn.Linear, HadamardRotatedLinear)
    available_linear_names = {
        name for name, module in model.named_modules() if isinstance(module, quantizable_types)
    }
    unknown_exclusions = excluded_names - available_linear_names
    if unknown_exclusions:
        raise ValueError(f"unknown excluded Linear modules: {sorted(unknown_exclusions)}")
    total_layers = len(available_linear_names - excluded_names)
    def elapsed_since(started: float, device: torch.device) -> float:
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        return time.perf_counter() - started

    for name, module in model.named_modules():
        if not isinstance(module, quantizable_types):
            continue
        if name in excluded_names:
            continue
        original = module.weight.data
        layer_started = time.perf_counter()
        phase_started = layer_started
        if input_embedding_weight is not None and original.data_ptr() == input_embedding_weight.data_ptr():
            stats["input_embedding_weight_shared_with_linear"] = True
        if original.shape[1] % 8:
            raise ValueError(f"{name}: in_features must be divisible by E8 block size 8")
        matrix = original.detach().float()
        rows, features = matrix.shape
        signs = None
        if rotation_seed is not None:
            signs = (torch.randint(0, 2, (features,), generator=rotation_generator) * 2 - 1).to(
                device=matrix.device, dtype=matrix.dtype)
            # Clone also protects FP32 input models from pre-commit mutation.
            rotated = torch.empty_like(matrix)
            for start in range(0, rows, rows_per_chunk):
                rotated[start:start + rows_per_chunk] = signed_hadamard(
                    matrix[start:start + rows_per_chunk], signs)
            matrix = rotated

        hessian_key = None
        if hessian_provider is not None:
            phase_started = time.perf_counter()
            hessian_key, hessian_cpu = hessian_provider(name)
            hessian = hessian_cpu.to(device=original.device, dtype=matrix.dtype)
            if isinstance(module, HadamardRotatedLinear):
                hessian = rotate_covariance(hessian, torch.ones(
                    hessian.shape[0], device=hessian.device, dtype=hessian.dtype
                ))
            elif signs is not None:
                hessian = rotate_covariance(hessian, signs)
            qa_record = (
                None if activation_variance_provider is None
                else activation_variance_provider(name)
            )
            epsilon = None
            if qa_record is not None:
                variance_key, activation_variance = qa_record
                epsilon_squared = activation_variance / qa_epsilon_constant
                epsilon = math.sqrt(epsilon_squared)
                qa_key = (
                    hessian_key, variance_key, float(epsilon_squared),
                    None if signs is None else name,
                )
                if qa_key != cached_qa_key:
                    symmetric = (hessian + hessian.T) * 0.5
                    cached_qa_hessian = symmetric + torch.eye(
                        symmetric.shape[0], device=symmetric.device, dtype=symmetric.dtype
                    ) * epsilon_squared
                    cached_qa_cholesky = torch.linalg.cholesky(cached_qa_hessian)
                    cached_qa_key = qa_key
                if epsilon_squared:
                    correction = torch.cholesky_solve(
                        matrix.T, cached_qa_cholesky
                    ).T
                    matrix = matrix - epsilon_squared * correction
                hessian = cached_qa_hessian
                stats.setdefault("layer_activation_variance_sources", {})[name] = variance_key
                stats.setdefault("layer_activation_variances", {})[name] = activation_variance
                stats.setdefault("layer_qa_epsilons", {})[name] = epsilon
            factor_key = (
                hessian_key,
                None if signs is None else name,
                None if epsilon is None else epsilon,
            )
            if factor_key != cached_hessian_key:
                cached_lower, _diagonal, cached_damping = block_unit_lower_ldl(
                    hessian, block_size=8, relative_damp=ldlq_relative_damp
                )
                cached_hessian_key = factor_key
                del _diagonal
            timing_hessian = elapsed_since(phase_started, original.device)
            stats.setdefault("layer_hessian_sources", {})[name] = hessian_key
            stats.setdefault("layer_hessian_coordinates", {})[name] = (
                "local_hadamard_input" if isinstance(module, HadamardRotatedLinear) else
                "original_input" if signs is None else "signed_hadamard_input"
            )
            stats.setdefault("layer_ldlq_damping", {})[name] = cached_damping
            del hessian
            phase_started = time.perf_counter()
        row_norms = torch.linalg.vector_norm(matrix, dim=-1)
        nonzero = row_norms > 0
        safe_norms = torch.where(nonzero, row_norms, torch.ones_like(row_norms))
        normalized = matrix * (math.sqrt(features) / safe_norms).unsqueeze(-1)
        normalized = torch.where(nonzero.unsqueeze(-1), normalized, torch.zeros_like(normalized))
        normalized_blocks = normalized.reshape(rows, features // 8, 8)
        timing = {"normalization_seconds": elapsed_since(phase_started, original.device)}
        if hessian_provider is not None:
            timing["qa_adjust_and_ldl_factor_seconds"] = timing_hessian

        phase_started = time.perf_counter()
        dither_state = bytes(generator.get_state().untyped_storage())
        dither_cpu = generate_e8_dither(rows * (features // 8), generator,
                                        device="cpu", dtype=torch.float32)
        dither_blocks_cpu = dither_cpu.reshape(rows, features // 8, 8)
        dither_blocks = dither_blocks_cpu.to(original.device)
        candidate_scales = tuple(value / q for value in coefficients)
        timing["dither_seconds"] = elapsed_since(phase_started, original.device)

        phase_started = time.perf_counter()
        mse_chunks, overload_chunks = [], []
        for start in range(0, rows, rows_per_chunk):
            end = min(start + rows_per_chunk, rows)
            vectors = normalized_blocks[start:end].reshape(-1, 8)
            block_dither = dither_blocks[start:end].reshape(-1, 8)
            mse_part, overload_part = build_candidate_tables(
                vectors, q, candidate_scales, dither=block_dither)
            mse_chunks.append(mse_part)
            overload_chunks.append(overload_part)
        mse = torch.cat(mse_chunks, dim=0)
        overloaded = torch.cat(overload_chunks, dim=0)
        timing["candidate_seconds"] = elapsed_since(phase_started, original.device)
        phase_started = time.perf_counter()
        try:
            selected = select_beta_subset(candidate_scales, mse, overloaded, selected_beta_count)
        except ValueError as error:
            overload_counts = overloaded.sum(dim=0).detach().cpu().tolist()
            raise ValueError(
                f"{name}: beta candidates cannot cover all normalized E8 blocks; "
                f"overloaded block counts by candidate={overload_counts}"
            ) from error
        timing["beta_search_seconds"] = elapsed_since(phase_started, original.device)

        phase_started = time.perf_counter()
        reconstructed = torch.empty_like(matrix)
        quantized_parts = []
        quantization_rows_per_chunk = (
            rows_per_chunk if hessian_provider is None else max(rows_per_chunk, 8192)
        )
        for start in range(0, rows, quantization_rows_per_chunk):
            end = min(start + quantization_rows_per_chunk, rows)
            if hessian_provider is None:
                quantized = quantize_rows(
                    matrix[start:end], q, selected.beta_values,
                    dither=dither_blocks[start:end],
                )
            else:
                quantized, _ldlq_metadata = quantize_rows_ldlq(
                    matrix[start:end], None, q, selected.beta_values,
                    dither=dither_blocks[start:end],
                    relative_damp=ldlq_relative_damp,
                    precomputed_lower=cached_lower,
                    absolute_damping_added=cached_damping,
                )
            restored_chunk = quantized.dequantize()
            if signs is not None:
                # W + (Q(WR) - WR) R^T is algebraically Q(WR) R^T,
                # but avoids BF16 changes from roundtrip error when Q is identity.
                restored_chunk = original[start:end].float() + signed_hadamard(
                    restored_chunk - matrix[start:end], signs, inverse=True)
            reconstructed[start:end].copy_(restored_chunk)
            if output_artifacts is not None:
                quantized_parts.append(quantized)
        replacements.append((original, reconstructed.to(dtype=original.dtype)))
        timing["quantization_rows_per_chunk"] = quantization_rows_per_chunk
        timing["quantize_reconstruct_seconds"] = elapsed_since(phase_started, original.device)

        if output_artifacts is not None:
            phase_started = time.perf_counter()
            full_quantized = QuantizedRows(
                codes=torch.cat([part.codes for part in quantized_parts], dim=0),
                beta_indices=torch.cat([part.beta_indices for part in quantized_parts], dim=0),
                dither=dither_blocks_cpu,
                row_norms=torch.cat([part.row_norms for part in quantized_parts], dim=0),
                q=q,
                betas=selected.beta_values,
                feature_count=features,
                output_dtype=matrix.dtype,
            )
            weight_name = f"{name}.weight" if name else "weight"
            staged_artifacts[weight_name] = PackedNestQuantTensor.from_quantized_rows(
                weight_name, full_quantized, dither_generator_state=dither_state,
                verify_generator_state=False,
            )
            timing["packing_seconds"] = elapsed_since(phase_started, original.device)

        stats["layers"] += 1
        stats["parameters"] += original.numel()
        stats["row_count"] += rows
        stats["e8_block_count"] += rows * (features // 8)
        stats["layer_beta_subsets"][name] = [coefficients[index] for index in selected.beta_indices]
        timing["total_seconds"] = elapsed_since(layer_started, original.device)
        stats["layer_timings"][name] = timing
        if progress_callback is not None:
            progress_callback({"layer": name, "completed_layers": stats["layers"],
                               "total_layers": total_layers, "shape": [rows, features],
                               "timing_seconds": timing})
    if stats["layers"] == 0:
        raise ValueError("model has no torch.nn.Linear modules to quantize")
    # Keep the model untouched if any later layer fails (e.g. beta coverage).
    with torch.no_grad():
        for original, reconstructed in replacements:
            original.copy_(reconstructed)
    if output_artifacts is not None:
        output_artifacts.update(staged_artifacts)
    stats["storage_estimate"] = _estimate_nestquant_w4_storage(stats)
    stats["storage_estimate"]["input_embedding_weight_is_included"] = stats["input_embedding_weight_shared_with_linear"]
    return stats


def _input_embedding_weight(model: nn.Module) -> torch.Tensor | None:
    getter = getattr(model, "get_input_embeddings", None)
    if getter is None:
        return None
    embedding = getter()
    return getattr(embedding, "weight", None)


def _estimate_rtn_w4_storage(stats: dict) -> dict:
    """Estimate packed RTN payload size; the adapter itself writes no packed data."""
    parameters = stats["parameters"]
    code_bits = 4 * parameters
    scale_bits = 32 * stats["scale_count"]  # FP32 scale per group
    total_bits = code_bits + scale_bits
    return {
        "status": "estimate_only_no_packed_tensor_written",
        "assumptions": "4-bit packed signed codes; FP32 scale per group; ignores tensor padding and container headers",
        "code_bits": code_bits,
        "scale_metadata_bits": scale_bits,
        "other_metadata_bits": 0,
        "total_bits": total_bits,
        "estimated_bytes": math.ceil(total_bits / 8),
        "effective_bits_per_quantized_parameter": total_bits / parameters,
    }


def _estimate_nestquant_w4_storage(stats: dict) -> dict:
    """Estimate a straightforward packed representation, not a real checkpoint."""
    parameters = stats["parameters"]
    residue_bits = math.ceil(math.log2(stats["q"]))
    code_bits = residue_bits * parameters
    beta_index_bits_per_block = math.ceil(math.log2(stats["selected_beta_count"]))
    beta_index_bits = beta_index_bits_per_block * stats["e8_block_count"]
    row_norm_bits = 32 * stats["row_count"]
    beta_table_bits = 32 * stats["selected_beta_count"] * stats["layers"]
    run_metadata_bits = 96  # 32-bit q plus a 64-bit reproducible dither seed
    total_bits = code_bits + beta_index_bits + row_norm_bits + beta_table_bits + run_metadata_bits
    return {
        "status": "estimate_only_no_packed_tensor_written",
        "assumptions": f"q-ary residues fixed-width packed at {residue_bits} bits each; beta IDs fixed-width packed; FP32 row norms and per-layer beta tables; 32-bit q and 64-bit dither seed; ignores alignment and container headers",
        "residue_code_bits": code_bits,
        "beta_index_bits": beta_index_bits,
        "row_norm_metadata_bits": row_norm_bits,
        "beta_table_metadata_bits": beta_table_bits,
        "run_metadata_bits": run_metadata_bits,
        "total_bits": total_bits,
        "estimated_bytes": math.ceil(total_bits / 8),
        "effective_bits_per_quantized_parameter": total_bits / parameters,
        "ideal_arithmetic_code_bits_per_residue": math.log2(stats["q"]),
        "note": "Entropy/arithmetic coding of q-ary residues could reduce code payload; no such coder exists in this project yet.",
    }

"""Compare fixed NestQuant upstream and local intermediate tensors on CPU."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import subprocess
import sys
import types
from datetime import datetime, timezone
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from nestquant.core.quantizer import quantize_e8_blocks
from nestquant.core.codec import decode_nested, encode_nested
from nestquant.core.rotation import hadamard
from nestquant.evaluation.upstream_alignment import tensor_alignment


PINNED_UPSTREAM_COMMIT = "6b269f84264afad990a09248d8eeb45d5f6248ff"
DEFAULT_COEFFICIENTS = (3.47, 4.74, 6.90, 18.11)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream-root", type=Path, required=True)
    parser.add_argument("--q", type=int, default=14)
    parser.add_argument("--coefficients", type=float, nargs="+", default=DEFAULT_COEFFICIENTS)
    parser.add_argument("--dimensions", type=int, nargs="+", default=(8, 16, 32))
    parser.add_argument("--seeds", type=int, nargs="+", default=(58, 59))
    parser.add_argument("--atol", type=float, default=2e-6)
    parser.add_argument("--rtol", type=float, default=2e-6)
    parser.add_argument(
        "--output", type=Path,
        default=PROJECT_ROOT / "results/raw/fixed_upstream_alignment.json",
    )
    args = parser.parse_args()
    if args.q < 2 or not args.coefficients or args.atol < 0 or args.rtol < 0:
        raise ValueError("invalid q, coefficients, or tolerances")
    if any(dimension < 8 or dimension % 8 or dimension & (dimension - 1)
           for dimension in args.dimensions):
        raise ValueError("dimensions must be power-of-two multiples of 8")
    upstream_root = args.upstream_root.resolve()
    commit = git_commit(upstream_root)
    if commit != PINNED_UPSTREAM_COMMIT:
        raise ValueError(f"upstream commit mismatch: {commit}")
    sources = {
        name: sha256_file(upstream_root / name)
        for name in ("coder.py", "e8.py", "hadamard.py", "parallel/quant_utils.py", "utils.py")
    }
    # Upstream utils.py imports the optional CUDA extension whenever CUDA is
    # visible, even for explicitly CPU tensors.  Keep this alignment run on
    # its original SciPy CPU branch and fail loudly if the CUDA symbol is used.
    if "fast_hadamard_transform" not in sys.modules:
        extension_stub = types.ModuleType("fast_hadamard_transform")

        def forbidden_cuda_hadamard(*_args, **_kwargs):
            raise RuntimeError("CUDA Hadamard is outside this CPU alignment protocol")

        extension_stub.hadamard_transform = forbidden_cuda_hadamard
        sys.modules["fast_hadamard_transform"] = extension_stub
    sys.path.insert(0, str(upstream_root))
    try:
        upstream_coder = importlib.import_module("coder")
        upstream_hadamard = importlib.import_module("hadamard")
    finally:
        sys.path.pop(0)

    beta_values = tuple(value / args.q for value in args.coefficients)
    cases = []
    all_passed = True
    for dimension in args.dimensions:
        for seed in args.seeds:
            generator = torch.Generator(device="cpu").manual_seed(seed + dimension)
            inputs = torch.randn(3, dimension, generator=generator, dtype=torch.float32)
            weights = torch.randn(5, dimension, generator=generator, dtype=torch.float32)
            case = compare_case(
                inputs, weights, q=args.q, coefficients=tuple(args.coefficients),
                beta_values=beta_values, upstream_coder=upstream_coder,
                upstream_hadamard=upstream_hadamard, atol=args.atol, rtol=args.rtol,
            )
            case.update({"dimension": dimension, "seed": seed})
            cases.append(case)
            all_passed = all_passed and case["passed"]
            print(
                f"n={dimension} seed={seed}: passed={case['passed']} "
                f"output_max_abs={case['stages']['linear_output']['max_abs_error']:.3e}",
                flush=True,
            )
    payload = {
        "schema_version": 1,
        "kind": "fixed_upstream_intermediate_alignment",
        "status": "passed" if all_passed else "failed",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "upstream": {
            "repository": "https://github.com/cookiedoth/nestquant",
            "root": str(upstream_root),
            "commit": commit,
            "source_sha256": sources,
        },
        "local": {"git_commit": git_commit(PROJECT_ROOT)},
        "protocol": {
            "device": "cpu",
            "dtype": "float32",
            "q": args.q,
            "coefficients_project_scale": list(args.coefficients),
            "beta_values_codec_scale": list(beta_values),
            "use_dither": False,
            "try_all": True,
            "upstream_hadamard_backend": "original_scipy_cpu_branch",
            "cuda_extension_import": "nonexecuting_stub_due_to_upstream_eager_import",
            "dimensions": list(args.dimensions),
            "seeds": list(args.seeds),
            "atol": args.atol,
            "rtol": args.rtol,
        },
        "cases": cases,
        "scope": (
            "Power-of-two single-process quantsim path only; excludes FairScale "
            "re_glue, non-power-of-two Kronecker matrices, LDLQ, and CUDA kernels."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if not all_passed:
        raise RuntimeError("one or more fixed-upstream alignment cases failed")
    print(f"Saved fixed-upstream alignment: {args.output}")


def compare_case(
    inputs: torch.Tensor,
    weights: torch.Tensor,
    *,
    q: int,
    coefficients: tuple[float, ...],
    beta_values: tuple[float, ...],
    upstream_coder,
    upstream_hadamard,
    atol: float,
    rtol: float,
) -> dict:
    upstream_input = upstream_stages(
        inputs, q, coefficients, upstream_coder, upstream_hadamard
    )
    upstream_weight = upstream_stages(
        weights, q, coefficients, upstream_coder, upstream_hadamard
    )
    local_input = local_stages(inputs, q, beta_values)
    local_weight = local_stages(weights, q, beta_values)
    local_input_from_upstream_rotated = local_quantize_rotated(
        upstream_input["rotated"], upstream_input["row_norms"], q, beta_values
    )
    local_weight_from_upstream_rotated = local_quantize_rotated(
        upstream_weight["rotated"], upstream_weight["row_norms"], q, beta_values
    )
    upstream_output = upstream_input["reconstructed"] @ upstream_weight["reconstructed"].T
    local_output = local_input["local_coordinate"] @ local_weight["local_coordinate"].T
    stages = {
        "input_normalized": tensor_alignment(
            upstream_input["normalized"], local_input["normalized"]
        ),
        "input_rotated": tensor_alignment(
            upstream_input["rotated"], local_input["rotated"]
        ),
        "input_decoded_rotated": tensor_alignment(
            upstream_input["decoded_rotated"], local_input["decoded_rotated"]
        ),
        "input_reconstructed": tensor_alignment(
            upstream_input["reconstructed"], local_input["reconstructed"]
        ),
        "weight_decoded_rotated": tensor_alignment(
            upstream_weight["decoded_rotated"], local_weight["decoded_rotated"]
        ),
        "weight_reconstructed": tensor_alignment(
            upstream_weight["reconstructed"], local_weight["reconstructed"]
        ),
        "linear_output": tensor_alignment(upstream_output, local_output),
    }
    exact = {
        "input_codes": torch.equal(upstream_input["codes"], local_input["codes"]),
        "input_beta_indices": torch.equal(
            upstream_input["beta_indices"], local_input["beta_indices"]
        ),
        "weight_codes": torch.equal(upstream_weight["codes"], local_weight["codes"]),
        "weight_beta_indices": torch.equal(
            upstream_weight["beta_indices"], local_weight["beta_indices"]
        ),
    }
    same_rotated_exact = {
        "input_codes": torch.equal(
            upstream_input["codes"], local_input_from_upstream_rotated["codes"]
        ),
        "input_beta_indices": torch.equal(
            upstream_input["beta_indices"],
            local_input_from_upstream_rotated["beta_indices"],
        ),
        "weight_codes": torch.equal(
            upstream_weight["codes"], local_weight_from_upstream_rotated["codes"]
        ),
        "weight_beta_indices": torch.equal(
            upstream_weight["beta_indices"],
            local_weight_from_upstream_rotated["beta_indices"],
        ),
    }
    candidate_diagnostics = {
        "input": compare_candidate_codecs(
            upstream_input["rotated"], q, coefficients, beta_values, upstream_coder
        ),
        "weight": compare_candidate_codecs(
            upstream_weight["rotated"], q, coefficients, beta_values, upstream_coder
        ),
    }
    comparisons = (
        (upstream_input["normalized"], local_input["normalized"]),
        (upstream_input["rotated"], local_input["rotated"]),
        (upstream_input["decoded_rotated"], local_input["decoded_rotated"]),
        (upstream_input["reconstructed"], local_input["reconstructed"]),
        (upstream_weight["decoded_rotated"], local_weight["decoded_rotated"]),
        (upstream_weight["reconstructed"], local_weight["reconstructed"]),
        (upstream_output, local_output),
    )
    passed = all(exact.values()) and all(
        torch.allclose(reference, actual, atol=atol, rtol=rtol)
        for reference, actual in comparisons
    )
    return {
        "passed": passed,
        "exact_integer_state": exact,
        "same_rotated_input_exact_integer_state": same_rotated_exact,
        "same_rotated_input_candidate_diagnostics": candidate_diagnostics,
        "stages": stages,
    }


def compare_candidate_codecs(rotated, q, coefficients, beta_values, coder) -> list[dict]:
    pieces = rotated.reshape(-1, 8)
    records = []
    for coefficient, beta in zip(coefficients, beta_values):
        upstream_codes, upstream_overload = coder.encode(pieces, q, None, coefficient / q)
        upstream_reconstructed = coder.decode(
            upstream_codes, q, None, coefficient / q
        )
        local_codes, local_overload = encode_nested(pieces / beta, q)
        local_reconstructed = decode_nested(
            local_codes, q, dtype=pieces.dtype
        ) * beta
        records.append({
            "coefficient": coefficient,
            "codes_exact": torch.equal(upstream_codes, local_codes),
            "code_entry_mismatch_count": int((upstream_codes != local_codes).sum().item()),
            "overload_exact": torch.equal(upstream_overload, local_overload),
            "overload_mismatch_count": int(
                (upstream_overload != local_overload).sum().item()
            ),
            "reconstruction": tensor_alignment(
                upstream_reconstructed, local_reconstructed
            ),
        })
    return records


def upstream_stages(tensor, q, coefficients, coder, upstream_hadamard) -> dict:
    row_norms = torch.linalg.vector_norm(tensor, dim=-1)
    normalized = tensor / row_norms.unsqueeze(-1) * (tensor.shape[-1] ** 0.5)
    rotated = upstream_hadamard.kron_h_ip(normalized)
    encoded = coder.encode_matrix(
        rotated, q, coefficients, 0, use_dither=False, try_all=True
    )
    decoded_rotated = coder.decode_matrix(
        encoded, q, coefficients, 0, use_dither=False, try_all=True
    )
    reconstructed = upstream_hadamard.kron_h_ip(decoded_rotated, inverse=True)
    reconstructed = reconstructed * row_norms.unsqueeze(-1) / (tensor.shape[-1] ** 0.5)
    return {
        "normalized": normalized,
        "rotated": rotated,
        "codes": encoded["enc"].reshape(tensor.shape[0], -1, 8),
        "beta_indices": encoded["beta_id"].reshape(tensor.shape[0], -1).to(torch.int64),
        "decoded_rotated": decoded_rotated,
        "reconstructed": reconstructed,
        "row_norms": row_norms,
    }


def local_stages(tensor, q, beta_values) -> dict:
    row_norms = torch.linalg.vector_norm(tensor, dim=-1)
    normalized = tensor / row_norms.unsqueeze(-1) * (tensor.shape[-1] ** 0.5)
    rotated = hadamard(normalized)
    result = local_quantize_rotated(rotated, row_norms, q, beta_values)
    result.update({"normalized": normalized, "rotated": rotated})
    return result


def local_quantize_rotated(rotated, row_norms, q, beta_values) -> dict:
    quantized = quantize_e8_blocks(
        rotated.reshape(rotated.shape[0], -1, 8), q, beta_values
    )
    decoded_rotated = quantized.dequantize().reshape_as(rotated)
    local_coordinate = (
        decoded_rotated * row_norms.unsqueeze(-1) / (rotated.shape[-1] ** 0.5)
    )
    reconstructed = hadamard(decoded_rotated)
    reconstructed = (
        reconstructed * row_norms.unsqueeze(-1) / (rotated.shape[-1] ** 0.5)
    )
    return {
        "codes": quantized.codes,
        "beta_indices": quantized.beta_indices,
        "decoded_rotated": decoded_rotated,
        "local_coordinate": local_coordinate,
        "reconstructed": reconstructed,
        "row_norms": row_norms,
    }


def git_commit(root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root,
        capture_output=True, text=True, check=True,
    )
    return result.stdout.strip()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    main()

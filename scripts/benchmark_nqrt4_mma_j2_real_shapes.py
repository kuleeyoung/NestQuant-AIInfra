from pathlib import Path

import torch
from torch.utils.cpp_extension import load

from nestquant.core.codec import generate_e8_dither_stateless
from nestquant.core.lattice import nearest_e8

ROOT = Path(__file__).resolve().parents[1]
ext = load(
    name="nqrt4_mma_j2_ext",
    sources=[
        str(ROOT / "nestquant/runtime/nqrt4_mma_j2_binding.cpp"),
        str(ROOT / "nestquant/runtime/nqrt4_mma_j2.cu"),
    ],
    extra_cuda_cflags=["-O3", "-Xptxas=-v"],
    verbose=False,
)


def make_packed(n: int, k: int):
    coordinates = torch.arange(n * k, dtype=torch.int64)
    values = ((coordinates * 11 + 4) % 14).to(torch.uint8).reshape(-1, 2)
    packed_codes = values[:, 0] | (values[:, 1] << 4)
    blocks = n * k // 8
    beta_values = ((torch.arange(blocks, dtype=torch.int64) * 7 + 2) % 4).to(torch.uint8)
    beta_groups = beta_values.reshape(-1, 4)
    packed_betas = (beta_groups[:, 0] | (beta_groups[:, 1] << 2) |
                    (beta_groups[:, 2] << 4) | (beta_groups[:, 3] << 6))
    return packed_codes.cuda(), packed_betas.cuda()


def unpack_formula_codes(start_block: int, block_count: int):
    coordinates = torch.arange(
        start_block * 8, (start_block + block_count) * 8,
        device="cuda", dtype=torch.int64,
    )
    return ((coordinates * 11 + 4) % 14).float().reshape(block_count, 8)


def beta_formula(start_block: int, block_count: int):
    blocks = torch.arange(start_block, start_block + block_count, device="cuda", dtype=torch.int64)
    return ((blocks * 7 + 2) % 4).long()


def reference_qproj(activation, row_norms, betas, seed, offset, rows_per_chunk=64):
    n, k = 2048, 2048
    blocks_per_row = k // 8
    output = torch.empty((16, n), device="cuda", dtype=torch.float32)
    for row_start in range(0, n, rows_per_chunk):
        rows = min(rows_per_chunk, n - row_start)
        start_block = row_start * blocks_per_row
        block_count = rows * blocks_per_row
        code = unpack_formula_codes(start_block, block_count)
        half = 0.5 * code[:, 7:8]
        coset = torch.cat([
            2 * code[:, 0:1] - code[:, 1:2] + half,
            code[:, 1:6] - code[:, 2:7] + half,
            code[:, 6:7] + half,
            half,
        ], dim=1)
        dither = generate_e8_dither_stateless(
            block_count, seed, block_offset=offset + start_block, device="cuda"
        )
        shifted = coset - dither
        coarse, _ = nearest_e8(shifted / 14.0)
        reconstructed = shifted - 14.0 * coarse
        beta_id = beta_formula(start_block, block_count)
        scales = betas[beta_id]
        scales *= row_norms[row_start:row_start + rows].repeat_interleave(blocks_per_row)
        scales *= k ** -0.5
        weight = (reconstructed * scales[:, None]).to(torch.bfloat16).reshape(rows, k)
        output[:, row_start:row_start + rows] = activation.float() @ weight.float().T
    return output


def timed(fn, warmup: int, iterations: int):
    for _ in range(warmup):
        fn()
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    start.record()
    for _ in range(iterations):
        fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000 / iterations


def median_timed(fn, warmup=25, iterations=200, repeats=7):
    values = sorted(timed(fn, warmup, iterations) for _ in range(repeats))
    return values[len(values) // 2], values[0], values[-1]


betas = torch.tensor([0.25, 0.5, 0.75, 1.0], device="cuda", dtype=torch.float32)
seed, offset = 58, 37

# Full q_proj correctness at an actual Llama-3.2-1B Linear shape.
n, k = 2048, 2048
activation = ((torch.arange(16 * k, device="cuda").reshape(16, k) % 29) - 14).to(torch.bfloat16) / 8
row_norms = torch.linspace(0.625, 1.875, n, device="cuda", dtype=torch.float32)
packed_codes, packed_betas = make_packed(n, k)
actual = ext.run_m16(activation, packed_codes, packed_betas, row_norms, betas, offset, seed)
reference = reference_qproj(activation, row_norms, betas, seed, offset)
error = (actual - reference).abs()
print(f"correctness shape=16x{k} @ {n}x{k} max_abs_error={error.max().item():.6g} "
      f"mean_abs_error={error.mean().item():.6g} finite={bool(torch.isfinite(actual).all())}")
for splits in (2, 4, 8, 16):
    split_output = ext.run_m16_splitk(
        activation, packed_codes, packed_betas, row_norms, betas, offset, seed, splits
    )
    split_error = (split_output - reference).abs()
    print(f"correctness splitk={splits} max_abs_error={split_error.max().item():.6g} "
          f"mean_abs_error={split_error.mean().item():.6g} "
          f"finite={bool(torch.isfinite(split_output).all())}")
del actual, reference, error, packed_codes, packed_betas, row_norms, activation
torch.cuda.empty_cache()

shapes = [
    ("q_proj", 2048, 2048),
    ("gate_up_proj", 8192, 2048),
    ("down_proj", 2048, 8192),
]
for name, n, k in shapes:
    activation = torch.randn((16, k), device="cuda", dtype=torch.bfloat16)
    resident_weight = torch.randn((n, k), device="cuda", dtype=torch.bfloat16)
    row_norms = torch.linspace(0.625, 1.875, n, device="cuda", dtype=torch.float32)
    packed_codes, packed_betas = make_packed(n, k)
    fused = median_timed(lambda: ext.run_m16(
        activation, packed_codes, packed_betas, row_norms, betas, offset, seed))
    split_results = {}
    split_candidates = (1, 2, 4, 8, 16, 32, 64)
    if name == "down_proj":
        split_candidates += (128, 256)
    for splits in split_candidates:
        split_results[splits] = median_timed(lambda splits=splits: ext.run_m16_splitk(
            activation, packed_codes, packed_betas, row_norms, betas, offset, seed, splits
        ))
    resident = median_timed(lambda: activation @ resident_weight.T)
    automatic = median_timed(lambda: ext.run_m16_auto(
        activation, packed_codes, packed_betas, row_norms, betas, offset, seed))
    packed_mib = (packed_codes.numel() + packed_betas.numel()) / 2**20
    bf16_mib = resident_weight.numel() * resident_weight.element_size() / 2**20
    print(f"benchmark name={name} M=16 K={k} N={n} "
          f"j2_median_us={fused[0]:.6g} j2_min_us={fused[1]:.6g} j2_max_us={fused[2]:.6g} "
          f"resident_median_us={resident[0]:.6g} ratio={fused[0] / resident[0]:.3f} "
          f"packed_mib={packed_mib:.3f} bf16_mib={bf16_mib:.3f}")
    print(f"auto name={name} median_us={automatic[0]:.6g} min_us={automatic[1]:.6g} "
          f"max_us={automatic[2]:.6g} speedup_vs_unsplit={fused[0] / automatic[0]:.3f}")
    for splits, measurement in split_results.items():
        workspace_mib = splits * 16 * n * 4 / 2**20
        print(f"splitk name={name} splits={splits} median_us={measurement[0]:.6g} "
              f"min_us={measurement[1]:.6g} max_us={measurement[2]:.6g} "
              f"speedup_vs_unsplit={fused[0] / measurement[0]:.3f} "
              f"workspace_mib={workspace_mib:.3f}")
    del activation, resident_weight, row_norms, packed_codes, packed_betas
    torch.cuda.empty_cache()

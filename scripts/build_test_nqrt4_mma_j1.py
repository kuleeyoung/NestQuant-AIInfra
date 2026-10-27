from pathlib import Path
import torch
from torch.utils.cpp_extension import load

from nestquant.core.codec import generate_e8_dither_stateless
from nestquant.core.lattice import nearest_e8

ROOT = Path(__file__).resolve().parents[1]
ext = load(
    name="nqrt4_mma_j1_ext",
    sources=[str(ROOT / "nestquant/runtime/nqrt4_mma_j1_binding.cpp"),
             str(ROOT / "nestquant/runtime/nqrt4_mma_j1.cu")],
    extra_cuda_cflags=["-O3"], verbose=False,
)

codes = torch.tensor([(i * 5 + 3) % 14 for i in range(128)], dtype=torch.uint8).reshape(16, 8)
beta_ids = torch.tensor([(i * 3 + 1) % 4 for i in range(16)], dtype=torch.uint8)
packed_codes = torch.zeros(64, dtype=torch.uint8)
packed_betas = torch.zeros(4, dtype=torch.uint8)
for i, value in enumerate(codes.flatten().tolist()):
    packed_codes[i // 2] |= value << ((i % 2) * 4)
for i, value in enumerate(beta_ids.tolist()):
    packed_betas[i // 4] |= value << ((i % 4) * 2)

activation = (torch.arange(256, device="cuda").reshape(16, 16) % 19 - 9).to(torch.bfloat16) / 4
row_norms = torch.linspace(0.75, 1.625, 8, device="cuda", dtype=torch.float32)
betas = torch.tensor([0.25, 0.5, 0.75, 1.0], device="cuda", dtype=torch.float32)

for seed, offset in [(58, 37), (0, 0), (1234567, 4096)]:
    code_f = codes.to(device="cuda", dtype=torch.float32)
    h = 0.5 * code_f[:, 7:8]
    coset = torch.cat([
        2 * code_f[:, 0:1] - code_f[:, 1:2] + h,
        code_f[:, 1:6] - code_f[:, 2:7] + h,
        code_f[:, 6:7] + h, h,
    ], dim=1)
    dither = generate_e8_dither_stateless(16, seed, block_offset=offset, device="cuda")
    shifted = coset - dither
    coarse, _ = nearest_e8(shifted / 14.0)
    reconstructed = shifted - 14.0 * coarse
    scale = betas[beta_ids.to(device="cuda", dtype=torch.long)]
    scale = scale * row_norms.repeat_interleave(2) / 4.0
    weight = (reconstructed * scale[:, None]).to(torch.bfloat16).reshape(8, 16)
    reference = activation.float() @ weight.float().T
    actual = ext.run(activation.contiguous(), packed_codes.cuda(), packed_betas.cuda(),
                     row_norms, betas, offset, seed, 0.25)
    error = (actual - reference).abs()
    print(f"seed={seed} offset={offset} max_abs_error={error.max().item():.6g} "
          f"mean_abs_error={error.mean().item():.6g} finite={bool(torch.isfinite(actual).all())}")

def benchmark(fn, warmup=100, iterations=5000):
    for _ in range(warmup): fn()
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    start.record()
    for _ in range(iterations): fn()
    end.record(); end.synchronize()
    return start.elapsed_time(end) * 1000 / iterations

packed_codes_cuda, packed_betas_cuda = packed_codes.cuda(), packed_betas.cuda()
fused_us = benchmark(lambda: ext.run(activation, packed_codes_cuda, packed_betas_cuda,
                                     row_norms, betas, 37, 58, 0.25))
resident_us = benchmark(lambda: activation @ weight.T)
print(f"fused_j1_us={fused_us:.6g} resident_bf16_us={resident_us:.6g} ratio={fused_us/resident_us:.3f}")

# K=128: 8 output rows, 16 E8 blocks per row.
codes128 = torch.tensor([(i * 7 + 2) % 14 for i in range(1024)], dtype=torch.uint8).reshape(128, 8)
beta_ids128 = torch.tensor([(i * 5 + 3) % 4 for i in range(128)], dtype=torch.uint8)
packed_codes128 = torch.zeros(512, dtype=torch.uint8)
packed_betas128 = torch.zeros(32, dtype=torch.uint8)
for i, value in enumerate(codes128.flatten().tolist()):
    packed_codes128[i // 2] |= value << ((i % 2) * 4)
for i, value in enumerate(beta_ids128.tolist()):
    packed_betas128[i // 4] |= value << ((i % 4) * 2)
activation128 = (torch.arange(2048, device="cuda").reshape(16, 128) % 23 - 11).to(torch.bfloat16) / 4

for seed, offset in [(58, 37), (0, 0), (1234567, 4096)]:
    code_f = codes128.to(device="cuda", dtype=torch.float32)
    h = 0.5 * code_f[:, 7:8]
    coset = torch.cat([2*code_f[:,0:1]-code_f[:,1:2]+h,
                       code_f[:,1:6]-code_f[:,2:7]+h,
                       code_f[:,6:7]+h, h], dim=1)
    dither = generate_e8_dither_stateless(128, seed, block_offset=offset, device="cuda")
    shifted = coset - dither
    coarse, _ = nearest_e8(shifted / 14.0)
    reconstructed = shifted - 14.0 * coarse
    scale = betas[beta_ids128.cuda().long()] * row_norms.repeat_interleave(16) / (128.0 ** 0.5)
    weight128 = (reconstructed * scale[:,None]).to(torch.bfloat16).reshape(8,128)
    reference128 = activation128.float() @ weight128.float().T
    actual128 = ext.run_k128(activation128.contiguous(), packed_codes128.cuda(), packed_betas128.cuda(),
                             row_norms, betas, offset, seed)
    error128 = (actual128-reference128).abs()
    print(f"k128 seed={seed} offset={offset} max_abs_error={error128.max().item():.6g} "
          f"mean_abs_error={error128.mean().item():.6g} finite={bool(torch.isfinite(actual128).all())}")

def repeated_benchmark(fn, repeats=9):
    values = [benchmark(fn, warmup=100, iterations=2000) for _ in range(repeats)]
    values.sort()
    return values[len(values)//2], values[0], values[-1]

pc128, pb128 = packed_codes128.cuda(), packed_betas128.cuda()
fused128 = repeated_benchmark(lambda: ext.run_k128(
    activation128, pc128, pb128, row_norms, betas, 37, 58))
resident128 = repeated_benchmark(lambda: activation128 @ weight128.T)
print(f"k128_fused_us median={fused128[0]:.6g} min={fused128[1]:.6g} max={fused128[2]:.6g}")
print(f"k128_resident_us median={resident128[0]:.6g} min={resident128[1]:.6g} max={resident128[2]:.6g}")
print(f"k128_median_ratio={fused128[0]/resident128[0]:.3f}")

# K=128, N=64: eight independent 16x8 output tiles in one launch.
codes_n64 = torch.tensor([(i * 11 + 4) % 14 for i in range(8192)], dtype=torch.uint8).reshape(1024, 8)
beta_ids_n64 = torch.tensor([(i * 7 + 2) % 4 for i in range(1024)], dtype=torch.uint8)
packed_codes_n64 = torch.zeros(4096, dtype=torch.uint8)
packed_betas_n64 = torch.zeros(256, dtype=torch.uint8)
for i, value in enumerate(codes_n64.flatten().tolist()):
    packed_codes_n64[i // 2] |= value << ((i % 2) * 4)
for i, value in enumerate(beta_ids_n64.tolist()):
    packed_betas_n64[i // 4] |= value << ((i % 4) * 2)
row_norms_n64 = torch.linspace(0.625, 1.875, 64, device="cuda", dtype=torch.float32)
pc_n64, pb_n64 = packed_codes_n64.cuda(), packed_betas_n64.cuda()

for seed, offset in [(58, 37), (0, 0), (1234567, 4096)]:
    code_f = codes_n64.to(device="cuda", dtype=torch.float32)
    h = 0.5 * code_f[:, 7:8]
    coset = torch.cat([2*code_f[:,0:1]-code_f[:,1:2]+h,
                       code_f[:,1:6]-code_f[:,2:7]+h,
                       code_f[:,6:7]+h, h], dim=1)
    dither = generate_e8_dither_stateless(1024, seed, block_offset=offset, device="cuda")
    shifted = coset - dither
    coarse, _ = nearest_e8(shifted / 14.0)
    reconstructed = shifted - 14.0 * coarse
    scale = betas[beta_ids_n64.cuda().long()] * row_norms_n64.repeat_interleave(16) / (128.0 ** 0.5)
    weight_n64 = (reconstructed * scale[:,None]).to(torch.bfloat16).reshape(64,128)
    reference_n64 = activation128.float() @ weight_n64.float().T
    actual_n64 = ext.run_k128_n64(activation128, pc_n64, pb_n64, row_norms_n64, betas, offset, seed)
    error_n64 = (actual_n64-reference_n64).abs()
    print(f"n64 seed={seed} offset={offset} max_abs_error={error_n64.max().item():.6g} "
          f"mean_abs_error={error_n64.mean().item():.6g} finite={bool(torch.isfinite(actual_n64).all())}")

fused_n64 = repeated_benchmark(lambda: ext.run_k128_n64(
    activation128, pc_n64, pb_n64, row_norms_n64, betas, 37, 58))
resident_n64 = repeated_benchmark(lambda: activation128 @ weight_n64.T)
print(f"n64_fused_us median={fused_n64[0]:.6g} min={fused_n64[1]:.6g} max={fused_n64[2]:.6g}")
print(f"n64_resident_us median={resident_n64[0]:.6g} min={resident_n64[1]:.6g} max={resident_n64[2]:.6g}")
print(f"n64_median_ratio={fused_n64[0]/resident_n64[0]:.3f}")

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
    verbose=True,
)

codes = torch.tensor([(i * 11 + 4) % 14 for i in range(8192)], dtype=torch.uint8).reshape(1024, 8)
beta_ids = torch.tensor([(i * 7 + 2) % 4 for i in range(1024)], dtype=torch.uint8)
packed_codes = torch.zeros(4096, dtype=torch.uint8)
packed_betas = torch.zeros(256, dtype=torch.uint8)
for i, value in enumerate(codes.flatten().tolist()):
    packed_codes[i // 2] |= value << ((i % 2) * 4)
for i, value in enumerate(beta_ids.tolist()):
    packed_betas[i // 4] |= value << ((i % 4) * 2)

activation = (torch.arange(2048, device="cuda").reshape(16, 128) % 23 - 11).to(torch.bfloat16) / 4
row_norms = torch.linspace(0.625, 1.875, 64, device="cuda", dtype=torch.float32)
betas = torch.tensor([0.25, 0.5, 0.75, 1.0], device="cuda", dtype=torch.float32)
packed_codes, packed_betas = packed_codes.cuda(), packed_betas.cuda()

last_weight = None
for seed, offset in [(58, 37), (0, 0), (1234567, 4096)]:
    code_f = codes.cuda().float()
    h = 0.5 * code_f[:, 7:8]
    coset = torch.cat([
        2 * code_f[:, 0:1] - code_f[:, 1:2] + h,
        code_f[:, 1:6] - code_f[:, 2:7] + h,
        code_f[:, 6:7] + h,
        h,
    ], dim=1)
    dither = generate_e8_dither_stateless(1024, seed, block_offset=offset, device="cuda")
    shifted = coset - dither
    coarse, _ = nearest_e8(shifted / 14.0)
    reconstructed = shifted - 14.0 * coarse
    scale = betas[beta_ids.cuda().long()] * row_norms.repeat_interleave(16) / (128.0**0.5)
    last_weight = (reconstructed * scale[:, None]).to(torch.bfloat16).reshape(64, 128)
    reference = activation.float() @ last_weight.float().T
    actual = ext.run_k128_n64(activation, packed_codes, packed_betas, row_norms, betas, offset, seed)
    dynamic = ext.run_m16(activation, packed_codes, packed_betas, row_norms, betas, offset, seed)
    error = (actual - reference).abs()
    dynamic_error = (dynamic - reference).abs()
    print(f"seed={seed} offset={offset} max_abs_error={error.max().item():.6g} "
          f"mean_abs_error={error.mean().item():.6g} finite={bool(torch.isfinite(actual).all())}")
    print(f"dynamic seed={seed} offset={offset} max_abs_error={dynamic_error.max().item():.6g} "
          f"mean_abs_error={dynamic_error.mean().item():.6g} finite={bool(torch.isfinite(dynamic).all())}")
    if not torch.isfinite(actual).all() or not torch.isfinite(dynamic).all():
        raise AssertionError("J2 produced a non-finite output")
    for splits in (1, 2, 4, 8):
        split_output = ext.run_m16_splitk(
            activation, packed_codes, packed_betas, row_norms, betas, offset, seed, splits
        )
        split_repeat = ext.run_m16_splitk(
            activation, packed_codes, packed_betas, row_norms, betas, offset, seed, splits
        )
        split_error = (split_output - reference).abs()
        print(f"splitk={splits} max_abs_error={split_error.max().item():.6g} "
              f"mean_abs_error={split_error.mean().item():.6g} "
              f"deterministic={bool(torch.equal(split_output, split_repeat))}")
        if not torch.isfinite(split_output).all():
            raise AssertionError(f"J2 split-K={splits} produced a non-finite output")
        if not torch.equal(split_output, split_repeat):
            raise AssertionError(f"J2 split-K={splits} is not deterministic")

# A PyTorch CUDA extension must launch on the current stream. Clone on the
# side stream before synchronizing so a regression to stream 0 creates a real
# dependency race instead of being hidden by a later default-stream read.
side_stream = torch.cuda.Stream()
with torch.cuda.stream(side_stream):
    side_output = ext.run_m16_splitk(
        activation, packed_codes, packed_betas, row_norms, betas, 4096, 1234567, 8
    )
    side_snapshot = side_output.clone()
side_stream.synchronize()
side_error = (side_snapshot - reference).abs()
print(f"non_default_stream max_abs_error={side_error.max().item():.6g} "
      f"finite={bool(torch.isfinite(side_snapshot).all())}")
if not torch.isfinite(side_snapshot).all():
    raise AssertionError("J2 non-default-stream output is non-finite")
if side_error.max().item() > 1e-4:
    raise AssertionError("J2 did not execute correctly on PyTorch's current CUDA stream")


def benchmark(fn, warmup=100, iterations=2000):
    for _ in range(warmup):
        fn()
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    start.record()
    for _ in range(iterations):
        fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000 / iterations


def median_benchmark(fn, repeats=9):
    values = sorted(benchmark(fn) for _ in range(repeats))
    return values[len(values) // 2], values[0], values[-1]


j2 = median_benchmark(lambda: ext.run_k128_n64(
    activation, packed_codes, packed_betas, row_norms, betas, 37, 58))
resident = median_benchmark(lambda: activation @ last_weight.T)
print(f"j2_us median={j2[0]:.6g} min={j2[1]:.6g} max={j2[2]:.6g}")
print(f"resident_us median={resident[0]:.6g} min={resident[1]:.6g} max={resident[2]:.6g}")
print(f"j2_to_resident_ratio={j2[0] / resident[0]:.3f}")

from pathlib import Path
import torch
from torch.utils.cpp_extension import load
from nestquant.core.lattice import nearest_e8

ROOT = Path(__file__).resolve().parents[1]
ext = load(
    name="nestquant_cuda_e8",
    sources=[str(ROOT / "nestquant/runtime/cuda_e8.cpp"), str(ROOT / "nestquant/runtime/cuda_e8.cu")],
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3"],
    verbose=True,
)

x = torch.randn(4096, 8, device="cuda", dtype=torch.float32)
y = ext.e8_nearest(x)
print(f"cuda={torch.cuda.get_device_name()} shape={tuple(y.shape)} finite={bool(torch.isfinite(y).all())}")
print(f"max_abs_input={x.abs().max().item():.6f} max_abs_output={y.abs().max().item():.6f}")
ref, _ = nearest_e8(x)
print(f"reference_max_abs_error={(y - ref).abs().max().item():.6g} exact={bool(torch.equal(y, ref))}")
bad = (y != ref).any(dim=1)
if bad.any():
    i = int(bad.nonzero()[0])
    print("first_mismatch", x[i].cpu().tolist(), y[i].cpu().tolist(), ref[i].cpu().tolist())
    raise AssertionError("CUDA E8 output differs from the reference implementation")

boundary = torch.tensor(
    [[0.25] * 8, [0.5] * 8, [-0.5] * 8, [1.5, 0.5, -0.5, -1.5, 2.5, -2.5, 0.0, 1.0]],
    device="cuda", dtype=torch.float32,
)
by = ext.e8_nearest(boundary)
bref, _ = nearest_e8(boundary)
print(f"boundary_max_abs_error={(by - bref).abs().max().item():.6g} boundary_exact={bool(torch.equal(by, bref))}")
if not torch.equal(by, bref):
    raise AssertionError("CUDA E8 boundary output differs from the reference implementation")

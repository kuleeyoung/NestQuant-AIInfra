#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${NESTQUANT_PYTHON:-python}"
clean_dir="$(mktemp -d /tmp/nestquant-publish-XXXXXX)"
wheel_dir="$(mktemp -d /tmp/nestquant-wheel-XXXXXX)"
sdist_dir="$(mktemp -d /tmp/nestquant-sdist-XXXXXX)"
sdist_wheel_dir="$(mktemp -d /tmp/nestquant-sdist-wheel-XXXXXX)"
install_dir="$(mktemp -d /tmp/nestquant-install-XXXXXX)"

cd "$repo_root"
git -c core.quotepath=false ls-files --cached --others --exclude-standard -z \
  | tar --null -T - -cf - \
  | tar -xf - -C "$clean_dir"

cd "$clean_dir"
"$python_bin" -m pip wheel --no-deps --wheel-dir "$wheel_dir" .
"$python_bin" -m pytest -q
"$python_bin" - "$sdist_dir" <<'PY'
import sys
import setuptools.build_meta

print(setuptools.build_meta.build_sdist(sys.argv[1]))
PY
sdist_path="$(find "$sdist_dir" -maxdepth 1 -name '*.tar.gz' -print -quit)"
"$python_bin" -m pip wheel --no-deps --wheel-dir "$sdist_wheel_dir" "$sdist_path"
wheel_path="$(find "$sdist_wheel_dir" -maxdepth 1 -name '*.whl' -print -quit)"
"$python_bin" - "$wheel_path" <<'PY'
import sys
import zipfile

required = {
    "nestquant/runtime/cuda_e8.cpp",
    "nestquant/runtime/cuda_e8.cu",
    "nestquant/runtime/nqrt4_mma_j2_binding.cpp",
    "nestquant/runtime/nqrt4_mma_j2.cu",
    "nestquant/runtime/nqrt4_cooperative_device.cuh",
    "nestquant/runtime/nqrt4_device.cuh",
}
with zipfile.ZipFile(sys.argv[1]) as archive:
    missing = required - set(archive.namelist())
if missing:
    raise SystemExit(f"wheel is missing runtime sources: {sorted(missing)}")
PY
"$python_bin" -m pip install --no-deps --target "$install_dir" "$wheel_path"
for command in nestquant-audit-repository nestquant-validate-service-benchmark nestquant-promote-service-benchmark nestquant-verify-service-evidence nestquant-validate-hybrid-pareto nestquant-promote-hybrid-pareto; do
  if [[ ! -x "$install_dir/bin/$command" ]]; then
    echo "installed wheel is missing console entry point: $command" >&2
    exit 1
  fi
  PATH="$install_dir/bin:$PATH" PYTHONPATH="$install_dir" "$command" --help >/dev/null
done
(cd /tmp && PYTHONPATH="$install_dir" "$python_bin" - "$install_dir" <<'PY'
from pathlib import Path
import sys
import nestquant.runtime

runtime_dir = Path(nestquant.runtime.__file__).resolve().parent
install_dir = Path(sys.argv[1]).resolve()
if install_dir not in runtime_dir.parents:
    raise SystemExit(f"import did not come from installed wheel target: {runtime_dir}")
required = ("cuda_e8.cpp", "cuda_e8.cu", "nqrt4_mma_j2.cu", "nqrt4_device.cuh")
missing = [name for name in required if not (runtime_dir / name).is_file()]
if missing:
    raise SystemExit(f"installed package is missing runtime sources: {missing}")
print(f"installed runtime sources verified in {runtime_dir}")
PY
)
if [[ "${NESTQUANT_VERIFY_INSTALLED_CUDA:-0}" == "1" ]]; then
  cuda_extension_dir="$(mktemp -d /tmp/nestquant-cuda-extension-XXXXXX)"
  (cd /tmp && TORCH_EXTENSIONS_DIR="$cuda_extension_dir" PYTHONPATH="$install_dir" "$python_bin" - "$install_dir" <<'PY'
from pathlib import Path
import sys
import torch
from nestquant.runtime.cuda_e8 import e8_nearest_cuda
import nestquant.runtime

runtime_dir = Path(nestquant.runtime.__file__).resolve().parent
if Path(sys.argv[1]).resolve() not in runtime_dir.parents:
    raise SystemExit(f"CUDA smoke imported the wrong package: {runtime_dir}")
x = torch.tensor([[0.5] * 8, [-0.5] * 8], dtype=torch.float32, device="cuda")
y = e8_nearest_cuda(x)
if y.shape != x.shape or not torch.isfinite(y).all().item():
    raise SystemExit("installed CUDA E8 smoke produced an invalid result")
print(f"installed CUDA E8 smoke passed from {runtime_dir}")
PY
  )
  echo "cuda_extension_dir=$cuda_extension_dir"
fi

echo "publishable tree smoke passed"
echo "clean_dir=$clean_dir"
echo "wheel_dir=$wheel_dir"
echo "sdist_dir=$sdist_dir"
echo "sdist_wheel_dir=$sdist_wheel_dir"
echo "install_dir=$install_dir"
ls -lh "$wheel_dir" "$sdist_dir" "$sdist_wheel_dir"

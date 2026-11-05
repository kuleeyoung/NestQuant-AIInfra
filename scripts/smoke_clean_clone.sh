#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${NESTQUANT_PYTHON:-python}"
source_dir="$(mktemp -d /tmp/nestquant-clean-source-XXXXXX)"
clone_parent="$(mktemp -d /tmp/nestquant-clean-clone-XXXXXX)"
clone_dir="$clone_parent/repo"
wheel_dir="$(mktemp -d /tmp/nestquant-clean-wheel-XXXXXX)"
install_dir="$(mktemp -d /tmp/nestquant-clean-install-XXXXXX)"
smoke_output_dir="$(mktemp -d /tmp/nestquant-clean-output-XXXXXX)"

# Materialize exactly the files Git considers publishable. This includes the
# current tracked and untracked implementation, but excludes ignored local
# models, raw evidence, caches, credentials, and build products.
cd "$repo_root"
git -c core.quotepath=false ls-files --cached --others --exclude-standard -z \
  | tar --null -T - -cf - \
  | tar -xf - -C "$source_dir"

# Commit only inside the temporary repository. The user's working repository
# and its history are not modified.
cd "$source_dir"
git init -q
git config user.name "NestQuant clean-clone validation"
git config user.email "clean-clone@local.invalid"
git add -A
git commit -q -m "test: clean-clone validation snapshot"
source_commit="$(git rev-parse HEAD)"

# --no-local forces normal Git object transfer rather than local hard links.
git clone -q --no-local "$source_dir" "$clone_dir"
cd "$clone_dir"
if [[ -n "$(git status --porcelain --untracked-files=all)" ]]; then
  echo "fresh clone is unexpectedly dirty" >&2
  git status --short >&2
  exit 1
fi

PYTHONPATH="$clone_dir" "$python_bin" -m nestquant.cli.audit_repository \
  --json-output "$smoke_output_dir/repository_audit.json"
"$python_bin" -m pip wheel --no-deps --wheel-dir "$wheel_dir" .
wheel_path="$(find "$wheel_dir" -maxdepth 1 -name '*.whl' -print -quit)"
if [[ -z "$wheel_path" ]]; then
  echo "wheel build produced no wheel" >&2
  exit 1
fi
"$python_bin" -m pip install --no-deps --target "$install_dir" "$wheel_path"

# Prove that an interpreter outside the clone resolves the installed wheel.
(cd /tmp && PYTHONPATH="$install_dir" "$python_bin" - "$install_dir" <<'PY'
from pathlib import Path
import sys
import nestquant

package_dir = Path(nestquant.__file__).resolve().parent
install_dir = Path(sys.argv[1]).resolve()
if install_dir not in package_dir.parents:
    raise SystemExit(f"import did not come from installed wheel: {package_dir}")
print(f"installed package import passed: {package_dir}")
PY
)

# Exercise the public CPU examples from the clone using only external output.
"$python_bin" scripts/demo_e8.py --device cpu
"$python_bin" scripts/demo_beta_search.py --device cpu
"$python_bin" scripts/run_synthetic_matrix_product.py \
  --device cpu \
  --output "$smoke_output_dir/synthetic_matrix_product.json"

# Verify every installed console script without needing GPU data or raw files.
for command in \
  nestquant-audit-repository \
  nestquant-validate-service-benchmark \
  nestquant-promote-service-benchmark \
  nestquant-verify-service-evidence \
  nestquant-verify-splitk-quality \
  nestquant-validate-hybrid-pareto \
  nestquant-promote-hybrid-pareto; do
  if [[ ! -x "$install_dir/bin/$command" ]]; then
    echo "installed wheel is missing console entry point: $command" >&2
    exit 1
  fi
  PATH="$install_dir/bin:$PATH" PYTHONPATH="$install_dir" "$command" --help >/dev/null
done

if [[ -n "$(git status --porcelain --untracked-files=all)" ]]; then
  echo "clean clone became dirty during validation" >&2
  git status --short >&2
  exit 1
fi

echo "clean-clone validation passed"
echo "source_commit=$source_commit"
echo "source_dir=$source_dir"
echo "clone_dir=$clone_dir"
echo "wheel_path=$wheel_path"
echo "install_dir=$install_dir"
echo "smoke_output_dir=$smoke_output_dir"

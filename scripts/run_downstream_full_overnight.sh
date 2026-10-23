#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="/mnt/d/code/NestQuant-AIInfra"
VENV_ROOT="/home/kulee/code/NestQuant-AIInfra/.venv"
RUN_DIR="$PROJECT_ROOT/results/raw/downstream_full_20261024"
CHECKPOINT="$PROJECT_ROOT/results/raw/llama32_1b_nestquant_w4_shared128_no_lm_head.nqmc"
KV_BETAS="$PROJECT_ROOT/configs/kv_betas_upstream_defaults_q14.json"
LOG_FILE="$RUN_DIR/overnight.log"

mkdir -p "$RUN_DIR"
exec > >(tee -a "$LOG_FILE") 2>&1

export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=0

cd "$PROJECT_ROOT"
source "$VENV_ROOT/bin/activate"

failure_marker() {
  local exit_code=$?
  printf '{"status":"failed","exit_code":%s,"finished_at":"%s"}\n' \
    "$exit_code" "$(date --iso-8601=seconds)" > "$RUN_DIR/overnight_status.json"
  exit "$exit_code"
}
trap failure_marker ERR

is_completed() {
  local output="$1"
  [[ -f "$output" ]] && python - "$output" <<'PY'
import json, sys
with open(sys.argv[1], encoding="utf-8") as handle:
    payload = json.load(handle)
raise SystemExit(0 if payload.get("status") == "completed" else 1)
PY
}

run_method() {
  local method="$1"
  local output="$RUN_DIR/${method}.json"
  if is_completed "$output"; then
    echo "[$(date --iso-8601=seconds)] skip completed method: $method"
    return
  fi
  local command=(
    python scripts/evaluate_downstream_tasks.py
    --method "$method"
    --tasks piqa hellaswag
    --batch-size 4
    --offline-model
    --output "$output"
  )
  if [[ "$method" != "bf16" ]]; then
    command+=(--checkpoint "$CHECKPOINT")
  fi
  if [[ "$method" == "w4a4kv4" ]]; then
    command+=(--kv-beta-config "$KV_BETAS")
  fi
  echo "[$(date --iso-8601=seconds)] start method: $method"
  "${command[@]}"
  echo "[$(date --iso-8601=seconds)] completed method: $method"
}

echo "[$(date --iso-8601=seconds)] overnight downstream evaluation started"
python - <<'PY'
import importlib.metadata, torch
print("torch", torch.__version__)
print("lm_eval", importlib.metadata.version("lm_eval"))
print("cuda", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
PY

run_method bf16
run_method w4
run_method w4a4kv4

python scripts/summarize_downstream_results.py \
  --run-dir "$RUN_DIR" \
  --json-output "$RUN_DIR/summary.json" \
  --markdown-output "$PROJECT_ROOT/results/llama32_1b_downstream_full.md"

printf '{"status":"completed","finished_at":"%s"}\n' \
  "$(date --iso-8601=seconds)" > "$RUN_DIR/overnight_status.json"
echo "[$(date --iso-8601=seconds)] overnight downstream evaluation completed"

#!/bin/bash
#SBATCH --job-name=gcwm-bridge-fast
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=64
#SBATCH --mem=256GB
#SBATCH --output=logs/slurm-%j-%x.out
#SBATCH --error=logs/slurm-%j-%x.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_DIR="${WORK_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
BASE_MODEL="${BASE_MODEL:-/path/to/base/model}"
CONTINUAL_STATS="${CONTINUAL_STATS:-${WORK_DIR}/merged_models/continual_gcwm_stats.json}"
EXPERT_ROOT="${EXPERT_ROOT:-/path/to/expert/full_models}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${WORK_DIR}/bridge_fast_outputs}"
ENV_SETUP="${ENV_SETUP:-}"
CONDA_ENV="${CONDA_ENV:-}"

RANK="${RANK:-32}"
ENERGY_THRESHOLD="${ENERGY_THRESHOLD:-0.95}"
COV_REG="${COV_REG:-1e-6}"
MODEL_DTYPE="${MODEL_DTYPE:-bfloat16}"
LAYER_FILTER="${LAYER_FILTER:-all}"
MAX_LAYERS="${MAX_LAYERS:-64}"
NUM_GPUS="${NUM_GPUS:-8}"

mkdir -p "${WORK_DIR}/logs" "${OUTPUT_ROOT}"

export WORK_DIR BASE_MODEL CONTINUAL_STATS EXPERT_ROOT OUTPUT_ROOT ENV_SETUP CONDA_ENV
export RANK ENERGY_THRESHOLD COV_REG MODEL_DTYPE LAYER_FILTER MAX_LAYERS NUM_GPUS

srun bash <<'INNER_EOF'
set -euo pipefail

if [[ -n "${ENV_SETUP:-}" && -f "${ENV_SETUP}" ]]; then
  source "${ENV_SETUP}"
fi

if [[ -n "${CONDA_ENV:-}" ]]; then
  conda activate "${CONDA_ENV}"
fi

cd "${WORK_DIR}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

resolve_stats_json() {
  python - <<PY
import os
run_dir_or_file = r"${CONTINUAL_STATS}"
if os.path.isfile(run_dir_or_file):
    print(run_dir_or_file)
else:
    p = os.path.join(run_dir_or_file, "continual_gcwm_stats.json")
    if os.path.exists(p):
        print(p)
    else:
        raise FileNotFoundError(f"Could not find continual_gcwm_stats.json under {run_dir_or_file}")
PY
}

STATS_JSON="$(resolve_stats_json)"
TOTAL_STEPS="$(python - <<PY
import json
with open(r"${STATS_JSON}", "r") as f:
    obj = json.load(f)
print(len(obj["steps"]))
PY
)"

mapfile -t MODEL_DIRS < <(
  find "${EXPERT_ROOT}" -mindepth 1 -maxdepth 1 -type d -printf '%f\t%p\n' \
  | sort -V \
  | cut -f2-
)

FILTERED_DIRS=()
FILTERED_LABELS=()
for d in "${MODEL_DIRS[@]}"; do
  base="$(basename "${d}")"
  if [[ -f "${d}/config.json" || -f "${d}/adapter_config.json" || -f "${d}/tokenizer_config.json" ]]; then
    FILTERED_DIRS+=("${d}")
    if [[ "${base}" =~ ^step_[0-9]+_(.+)$ ]]; then
      FILTERED_LABELS+=("${BASH_REMATCH[1]}")
    else
      FILTERED_LABELS+=("${base}")
    fi
  fi
done
MODEL_DIRS=("${FILTERED_DIRS[@]}")
MODEL_LABELS=("${FILTERED_LABELS[@]}")

EXPERT_MODELS_CSV="$(IFS=,; echo "${MODEL_DIRS[*]}")"
TASK_LABELS_CSV="$(IFS=,; echo "${MODEL_LABELS[*]}")"

PART_ROOT="${OUTPUT_ROOT}/parts"
mkdir -p "${PART_ROOT}" "${OUTPUT_ROOT}/merged"

run_worker() {
  local gpu_id="$1"
  for ((step=gpu_id+1; step<=TOTAL_STEPS; step+=NUM_GPUS)); do
    step_tag="$(printf "%02d" "${step}")"
    part_dir="${PART_ROOT}/step_${step_tag}"
    mkdir -p "${part_dir}"
    CUDA_VISIBLE_DEVICES="${gpu_id}" python "${WORK_DIR}/analysis/bridge_analyzer_fast_sharded_gpu.py" \
      --base-model "${BASE_MODEL}" \
      --expert-models "${EXPERT_MODELS_CSV}" \
      --task-labels "${TASK_LABELS_CSV}" \
      --continual-stats "${STATS_JSON}" \
      --output-dir "${part_dir}" \
      --rank "${RANK}" \
      --energy-threshold "${ENERGY_THRESHOLD}" \
      --cov-reg "${COV_REG}" \
      --device cuda:0 \
      --linalg-device cuda:0 \
      --model-dtype "${MODEL_DTYPE}" \
      --layer-filter "${LAYER_FILTER}" \
      --max-layers "${MAX_LAYERS}" \
      --step-start "${step}" \
      --step-end "${step}" \
      > "${OUTPUT_ROOT}/worker_gpu${gpu_id}_step_${step_tag}.log" 2>&1
  done
}

pids=()
for ((gpu=0; gpu<NUM_GPUS; gpu++)); do
  run_worker "${gpu}" &
  pids+=($!)
done

for pid in "${pids[@]}"; do
  wait "${pid}"
done

python "${WORK_DIR}/analysis/merge_bridge_step_outputs.py" \
  --run-dir "${OUTPUT_ROOT}" \
  --output-subdir "merged"
INNER_EOF

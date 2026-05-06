#!/bin/bash
#SBATCH --job-name=gcwm-bridge-grad
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
DATASET_FILE="${DATASET_FILE:-/path/to/mmlupro.parquet}"
TASK_MAP_JSON="${TASK_MAP_JSON:-}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${WORK_DIR}/bridge_grad_outputs}"
ENV_SETUP="${ENV_SETUP:-}"
CONDA_ENV="${CONDA_ENV:-}"

EXAMPLES_PER_TASK="${EXAMPLES_PER_TASK:-16}"
BATCH_SIZE="${BATCH_SIZE:-4}"
MAX_LENGTH="${MAX_LENGTH:-1024}"
MODEL_DTYPE="${MODEL_DTYPE:-bfloat16}"
GRAD_STORE_DTYPE="${GRAD_STORE_DTYPE:-float16}"
LAYER_FILTER="${LAYER_FILTER:-all}"
MAX_LAYERS="${MAX_LAYERS:-64}"
NUM_GPUS="${NUM_GPUS:-8}"

mkdir -p "${WORK_DIR}/logs" "${OUTPUT_ROOT}"

export WORK_DIR BASE_MODEL CONTINUAL_STATS DATASET_FILE TASK_MAP_JSON OUTPUT_ROOT ENV_SETUP CONDA_ENV
export EXAMPLES_PER_TASK BATCH_SIZE MAX_LENGTH MODEL_DTYPE GRAD_STORE_DTYPE LAYER_FILTER MAX_LAYERS NUM_GPUS

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

run_worker() {
  local gpu_id="$1"
  for ((step=gpu_id+1; step<=TOTAL_STEPS; step+=NUM_GPUS)); do
    step_tag="$(printf "%02d" "${step}")"
    part_dir="${OUTPUT_ROOT}/step_${step_tag}"
    mkdir -p "${part_dir}"

    cmd=(
      python "${WORK_DIR}/analysis/bridge_analyzer_grad.py"
      --base-model "${BASE_MODEL}"
      --continual-stats "${STATS_JSON}"
      --dataset-file "${DATASET_FILE}"
      --output-dir "${part_dir}"
      --examples-per-task "${EXAMPLES_PER_TASK}"
      --batch-size "${BATCH_SIZE}"
      --max-length "${MAX_LENGTH}"
      --model-dtype "${MODEL_DTYPE}"
      --grad-store-dtype "${GRAD_STORE_DTYPE}"
      --device cuda:0
      --layer-filter "${LAYER_FILTER}"
      --max-layers "${MAX_LAYERS}"
      --step-start "${step}"
      --step-end "${step}"
    )

    if [[ -n "${TASK_MAP_JSON}" ]]; then
      cmd+=( --task-map-json "${TASK_MAP_JSON}" )
    fi

    CUDA_VISIBLE_DEVICES="${gpu_id}" "${cmd[@]}" > "${OUTPUT_ROOT}/worker_gpu${gpu_id}_step_${step_tag}.log" 2>&1
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
INNER_EOF

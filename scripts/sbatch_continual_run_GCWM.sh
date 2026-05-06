#!/bin/bash

#SBATCH --job-name=continual-gcwm
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=256GB
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --output=logs/%j-%x.out
#SBATCH --error=logs/%j-%x.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_DIR="${WORK_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
SAVE_PATH="${SAVE_PATH:-${WORK_DIR}/merged_models}"
EXPERT_ROOT="${EXPERT_ROOT:-/path/to/expert/full_models}"
BASE_MODEL="${BASE_MODEL:-/path/to/base/model}"
ENV_SETUP="${ENV_SETUP:-}"
CONDA_ENV="${CONDA_ENV:-}"

SCALING_COEF="${SCALING_COEF:-0.2}"
ITER_NUM="${ITER_NUM:-100}"
MEMORY_MODE="${MEMORY_MODE:-all_history}"
MEMORY_SIZE="${MEMORY_SIZE:--1}"
DEVICE="${DEVICE:-cuda}"

CONTAINER_IMAGE="${CONTAINER_IMAGE:-}"
CONTAINER_MOUNTS="${CONTAINER_MOUNTS:-${WORK_DIR}:${WORK_DIR}}"

mkdir -p "${WORK_DIR}/logs"

merge_scripts='
set -euo pipefail
set -x

if [[ -n "${ENV_SETUP:-}" && -f "${ENV_SETUP}" ]]; then
  source "${ENV_SETUP}"
fi

if [[ -n "${CONDA_ENV:-}" ]]; then
  conda activate "${CONDA_ENV}"
fi

cd "${WORK_DIR}"

mapfile -t MODEL_DIRS < <(
  find "${EXPERT_ROOT}" -mindepth 1 -maxdepth 1 -type d -printf "%f\t%p\n" \
  | sort -V \
  | cut -f2-
)

FILTERED_DIRS=()
for d in "${MODEL_DIRS[@]}"; do
  if [[ -f "${d}/config.json" || -f "${d}/adapter_config.json" || -f "${d}/tokenizer_config.json" ]]; then
    FILTERED_DIRS+=("${d}")
  fi
done
MODEL_DIRS=("${FILTERED_DIRS[@]}")
unset FILTERED_DIRS

if [[ ${#MODEL_DIRS[@]} -eq 0 ]]; then
  echo "[ERROR] No expert model directories found under: ${EXPERT_ROOT}"
  exit 1
fi

TASK_ORDER="$(IFS=,; echo "${MODEL_DIRS[*]}")"

python "${WORK_DIR}/merging/main_continual_gcwm.py" \
  --algo GCWM \
  --continual \
  --task-order "${TASK_ORDER}" \
  --memory-mode "${MEMORY_MODE}" \
  --memory-size "${MEMORY_SIZE}" \
  --old-weight 1.0 \
  --new-weight 1.0 \
  --continual-step-coef "${SCALING_COEF}" \
  --step-decay 1.0 \
  --save-each-step \
  --save-stats \
  --scaling-coef "${SCALING_COEF}" \
  --iter-num "${ITER_NUM}" \
  --base-model "${BASE_MODEL}" \
  --save-path "${SAVE_PATH}" \
  --device "${DEVICE}"
'

export WORK_DIR SAVE_PATH EXPERT_ROOT BASE_MODEL ENV_SETUP CONDA_ENV
export SCALING_COEF ITER_NUM MEMORY_MODE MEMORY_SIZE DEVICE

if [[ -n "${CONTAINER_IMAGE}" ]]; then
  srun --container-image="${CONTAINER_IMAGE}" \
    --container-mounts="${CONTAINER_MOUNTS}" \
    --container-remap-root \
    --container-writable \
    bash -c "${merge_scripts}"
else
  srun bash -c "${merge_scripts}"
fi

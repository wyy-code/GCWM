# GCWM

Code for **GCWM: Geometry Conflict: Explaining and Controlling Forgetting in LLM Continual Post-Training**.

This repository is a cleaned release package, which keeps only the GCWM continual merging implementation and the geometry-conflict analysis utilities needed for the paper.

## Contents

- `merging/main_continual_gcwm.py`: entry point for continual GCWM merging.
- `merging/prepare_args_gcwm.py`: GCWM-specific argument parser.
- `merging/merging_methods/gcwm.py`: GCWM implementation.
- `scripts/run_gcwm.sh`: direct shell launcher.
- `scripts/sbatch_continual_run_GCWM.sh`: Slurm launcher.
- `analysis/bridge_analyzer_fast_sharded*.py`: fast model-delta geometry analysis.
- `analysis/bridge_analyzer_grad*.py`: gradient bridge analysis.
- `analysis/analysis_pipeline/`: table and plotting helpers for paper analysis.

## Environment

Activate a Python environment with the required dependencies:

```bash
conda activate xxx
```

If you need to install dependencies manually:

```bash
pip install -r merging/requirements.txt
```

## Continual GCWM Merge

Set the base model, expert root, and output path, then run:

```bash
export WORK_DIR=/path/to/GCWM
export BASE_MODEL=/path/to/base/model
export EXPERT_ROOT=/path/to/expert/full_models
export SAVE_PATH="${WORK_DIR}/merged_models"
export SCALING_COEF=0.2
export ITER_NUM=100
export DEVICE=cuda

bash scripts/run_gcwm.sh
```

For Slurm:

```bash
sbatch scripts/sbatch_continual_run_GCWM.sh
```

You can override paths and hyperparameters through environment variables:

```bash
WORK_DIR=/path/to/GCWM \
BASE_MODEL=/path/to/base/model \
EXPERT_ROOT=/path/to/expert/full_models \
SAVE_PATH=/path/to/GCWM/merged_models \
SCALING_COEF=0.2 \
ITER_NUM=100 \
MEMORY_MODE=all_history \
MEMORY_SIZE=-1 \
DEVICE=cuda \
sbatch scripts/sbatch_continual_run_GCWM.sh
```

GCWM writes `continual_gcwm_stats.json` and optional per-step `gcwm_layer_stats.json` files when `--save-stats` is enabled.

## Geometry-Conflict Analysis

After a GCWM run, use the saved continual stats for bridge analysis:

```bash
export WORK_DIR=/path/to/GCWM
export BASE_MODEL=/path/to/base/model
export EXPERT_ROOT=/path/to/expert/full_models
export CONTINUAL_STATS=/path/to/GCWM/merged_models/.../continual_gcwm_stats.json
export OUTPUT_ROOT=/path/to/GCWM/bridge_fast_outputs/run_name

sbatch analysis/run_bridge_fast_8gpu.slurm.sh
```

For gradient-based bridge analysis, additionally set `DATASET_FILE` and optionally `TASK_MAP_JSON`:

```bash
export DATASET_FILE=/path/to/mmlupro.parquet
export TASK_MAP_JSON=/path/to/task_map.json
sbatch analysis/run_bridge_grad_8gpu.slurm.sh
```

## Notes

- The release package intentionally removes unrelated merge baselines and keeps only the GCWM path.
- Expert models are discovered by scanning one level under `EXPERT_ROOT`, sorted with `sort -V`.
- `MEMORY_SIZE=-1` keeps all previous tasks in the continual memory. Use a positive value to keep only the most recent `N` tasks.


# GCWM Merging

This directory contains the continual GCWM merge entry point and implementation.

Minimal command:

```bash
python merging/main_continual_gcwm.py \
  --algo GCWM \
  --continual \
  --task-order "/path/to/expert_1,/path/to/expert_2" \
  --memory-mode all_history \
  --memory-size -1 \
  --continual-step-coef 0.2 \
  --scaling-coef 0.2 \
  --iter-num 100 \
  --base-model /path/to/base/model \
  --save-path /path/to/output/root \
  --device cuda
```

Main hyperparameters:

- `--scaling-coef`: outer merge coefficient.
- `--continual-step-coef`: per-step continual update coefficient.
- `--memory-mode`: `all_history` or `current_anchor`.
- `--memory-size`: number of recent tasks retained in all-history mode; `-1` keeps all tasks.
- `--gcwm-lr`: inner optimizer learning rate.
- `--rank`: low-rank geometry basis size.
- `--metric-mode`: `barycenter`, `mean`, or `dense_mean`.
- `--gate-mode`: conflict gate, either `conflict` or `none`.

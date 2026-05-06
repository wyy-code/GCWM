#!/usr/bin/env python3
import argparse
from pathlib import Path
from typing import Dict, List

import pandas as pd

from common import (
    ensure_dir,
    infer_overall_col,
    infer_task_columns,
    load_alias_map,
    load_config,
    maybe_load_task_order,
    save_csv,
)


def build_step_downstream_metrics(df: pd.DataFrame, task_order: List[str], task_col_map: Dict[str, str], overall_col: str) -> pd.DataFrame:
    rows = []
    for idx, row in df.iterrows():
        step = idx + 1
        new_task = task_order[idx]
        current_col = task_col_map[new_task]
        old_tasks = task_order[:idx]
        old_cols = [task_col_map[t] for t in old_tasks if t in task_col_map]

        current_score = float(row[current_col]) if current_col in row and pd.notna(row[current_col]) else float("nan")
        old_mean = float(row[old_cols].mean()) if old_cols else float("nan")
        old_min = float(row[old_cols].min()) if old_cols else float("nan")

        if idx == 0 or not old_cols:
            old_drop_mean = float("nan")
            old_drop_min = float("nan")
        else:
            prev = df.iloc[idx - 1]
            drops = row[old_cols] - prev[old_cols]
            old_drop_mean = float(drops.mean())
            old_drop_min = float(drops.min())

        rows.append(
            {
                "step": step,
                "new_task": new_task,
                "overall": float(row[overall_col]),
                "current_task_score": current_score,
                "old_task_mean": old_mean,
                "old_task_min": old_min,
                "old_drop_mean": old_drop_mean,
                "old_drop_min": old_drop_min,
                "plasticity_retention_gap": current_score - old_mean if pd.notna(old_mean) else float("nan"),
            }
        )
    return pd.DataFrame(rows)



def build_pair_downstream_drop(df: pd.DataFrame, task_order: List[str], task_col_map: Dict[str, str]) -> pd.DataFrame:
    rows = []
    for idx in range(1, len(df)):
        step = idx + 1
        new_task = task_order[idx]
        cur = df.iloc[idx]
        prev = df.iloc[idx - 1]
        for old_task in task_order[:idx]:
            col = task_col_map[old_task]
            prev_score = float(prev[col])
            curr_score = float(cur[col])
            rows.append(
                {
                    "step": step,
                    "new_task": new_task,
                    "old_task": old_task,
                    "old_task_prev_score": prev_score,
                    "old_task_curr_score": curr_score,
                    "old_drop": curr_score - prev_score,
                }
            )
    return pd.DataFrame(rows)



def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare downstream metrics from benchmark results.")
    parser.add_argument("--config", required=True, help="Path to YAML/JSON config")
    args = parser.parse_args()

    cfg = load_config(args.config)
    alias_map = load_alias_map(cfg)

    input_cfg = cfg["input"]
    output_root = ensure_dir(cfg["output_root"])
    derived_dir = ensure_dir(output_root / "derived")

    benchmark_path = Path(input_cfg["benchmark"])
    benchmark_cfg = cfg.get("benchmark", {})
    model_col = benchmark_cfg.get("model_col", "model")
    overall_col = benchmark_cfg.get("overall_col", None)

    df = pd.read_csv(benchmark_path)
    overall_col = infer_overall_col(df.columns, preferred=overall_col)
    task_order = maybe_load_task_order(cfg, df, alias_map)
    task_columns = infer_task_columns(df, overall_col=overall_col, model_col=model_col)

    # Align task order to actual columns using exact task names or config overrides.
    task_col_map: Dict[str, str] = {}
    lower_to_col = {str(c).lower(): c for c in task_columns}
    explicit_task_col_map = benchmark_cfg.get("task_column_map", {}) or {}
    explicit_task_cols = benchmark_cfg.get("task_columns", None)
    if explicit_task_cols is not None:
        if len(explicit_task_cols) != len(task_order):
            raise ValueError("benchmark.task_columns length must equal task_order length")
        explicit_task_col_map = {t: c for t, c in zip(task_order, explicit_task_cols)}

    for t in task_order:
        if t in explicit_task_col_map:
            task_col_map[t] = explicit_task_col_map[t]
        elif t in task_columns:
            task_col_map[t] = t
        elif t.lower() in lower_to_col:
            task_col_map[t] = lower_to_col[t.lower()]
        else:
            raise KeyError(f"Task '{t}' not found in benchmark task columns {task_columns}; consider benchmark.task_column_map in config")

    if len(df) != len(task_order):
        raise ValueError(f"Benchmark rows ({len(df)}) != task_order length ({len(task_order)})")

    step_df = build_step_downstream_metrics(df, task_order, task_col_map, overall_col)
    pair_df = build_pair_downstream_drop(df, task_order, task_col_map)

    save_csv(step_df, derived_dir / "step_downstream_metrics.csv")
    save_csv(pair_df, derived_dir / "pair_downstream_drop.csv")

    meta = pd.DataFrame(
        {
            "task": task_order,
            "benchmark_column": [task_col_map[t] for t in task_order],
            "step": list(range(1, len(task_order) + 1)),
        }
    )
    save_csv(meta, derived_dir / "task_order.csv")

    print(f"[OK] step_downstream_metrics -> {derived_dir / 'step_downstream_metrics.csv'}")
    print(f"[OK] pair_downstream_drop -> {derived_dir / 'pair_downstream_drop.csv'}")
    print(f"[OK] task_order -> {derived_dir / 'task_order.csv'}")


if __name__ == "__main__":
    main()

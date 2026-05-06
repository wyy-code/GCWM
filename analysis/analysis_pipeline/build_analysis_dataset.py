#!/usr/bin/env python3
import argparse
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from common import (
    canonical_task_name,
    compute_correlations,
    ensure_dir,
    load_alias_map,
    load_config,
    save_csv,
)


STEP_MECH_COLS_FOR_RANKING = [
    "mean_active_pair_sar",
    "mean_active_pair_geometry_conflict",
    "mean_merged_vs_active_sar",
    "mean_merged_vs_active_geometry_conflict",
    "global_active_geometry_mean",
    "global_with_merged_geometry_mean",
    "mean_active_pair_grad_cos",
    "min_active_pair_grad_cos",
    "neg_conflict_ratio",
    "mean_negative_conflict_cos",
    "merged_total_norm",
    "increment_total_norm",
]

STEP_TARGET_COLS = [
    "overall",
    "current_task_score",
    "old_task_mean",
    "old_task_min",
    "old_drop_mean",
    "old_drop_min",
    "plasticity_retention_gap",
]



def join_step_metrics(step_downstream: pd.DataFrame, step_mechanism: pd.DataFrame) -> pd.DataFrame:
    out = step_downstream.merge(step_mechanism, on=["step", "new_task"], how="left", validate="one_to_one")
    return out.sort_values("step").reset_index(drop=True)



def _select_new_old_pairs(pair_mechanism: pd.DataFrame, task_order_df: pd.DataFrame) -> pd.DataFrame:
    # Only keep active_pair context and pairs where one side is the new task for that step.
    pm = pair_mechanism.copy()
    pm = pm[pm["context"] == "active_pair"].copy()
    step_to_new = dict(zip(task_order_df["step"], task_order_df["task"]))
    pm["new_task_expected"] = pm["step"].map(step_to_new)
    keep_rows = []
    for _, row in pm.iterrows():
        new_task = row["new_task_expected"]
        a = row["obj_a"]
        b = row["obj_b"]
        if a == new_task and b != new_task:
            old_task = b
        elif b == new_task and a != new_task:
            old_task = a
        else:
            continue
        r = row.to_dict()
        r["new_task"] = new_task
        r["old_task"] = old_task
        keep_rows.append(r)
    return pd.DataFrame(keep_rows)



def join_pair_metrics(pair_drop: pd.DataFrame, pair_mechanism: pd.DataFrame, task_order_df: pd.DataFrame) -> pd.DataFrame:
    pm = _select_new_old_pairs(pair_mechanism, task_order_df)
    out = pair_drop.merge(pm, on=["step", "new_task", "old_task"], how="left")
    return out.sort_values(["step", "new_task", "old_task"]).reset_index(drop=True)



def build_step_explanation_ranking(step_join: pd.DataFrame) -> pd.DataFrame:
    x_cols = [c for c in STEP_MECH_COLS_FOR_RANKING if c in step_join.columns]
    y_cols = [c for c in STEP_TARGET_COLS if c in step_join.columns]
    return compute_correlations(step_join, x_cols, y_cols)



def build_top_harmful_pairs(pair_join: pd.DataFrame, top_k: int = 20) -> pd.DataFrame:
    cols = [
        c for c in [
            "step", "new_task", "old_task", "old_drop",
            "sar_sym_mean", "geometry_conflict_mean",
            "grad_cos_mean", "negative_conflict_ratio",
            "sar_sym_max", "geometry_conflict_max", "grad_cos_min",
        ] if c in pair_join.columns
    ]
    out = pair_join.sort_values("old_drop", ascending=True).head(top_k)[cols].reset_index(drop=True)
    return out



def main() -> None:
    parser = argparse.ArgumentParser(description="Build joined analysis datasets.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    output_root = ensure_dir(cfg["output_root"])
    derived_dir = ensure_dir(output_root / "derived")

    step_downstream = pd.read_csv(derived_dir / "step_downstream_metrics.csv")
    pair_drop = pd.read_csv(derived_dir / "pair_downstream_drop.csv")
    task_order_df = pd.read_csv(derived_dir / "task_order.csv")
    step_mech = pd.read_csv(derived_dir / "step_mechanism.csv")
    pair_mech = pd.read_csv(derived_dir / "pair_mechanism.csv")

    step_join = join_step_metrics(step_downstream, step_mech)
    pair_join = join_pair_metrics(pair_drop, pair_mech, task_order_df)
    ranking = build_step_explanation_ranking(step_join)
    top_pairs = build_top_harmful_pairs(pair_join, top_k=int(cfg.get("plot", {}).get("top_k_pairs", 20)))

    save_csv(step_join, derived_dir / "step_mechanism_join.csv")
    save_csv(pair_join, derived_dir / "pair_mechanism_join.csv")
    save_csv(ranking, derived_dir / "step_explanation_ranking.csv")
    save_csv(top_pairs, derived_dir / "top_harmful_pairs.csv")

    print(f"[OK] step_mechanism_join -> {derived_dir / 'step_mechanism_join.csv'}")
    print(f"[OK] pair_mechanism_join -> {derived_dir / 'pair_mechanism_join.csv'}")
    print(f"[OK] step_explanation_ranking -> {derived_dir / 'step_explanation_ranking.csv'}")
    print(f"[OK] top_harmful_pairs -> {derived_dir / 'top_harmful_pairs.csv'}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
import argparse
from pathlib import Path
from typing import Dict, List

import pandas as pd

from common import (
    canonical_task_name,
    ensure_dir,
    load_alias_map,
    load_config,
    pair_key,
    parse_global_geometry_value,
    save_csv,
)


BRIDGE_STEP_COLS = [
    "step",
    "new_task",
    "merged_total_norm",
    "increment_total_norm",
    "mean_active_pair_sar",
    "mean_active_pair_geometry_conflict",
    "mean_merged_vs_active_sar",
    "mean_merged_vs_active_geometry_conflict",
]

GRAD_STEP_COLS = [
    "step",
    "new_task",
    "mean_active_pair_grad_cos",
    "min_active_pair_grad_cos",
    "max_active_pair_grad_cos",
    "neg_conflict_ratio",
    "mean_negative_conflict_cos",
]


def build_step_mechanism(step_bridge: pd.DataFrame, step_grad: pd.DataFrame, alias_map: Dict[str, str]) -> pd.DataFrame:
    sb = step_bridge.copy()
    sg = step_grad.copy()

    sb["new_task"] = sb["new_task"].map(lambda x: canonical_task_name(x, alias_map))
    sg["new_task"] = sg["new_task"].map(lambda x: canonical_task_name(x, alias_map))

    sb["global_active_geometry_mean"] = sb["global_active_geometry"].map(parse_global_geometry_value)
    sb["global_with_merged_geometry_mean"] = sb["global_with_merged_geometry"].map(parse_global_geometry_value)

    keep_bridge = BRIDGE_STEP_COLS + ["global_active_geometry_mean", "global_with_merged_geometry_mean"]
    keep_bridge = [c for c in keep_bridge if c in sb.columns]
    keep_grad = [c for c in GRAD_STEP_COLS if c in sg.columns]

    out = sb[keep_bridge].merge(sg[keep_grad], on=["step", "new_task"], how="outer", validate="one_to_one")
    out = out.sort_values("step").reset_index(drop=True)
    return out



def normalize_pair_bridge(pair_bridge: pd.DataFrame, alias_map: Dict[str, str]) -> pd.DataFrame:
    pb = pair_bridge.copy()
    pb["context_norm"] = pb["context"].astype(str)
    pb["object_i_norm"] = pb["object_i"].map(lambda x: canonical_task_name(x, alias_map))
    pb["object_j_norm"] = pb["object_j"].map(lambda x: canonical_task_name(x, alias_map))
    pb[["obj_a", "obj_b"]] = pd.DataFrame(
        [pair_key(a, b, alias_map=alias_map) for a, b in zip(pb["object_i"], pb["object_j"])],
        index=pb.index,
    )
    return pb



def normalize_pair_grad(pair_grad: pd.DataFrame, alias_map: Dict[str, str]) -> pd.DataFrame:
    pg = pair_grad.copy()
    pg["context_norm"] = pg["context"].astype(str).str.replace("_grad", "", regex=False)
    obj_i_col = "object_i" if "object_i" in pg.columns else "task_i"
    obj_j_col = "object_j" if "object_j" in pg.columns else "task_j"
    pg["object_i_norm"] = pg[obj_i_col].map(lambda x: canonical_task_name(x, alias_map))
    pg["object_j_norm"] = pg[obj_j_col].map(lambda x: canonical_task_name(x, alias_map))
    pg[["obj_a", "obj_b"]] = pd.DataFrame(
        [pair_key(a, b, alias_map=alias_map) for a, b in zip(pg[obj_i_col], pg[obj_j_col])],
        index=pg.index,
    )
    return pg



def build_pair_mechanism(pair_bridge: pd.DataFrame, pair_grad: pd.DataFrame, alias_map: Dict[str, str]) -> pd.DataFrame:
    pb = normalize_pair_bridge(pair_bridge, alias_map)
    pg = normalize_pair_grad(pair_grad, alias_map)

    bridge_keep = [
        "step", "context_norm", "obj_a", "obj_b",
        "object_i_norm", "object_j_norm",
        "sar_sym_mean", "sar_sym_median", "sar_sym_max",
        "geometry_conflict_mean", "geometry_conflict_median", "geometry_conflict_max",
        "n_layers",
    ]
    bridge_keep = [c for c in bridge_keep if c in pb.columns]
    grad_keep = [
        "step", "context_norm", "obj_a", "obj_b",
        "object_i_norm", "object_j_norm",
        "grad_cos_mean", "grad_cos_median", "grad_cos_min", "negative_conflict_ratio", "n_layers",
    ]
    grad_keep = [c for c in grad_keep if c in pg.columns]

    out = pb[bridge_keep].merge(
        pg[grad_keep],
        on=["step", "context_norm", "obj_a", "obj_b"],
        how="outer",
        suffixes=("_bridge", "_grad"),
    )
    out = out.rename(columns={"context_norm": "context"})
    out = out.sort_values(["step", "context", "obj_a", "obj_b"]).reset_index(drop=True)
    return out



def build_family_tables(family_df: pd.DataFrame, alias_map: Dict[str, str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    ff = family_df.copy()
    ff["context_norm"] = ff["context"].astype(str).str.replace("_grad", "", regex=False)
    ff["object_i_norm"] = ff["object_i"].map(lambda x: canonical_task_name(x, alias_map))
    ff["object_j_norm"] = ff["object_j"].map(lambda x: canonical_task_name(x, alias_map))
    ff[["obj_a", "obj_b"]] = pd.DataFrame(
        [pair_key(a, b, alias_map=alias_map) for a, b in zip(ff["object_i"], ff["object_j"])],
        index=ff.index,
    )
    by_step = ff.rename(columns={"context_norm": "context"}).sort_values(["step", "context", "family", "obj_a", "obj_b"]).reset_index(drop=True)

    metrics = [c for c in ["sar_sym_mean", "geometry_conflict_mean", "grad_cos_mean", "negative_conflict_ratio"] if c in ff.columns]
    overall = ff.groupby("family", as_index=False)[metrics].mean(numeric_only=True)
    return by_step, overall



def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize and merge mechanism tables.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    alias_map = load_alias_map(cfg)
    input_cfg = cfg["input"]
    output_root = ensure_dir(cfg["output_root"])
    raw_dir = ensure_dir(output_root / "raw")
    derived_dir = ensure_dir(output_root / "derived")

    # Copy raw inputs into output_root/raw if not already there.
    raw_paths = {
        "step_bridge_summary.csv": Path(input_cfg["step_bridge"]),
        "pair_bridge_agg.csv": Path(input_cfg["pair_bridge"]),
        "family_agg.csv": Path(input_cfg["family"]),
        "top_geometry_conflict_layers.csv": Path(input_cfg["top_geometry"]),
        "step_grad_summary.csv": Path(input_cfg["step_grad"]),
        "pair_grad_agg.csv": Path(input_cfg["pair_grad"]),
        "top_negative_grad_cos_layers.csv": Path(input_cfg["top_negative_grad"]),
        "mmlu_pro_all_results.csv": Path(input_cfg["benchmark"]),
    }
    for name, src in raw_paths.items():
        dst = raw_dir / name
        if src.resolve() != dst.resolve():
            df = pd.read_csv(src)
            save_csv(df, dst)

    step_bridge = pd.read_csv(input_cfg["step_bridge"])
    step_grad = pd.read_csv(input_cfg["step_grad"])
    pair_bridge = pd.read_csv(input_cfg["pair_bridge"])
    pair_grad = pd.read_csv(input_cfg["pair_grad"])
    family_df = pd.read_csv(input_cfg["family"])

    step_mech = build_step_mechanism(step_bridge, step_grad, alias_map)
    pair_mech = build_pair_mechanism(pair_bridge, pair_grad, alias_map)
    family_by_step, family_overall = build_family_tables(family_df, alias_map)

    save_csv(step_mech, derived_dir / "step_mechanism.csv")
    save_csv(pair_mech, derived_dir / "pair_mechanism.csv")
    save_csv(family_by_step, derived_dir / "family_metric_by_step.csv")
    save_csv(family_overall, derived_dir / "family_metric_matrix.csv")

    print(f"[OK] step_mechanism -> {derived_dir / 'step_mechanism.csv'}")
    print(f"[OK] pair_mechanism -> {derived_dir / 'pair_mechanism.csv'}")
    print(f"[OK] family_metric_by_step -> {derived_dir / 'family_metric_by_step.csv'}")
    print(f"[OK] family_metric_matrix -> {derived_dir / 'family_metric_matrix.csv'}")


if __name__ == "__main__":
    main()

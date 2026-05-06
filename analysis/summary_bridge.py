'''
python analysis/summary_bridge.py \
  --bridge-dir /path/to/bridge_fast_outputs/run_name/merged \
  --grad-dir /path/to/bridge_grad_outputs/run_name/merged \
  --out-dir /path/to/analysis_final_results/run_name
'''

import argparse
import json
import pandas as pd
from pathlib import Path


parser = argparse.ArgumentParser()
parser.add_argument("--bridge-dir", required=True)
parser.add_argument("--grad-dir", required=True)
parser.add_argument("--out-dir", required=True)
args = parser.parse_args()

bridge_dir = Path(args.bridge_dir)
grad_dir = Path(args.grad_dir)
out_dir = Path(args.out_dir)
out_dir.mkdir(parents=True, exist_ok=True)


def find_first_existing(df, candidates, required=True):
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        raise KeyError(f"None of candidate columns found: {candidates}. Actual columns: {df.columns.tolist()}")
    return None


def normalize_pair_df(df: pd.DataFrame, kind: str) -> pd.DataFrame:
    df = df.copy()

    step_col = find_first_existing(df, ["step"])
    context_col = find_first_existing(df, ["context"], required=False)
    if context_col is None:
        df["context"] = "unknown"
        context_col = "context"

    obj_i_col = find_first_existing(
        df,
        ["object_i", "obj_i", "name_i", "task_i", "source_i", "left_name", "left_obj"]
    )
    obj_j_col = find_first_existing(
        df,
        ["object_j", "obj_j", "name_j", "task_j", "source_j", "right_name", "right_obj"]
    )
    layer_col = find_first_existing(
        df,
        ["layer_key", "layer", "layer_name"]
    )

    rename_map = {
        step_col: "step",
        context_col: "context",
        obj_i_col: "object_i",
        obj_j_col: "object_j",
        layer_col: "layer_key",
    }
    df = df.rename(columns=rename_map)

    # bridge-specific metrics
    if kind == "bridge":
        sar_col = find_first_existing(df, ["sar_sym", "sar", "sar_symmetric"], required=False)
        gc_col = find_first_existing(df, ["geometry_conflict", "geom_conflict", "conflict"], required=False)

        if sar_col is not None and sar_col != "sar_sym":
            df = df.rename(columns={sar_col: "sar_sym"})
        if gc_col is not None and gc_col != "geometry_conflict":
            df = df.rename(columns={gc_col: "geometry_conflict"})

    # grad-specific metrics
    if kind == "grad":
        grad_cos_col = find_first_existing(df, ["grad_cos", "gradient_cosine", "cosine"], required=False)
        neg_col = find_first_existing(df, ["negative_conflict", "neg_conflict", "is_negative_conflict"], required=False)

        if grad_cos_col is not None and grad_cos_col != "grad_cos":
            df = df.rename(columns={grad_cos_col: "grad_cos"})
        if neg_col is not None and neg_col != "negative_conflict":
            df = df.rename(columns={neg_col: "negative_conflict"})

        if "negative_conflict" in df.columns:
            df["negative_conflict"] = df["negative_conflict"].astype(float)

    return df


# -------------------------
# 1) step-level summaries
# -------------------------
step_bridge = pd.read_json(bridge_dir / "step_metrics.jsonl", lines=True)
step_bridge.to_csv(out_dir / "step_bridge_summary.csv", index=False)

step_grad = pd.read_json(grad_dir / "step_grad_metrics.jsonl", lines=True)
step_grad.to_csv(out_dir / "step_grad_summary.csv", index=False)

# -------------------------
# 2) pairwise bridge agg
# -------------------------
pair_bridge = pd.read_json(bridge_dir / "pairwise_layer_metrics.jsonl", lines=True)
pair_bridge = normalize_pair_df(pair_bridge, kind="bridge")

bridge_group_cols = ["step", "context", "object_i", "object_j"]

bridge_aggs = {}
if "sar_sym" in pair_bridge.columns:
    bridge_aggs.update({
        "sar_sym_mean": ("sar_sym", "mean"),
        "sar_sym_median": ("sar_sym", "median"),
        "sar_sym_max": ("sar_sym", "max"),
    })
if "geometry_conflict" in pair_bridge.columns:
    bridge_aggs.update({
        "geometry_conflict_mean": ("geometry_conflict", "mean"),
        "geometry_conflict_median": ("geometry_conflict", "median"),
        "geometry_conflict_max": ("geometry_conflict", "max"),
    })
bridge_aggs["n_layers"] = ("layer_key", "count")

pair_bridge_agg = (
    pair_bridge
    .groupby(bridge_group_cols, dropna=False)
    .agg(**bridge_aggs)
    .reset_index()
)
pair_bridge_agg.to_csv(out_dir / "pair_bridge_agg.csv", index=False)

# -------------------------
# 3) pairwise grad agg
# -------------------------
pair_grad = pd.read_json(grad_dir / "pairwise_layer_grad_metrics.jsonl", lines=True)
pair_grad = normalize_pair_df(pair_grad, kind="grad")

grad_group_cols = ["step", "context", "object_i", "object_j"]

grad_aggs = {}
if "grad_cos" in pair_grad.columns:
    grad_aggs.update({
        "grad_cos_mean": ("grad_cos", "mean"),
        "grad_cos_median": ("grad_cos", "median"),
        "grad_cos_min": ("grad_cos", "min"),
    })
if "negative_conflict" in pair_grad.columns:
    grad_aggs.update({
        "negative_conflict_ratio": ("negative_conflict", "mean"),
    })
grad_aggs["n_layers"] = ("layer_key", "count")

pair_grad_agg = (
    pair_grad
    .groupby(grad_group_cols, dropna=False)
    .agg(**grad_aggs)
    .reset_index()
)
pair_grad_agg.to_csv(out_dir / "pair_grad_agg.csv", index=False)

# -------------------------
# 4) layer family agg
# -------------------------
def layer_family(layer_key: str) -> str:
    if "q_proj.weight" in layer_key:
        return "q_proj"
    if "k_proj.weight" in layer_key:
        return "k_proj"
    if "v_proj.weight" in layer_key:
        return "v_proj"
    if "o_proj.weight" in layer_key:
        return "o_proj"
    if "gate_proj.weight" in layer_key:
        return "gate_proj"
    if "up_proj.weight" in layer_key:
        return "up_proj"
    if "down_proj.weight" in layer_key:
        return "down_proj"
    return "other"

pair_bridge["family"] = pair_bridge["layer_key"].map(layer_family)
pair_grad["family"] = pair_grad["layer_key"].map(layer_family)

bridge_family_aggs = {}
if "sar_sym" in pair_bridge.columns:
    bridge_family_aggs["sar_sym_mean"] = ("sar_sym", "mean")
if "geometry_conflict" in pair_bridge.columns:
    bridge_family_aggs["geometry_conflict_mean"] = ("geometry_conflict", "mean")
bridge_family_aggs["n_layers_bridge"] = ("layer_key", "count")

grad_family_aggs = {}
if "grad_cos" in pair_grad.columns:
    grad_family_aggs["grad_cos_mean"] = ("grad_cos", "mean")
if "negative_conflict" in pair_grad.columns:
    grad_family_aggs["negative_conflict_ratio"] = ("negative_conflict", "mean")
grad_family_aggs["n_layers_grad"] = ("layer_key", "count")

bridge_family_agg = (
    pair_bridge
    .groupby(["step", "context", "object_i", "object_j", "family"], dropna=False)
    .agg(**bridge_family_aggs)
    .reset_index()
)

grad_family_agg = (
    pair_grad
    .groupby(["step", "context", "object_i", "object_j", "family"], dropna=False)
    .agg(**grad_family_aggs)
    .reset_index()
)

family_agg = bridge_family_agg.merge(
    grad_family_agg,
    on=["step", "context", "object_i", "object_j", "family"],
    how="outer"
)
family_agg.to_csv(out_dir / "family_agg.csv", index=False)

# -------------------------
# 5) top-k extremes
# -------------------------
if "geometry_conflict" in pair_bridge.columns:
    top_gc = pair_bridge.sort_values("geometry_conflict", ascending=False).head(200)
    top_gc.to_csv(out_dir / "top_geometry_conflict_layers.csv", index=False)

if "grad_cos" in pair_grad.columns:
    top_neg_grad = pair_grad.sort_values("grad_cos", ascending=True).head(200)
    top_neg_grad.to_csv(out_dir / "top_negative_grad_cos_layers.csv", index=False)

print("Saved CSVs to:", out_dir)

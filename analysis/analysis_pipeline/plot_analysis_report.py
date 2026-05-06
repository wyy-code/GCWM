#!/usr/bin/env python3
import argparse
from pathlib import Path
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import dataframe_to_heatmap_matrix, ensure_dir, load_config


plt.rcParams.update({
    "figure.dpi": 140,
    "savefig.dpi": 180,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
})


FAMILY_ORDER = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


def _save(fig, path: Path):
    ensure_dir(path.parent)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)



def _plot_line(ax, df, x_col, y_cols, title, ylabel=None):
    for col in y_cols:
        if col in df.columns:
            ax.plot(df[x_col], df[col], marker="o", label=col)
    ax.set_title(title)
    ax.set_xlabel(x_col)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)



def plot_step_master(step_join: pd.DataFrame, save_path: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.ravel()

    _plot_line(axes[0], step_join, "step", ["overall", "current_task_score", "old_task_mean"], "Benchmark dynamics")
    _plot_line(axes[1], step_join, "step", ["old_drop_mean", "old_drop_min"], "Forgetting summary")
    _plot_line(axes[2], step_join, "step", ["mean_active_pair_sar", "mean_merged_vs_active_sar"], "SAR")
    _plot_line(axes[3], step_join, "step", ["mean_active_pair_geometry_conflict", "mean_merged_vs_active_geometry_conflict"], "Geometry conflict")
    _plot_line(axes[4], step_join, "step", ["mean_active_pair_grad_cos", "neg_conflict_ratio"], "Gradient compatibility")
    _plot_line(axes[5], step_join, "step", ["increment_total_norm", "global_with_merged_geometry_mean"], "Drift vs global geometry")

    _save(fig, save_path)



def plot_pair_scatter_matrix(pair_join: pd.DataFrame, save_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    specs = [
        ("sar_sym_mean", "Selective forgetting vs SAR"),
        ("geometry_conflict_mean", "Selective forgetting vs geometry"),
        ("grad_cos_mean", "Selective forgetting vs grad cosine"),
    ]
    color_vals = pair_join["step"] if "step" in pair_join.columns else np.arange(len(pair_join))
    for ax, (xcol, title) in zip(axes, specs):
        sub = pair_join[[xcol, "old_drop", "step"]].dropna()
        sc = ax.scatter(sub[xcol], sub["old_drop"], c=sub["step"], cmap="viridis", alpha=0.8, s=22)
        ax.axhline(0.0, color="black", linestyle="--", linewidth=1)
        ax.set_title(title)
        ax.set_xlabel(xcol)
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("old_drop")
    cbar = fig.colorbar(sc, ax=axes.tolist(), shrink=0.9)
    cbar.set_label("step")
    _save(fig, save_path)



def _draw_heatmap(ax, mat: pd.DataFrame, title: str, cmap: str = "viridis", center: Optional[float] = None, annotate: bool = False):
    vals = mat.values.astype(float)
    if center is None:
        im = ax.imshow(vals, aspect="auto", cmap=cmap)
    else:
        vmax = np.nanmax(np.abs(vals - center)) if np.isfinite(vals).any() else 1.0
        im = ax.imshow(vals, aspect="auto", cmap=cmap, vmin=center - vmax, vmax=center + vmax)
    ax.set_title(title)
    ax.set_xticks(range(len(mat.columns)))
    ax.set_xticklabels(mat.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(mat.index)))
    ax.set_yticklabels(mat.index)
    if annotate:
        for i in range(vals.shape[0]):
            for j in range(vals.shape[1]):
                v = vals[i, j]
                if np.isfinite(v):
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6, color="white")
    return im



def plot_taskpair_heatmaps(pair_join: pd.DataFrame, save_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    metrics = [
        ("sar_sym_mean", "Pairwise SAR"),
        ("geometry_conflict_mean", "Pairwise geometry conflict"),
        ("grad_cos_mean", "Pairwise grad cosine"),
    ]
    for ax, (metric, title) in zip(axes, metrics):
        sub = pair_join[["old_task", "new_task", metric]].dropna()
        mat = dataframe_to_heatmap_matrix(sub, "old_task", "new_task", metric)
        center = 0.0 if metric == "grad_cos_mean" else None
        im = _draw_heatmap(ax, mat, title, cmap="viridis" if center is None else "coolwarm", center=center)
        fig.colorbar(im, ax=ax, shrink=0.85)
    _save(fig, save_path)



def plot_family_metric_heatmap(family_by_step: pd.DataFrame, save_path: Path, focus_steps: Optional[Sequence[int]] = None) -> None:
    df = family_by_step.copy()
    if focus_steps is not None:
        df = df[df["step"].isin(list(focus_steps))].copy()
    # Prefer active_pair rows to align with pairwise task interaction
    if "context" in df.columns:
        df = df[df["context"] == "active_pair"].copy()
    metrics = [m for m in ["sar_sym_mean", "geometry_conflict_mean", "grad_cos_mean", "negative_conflict_ratio"] if m in df.columns]
    agg = df.groupby("family", as_index=False)[metrics].mean(numeric_only=True)
    agg = agg.set_index("family").reindex([f for f in FAMILY_ORDER if f in agg.index])
    mat = agg.T

    fig, ax = plt.subplots(figsize=(9, 4.5))
    im = _draw_heatmap(ax, mat, "Family × metric", cmap="coolwarm", center=0.0)
    fig.colorbar(im, ax=ax, shrink=0.9)
    _save(fig, save_path)



def plot_global_vs_local_geometry(step_join: pd.DataFrame, save_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 4.5))
    _plot_line(
        ax,
        step_join,
        "step",
        [
            "mean_active_pair_geometry_conflict",
            "global_active_geometry_mean",
            "mean_merged_vs_active_geometry_conflict",
            "global_with_merged_geometry_mean",
        ],
        "Global vs local geometry",
    )
    _save(fig, save_path)



def plot_top_layer_cases(top_geom: pd.DataFrame, top_grad: pd.DataFrame, save_path: Path, top_k: int = 12) -> None:
    geom = top_geom.nlargest(top_k, "geometry_conflict").copy() if "geometry_conflict" in top_geom.columns else top_geom.head(top_k).copy()
    grad = top_grad.nsmallest(top_k, "grad_cos").copy() if "grad_cos" in top_grad.columns else top_grad.head(top_k).copy()

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    axes[0].scatter(geom["geometry_conflict"], range(len(geom)), c=pd.Categorical(geom["family"]).codes, cmap="tab10", s=40)
    axes[0].set_yticks(range(len(geom)))
    axes[0].set_yticklabels(geom["layer_key"])
    axes[0].invert_yaxis()
    axes[0].set_title("Top geometry-conflict layers")
    axes[0].set_xlabel("geometry_conflict")
    axes[0].grid(True, alpha=0.3)

    axes[1].scatter(grad["grad_cos"], range(len(grad)), c=pd.Categorical(grad["family"]).codes, cmap="tab10", s=40)
    axes[1].set_yticks(range(len(grad)))
    axes[1].set_yticklabels(grad["layer_key"])
    axes[1].invert_yaxis()
    axes[1].set_title("Top negative-grad layers")
    axes[1].set_xlabel("grad_cos")
    axes[1].grid(True, alpha=0.3)

    _save(fig, save_path)



def plot_step_explanation_heatmap(ranking_df: pd.DataFrame, save_path: Path) -> None:
    mat = ranking_df.pivot_table(index="metric", columns="target", values="spearman", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(8, max(4, 0.3 * len(mat))))
    im = _draw_heatmap(ax, mat, "Step-level explanation ranking (Spearman)", cmap="coolwarm", center=0.0)
    fig.colorbar(im, ax=ax, shrink=0.9)
    _save(fig, save_path)



def plot_top_harmful_pairs(top_pairs_df: pd.DataFrame, save_path: Path) -> None:
    df = top_pairs_df.copy().head(20)
    labels = [f"{r['new_task']} → {r['old_task']}" for _, r in df.iterrows()]
    fig, ax = plt.subplots(figsize=(10, max(4, 0.35 * len(df))))
    colors = df.get("geometry_conflict_mean", pd.Series(np.zeros(len(df))))
    sc = ax.barh(range(len(df)), df["old_drop"], color=plt.cm.viridis((colors - colors.min()) / (colors.max() - colors.min() + 1e-12)))
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_title("Top harmful new-old task pairs")
    ax.set_xlabel("old_drop")
    ax.grid(True, axis="x", alpha=0.3)
    _save(fig, save_path)



def main() -> None:
    parser = argparse.ArgumentParser(description="Plot analysis report figures.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    output_root = ensure_dir(cfg["output_root"])
    derived_dir = output_root / "derived"
    raw_dir = output_root / "raw"
    figs_dir = ensure_dir(output_root / "figs")

    step_join = pd.read_csv(derived_dir / "step_mechanism_join.csv")
    pair_join = pd.read_csv(derived_dir / "pair_mechanism_join.csv")
    family_by_step = pd.read_csv(derived_dir / "family_metric_by_step.csv")
    ranking_df = pd.read_csv(derived_dir / "step_explanation_ranking.csv")
    top_pairs_df = pd.read_csv(derived_dir / "top_harmful_pairs.csv")
    top_geom = pd.read_csv(raw_dir / "top_geometry_conflict_layers.csv")
    top_grad = pd.read_csv(raw_dir / "top_negative_grad_cos_layers.csv")

    top_k_layers = int(cfg.get("plot", {}).get("top_k_layers", 12))

    plot_step_master(step_join, figs_dir / "fig1_step_master.png")
    plot_pair_scatter_matrix(pair_join, figs_dir / "fig2_selective_forgetting_vs_compatibility.png")
    plot_taskpair_heatmaps(pair_join, figs_dir / "fig3_taskpair_heatmaps.png")
    plot_family_metric_heatmap(family_by_step, figs_dir / "fig4_family_metric_heatmap.png")
    plot_global_vs_local_geometry(step_join, figs_dir / "fig5_global_vs_local_geometry.png")
    plot_top_layer_cases(top_geom, top_grad, figs_dir / "fig6_top_layer_cases.png", top_k=top_k_layers)
    plot_step_explanation_heatmap(ranking_df, figs_dir / "fig7_step_explanation_heatmap.png")
    plot_top_harmful_pairs(top_pairs_df, figs_dir / "fig8_top_harmful_pairs.png")

    print(f"[OK] figures saved under {figs_dir}")


if __name__ == "__main__":
    main()

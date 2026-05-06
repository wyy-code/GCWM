import ast
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd


DEFAULT_OVERALL_COLS = [
    "overall_accuracy",
    "overall",
    "avg",
    "average",
]


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def load_config(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".json"}:
        return json.loads(text)
    try:
        import yaml  # type: ignore
        return yaml.safe_load(text)
    except Exception:
        # very small fallback: treat as json
        return json.loads(text)



def save_csv(df: pd.DataFrame, path: str | Path) -> None:
    ensure_dir(Path(path).parent)
    df.to_csv(path, index=False)



def canonical_task_name(name: Any, alias_map: Optional[Dict[str, str]] = None) -> str:
    s = str(name).strip()
    s = s.replace("\\", "/")
    s = os.path.basename(s)
    s = s.strip()

    patterns = [
        r"^step_(\d+)_(.+)$",
        r"^(\d+)_([A-Za-z0-9_.\-]+)$",
        r"^merged_step_(\d+)$",
    ]
    for pat in patterns:
        m = re.match(pat, s)
        if m:
            if pat == r"^merged_step_(\d+)$":
                return f"merged_step_{int(m.group(1))}"
            s = m.group(2)
            break

    s = s.replace(" ", "_")
    s = s.replace("/", "_")
    s = re.sub(r"[^A-Za-z0-9_.\-]+", "_", s)
    s = s.strip("_")
    s_low = s.lower()
    if alias_map:
        if s in alias_map:
            return alias_map[s]
        if s_low in alias_map:
            return alias_map[s_low]
    return s



def infer_task_order_from_benchmark(df: pd.DataFrame, model_col: str = "model", alias_map: Optional[Dict[str, str]] = None) -> List[str]:
    if model_col not in df.columns:
        raise KeyError(f"Benchmark results missing model column: {model_col}")
    order = []
    for m in df[model_col].tolist():
        order.append(canonical_task_name(m, alias_map=alias_map))
    return order



def infer_overall_col(columns: Sequence[str], preferred: Optional[str] = None) -> str:
    if preferred and preferred in columns:
        return preferred
    for c in DEFAULT_OVERALL_COLS:
        if c in columns:
            return c
    raise KeyError(f"Could not infer overall column from {list(columns)}")



def infer_task_columns(df: pd.DataFrame, overall_col: str, model_col: str = "model") -> List[str]:
    skip = {
        model_col,
        overall_col,
        "step",
        "checkpoint",
        "ckpt",
        "save_dir",
        "path",
    }
    out = []
    for c in df.columns:
        if c in skip:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            out.append(c)
    return out



def parse_global_geometry_value(x: Any) -> float | None:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return None
    if isinstance(x, (int, float)):
        return float(x)
    if isinstance(x, dict):
        v = x.get("mean_pairwise_geometry_conflict", None)
        return None if v is None else float(v)
    if isinstance(x, str):
        s = x.strip()
        if not s:
            return None
        try:
            obj = json.loads(s)
        except Exception:
            try:
                obj = ast.literal_eval(s)
            except Exception:
                obj = None
        if isinstance(obj, dict):
            v = obj.get("mean_pairwise_geometry_conflict", None)
            return None if v is None else float(v)
        try:
            return float(s)
        except Exception:
            return None
    return None



def safe_literal_eval(x: Any) -> Any:
    if x is None:
        return None
    if isinstance(x, (list, tuple, dict)):
        return x
    if isinstance(x, str):
        s = x.strip()
        if not s:
            return None
        try:
            return ast.literal_eval(s)
        except Exception:
            return x
    return x



def pair_key(a: Any, b: Any, alias_map: Optional[Dict[str, str]] = None) -> Tuple[str, str]:
    aa = canonical_task_name(a, alias_map=alias_map)
    bb = canonical_task_name(b, alias_map=alias_map)
    return tuple(sorted([aa, bb]))



def extract_context_family(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "context" in out.columns:
        out["context_norm"] = out["context"].astype(str).str.replace("_grad", "", regex=False)
    return out



def standardize_task_order(task_order: Optional[Sequence[str]], alias_map: Optional[Dict[str, str]] = None) -> List[str]:
    if not task_order:
        return []
    return [canonical_task_name(x, alias_map=alias_map) for x in task_order]



def load_alias_map(config: Dict[str, Any]) -> Dict[str, str]:
    alias_map = config.get("task_alias_map", {}) or {}
    out: Dict[str, str] = {}
    for k, v in alias_map.items():
        out[str(k)] = str(v)
        out[str(k).lower()] = str(v)
    return out



def maybe_load_task_order(config: Dict[str, Any], benchmark_df: pd.DataFrame, alias_map: Dict[str, str]) -> List[str]:
    task_order = config.get("task_order", None)
    if task_order:
        return standardize_task_order(task_order, alias_map=alias_map)
    benchmark_cfg = config.get("benchmark", {})
    model_col = benchmark_cfg.get("model_col", "model")
    return infer_task_order_from_benchmark(benchmark_df, model_col=model_col, alias_map=alias_map)



def dataframe_to_heatmap_matrix(
    df: pd.DataFrame,
    row_col: str,
    col_col: str,
    val_col: str,
    row_order: Optional[Sequence[str]] = None,
    col_order: Optional[Sequence[str]] = None,
    agg: str = "mean",
) -> pd.DataFrame:
    pt = df.pivot_table(index=row_col, columns=col_col, values=val_col, aggfunc=agg)
    if row_order is not None:
        rows = [r for r in row_order if r in pt.index] + [r for r in pt.index if r not in set(row_order)]
        pt = pt.reindex(rows)
    if col_order is not None:
        cols = [c for c in col_order if c in pt.columns] + [c for c in pt.columns if c not in set(col_order)]
        pt = pt.reindex(columns=cols)
    return pt



def compute_correlations(df: pd.DataFrame, x_cols: Sequence[str], y_cols: Sequence[str]) -> pd.DataFrame:
    records = []
    for x in x_cols:
        if x not in df.columns:
            continue
        for y in y_cols:
            if y not in df.columns:
                continue
            sub = df[[x, y]].dropna()
            if len(sub) < 3:
                pearson = float("nan")
                spearman = float("nan")
            else:
                pearson = sub[x].corr(sub[y], method="pearson")
                spearman = sub[x].corr(sub[y], method="spearman")
            records.append(
                {
                    "metric": x,
                    "target": y,
                    "pearson": pearson,
                    "spearman": spearman,
                    "abs_spearman": abs(spearman) if pd.notna(spearman) else float("nan"),
                    "n": len(sub),
                }
            )
    out = pd.DataFrame(records)
    if len(out) > 0:
        out = out.sort_values(["target", "abs_spearman"], ascending=[True, False]).reset_index(drop=True)
    return out

'''
CUDA_VISIBLE_DEVICES=0 python analysis/bridge_analyzer_fast_sharded.py \
  --base-model ... \
  --expert-models ... \
  --continual-stats ... \
  --output-dir /path/to/bridge_part0 \
  --rank 32 \
  --energy-threshold 0.95 \
  --cov-reg 1e-6 \
  --device cuda:0 \
  --step-start 1 \
  --step-end 5
python merge_bridge_outputs.py \
  --part-dirs /path/to/bridge_part0,/path/to/bridge_part1,/path/to/bridge_part2,/path/to/bridge_part3 \
  --output-dir /path/to/bridge_analysis_merged
'''
import os
import re
import gc
import json
import ctypes
import pathlib
import argparse
import traceback
from collections import OrderedDict
from itertools import combinations

import torch
from transformers import AutoModelForCausalLM
from tqdm import tqdm


TARGET_KEYWORDS = (
    "q_proj.weight",
    "k_proj.weight",
    "v_proj.weight",
    "o_proj.weight",
    "gate_proj.weight",
    "up_proj.weight",
    "down_proj.weight",
)


def _safe_name(s: str) -> str:
    s = str(s)
    s = s.replace("/", "__").replace("\\", "__")
    s = re.sub(r"[^A-Za-z0-9_.\-]+", "_", s)
    return s[:220]


def _parse_csv(s):
    if s is None:
        return None
    parts = [x.strip() for x in s.split(",")]
    return [x for x in parts if len(x) > 0]


def _canonical_task_label(s: str) -> str:
    base = os.path.basename(str(s).rstrip("/"))
    m = re.match(r"^step_\d+_(.+)$", base)
    if m:
        return m.group(1)
    return base


def _resolve_device(device: str | None, what: str = "device") -> torch.device:
    if device is None:
        req = "cuda:0" if torch.cuda.is_available() else "cpu"
    else:
        req = str(device)
    if req.startswith("cuda") and not torch.cuda.is_available():
        print(f"[WARN] Requested {what}={req} but torch.cuda.is_available() is False; fallback to cpu.")
        return torch.device("cpu")
    return torch.device(req)

def _preload_torch_cuda_libraries():
    """
    Explicitly preload PyTorch CUDA shared libraries to avoid lazy dlopen failures
    (especially libtorch_cuda_linalg.so) in newly spawned worker processes.

    Safe to call multiple times.
    """
    if not torch.cuda.is_available():
        return

    torch_dir = pathlib.Path(torch.__file__).resolve().parent
    lib_dir = torch_dir / "lib"

    # Preload in dependency-friendly order.
    candidates = [
        "libc10_cuda.so",
        "libtorch_cuda.so",
        "libtorch_cuda_linalg.so",
    ]

    for name in candidates:
        lib_path = lib_dir / name
        if lib_path.exists():
            try:
                ctypes.CDLL(str(lib_path), mode=ctypes.RTLD_GLOBAL)
            except OSError as e:
                raise RuntimeError(
                    f"Failed to preload {lib_path}. "
                    f"This usually means LD_LIBRARY_PATH / container runtime is incomplete. "
                    f"Original error: {e}"
                ) from e


def _cuda_linalg_smoke_test(device: torch.device):
    """
    Trigger CUDA linalg early so the process fails fast here instead of
    later inside torch.linalg.eigh during step processing.
    """
    if device.type != "cuda":
        return
    x = torch.randn(32, 32, device=device, dtype=torch.float32)
    y = x @ x.T
    _ = torch.linalg.eigh(y)
    del x, y, _
    torch.cuda.synchronize(device)

def _select_trainable_state_dict(model):
    sd = model.state_dict()
    out = OrderedDict()
    for k, v in sd.items():
        lk = k.lower()
        if "embed" in lk or "lm_head" in lk:
            continue
        if not torch.is_floating_point(v):
            continue
        out[k] = v.detach().cpu()
    return out


def _is_target_key(key, tensor):
    if tensor is None or tensor.ndim != 2:
        return False
    return any(x in key for x in TARGET_KEYWORDS)


def _load_selected_state_dict(model_path, dtype="bfloat16"):
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=getattr(torch, dtype),
        low_cpu_mem_usage=True,
    )
    sd = _select_trainable_state_dict(model)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return sd


def _build_delta_dict(model_path, base_sd):
    model_sd = _load_selected_state_dict(model_path)
    delta = OrderedDict()
    for k, v in base_sd.items():
        if k in model_sd:
            delta[k] = model_sd[k] - v
    return delta


def _infer_stats_path(run_dir_or_file):
    if os.path.isfile(run_dir_or_file):
        return run_dir_or_file
    candidates = [
        "continual_gcwm_stats.json",
    ]
    for c in candidates:
        p = os.path.join(run_dir_or_file, c)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"Could not find continual stats json under {run_dir_or_file}")


def _load_stats(stats_path):
    with open(stats_path, "r") as f:
        return json.load(f)


def _build_label_to_path(expert_models, task_labels=None):
    if task_labels is None:
        task_labels = [os.path.basename(x.rstrip("/")) for x in expert_models]
    if len(task_labels) != len(expert_models):
        raise ValueError("task_labels length must equal expert_models length")

    label_to_path = {}
    for path, label in zip(expert_models, task_labels):
        aliases = set()
        aliases.add(str(label))
        aliases.add(_canonical_task_label(label))
        base = os.path.basename(path.rstrip("/"))
        aliases.add(base)
        aliases.add(_canonical_task_label(base))
        for alias in aliases:
            if alias in label_to_path and label_to_path[alias] != path:
                continue
            label_to_path[alias] = path
    return label_to_path


def _resolve_active_object(name, step_idx, steps_meta, label_to_path):
    if name in label_to_path:
        return ("expert", label_to_path[name])

    canon = _canonical_task_label(name)
    if canon in label_to_path:
        return ("expert", label_to_path[canon])

    m = re.match(r"merged_until_(\d+)", name)
    if m:
        idx = int(m.group(1))
        if idx < 0 or idx >= len(steps_meta):
            raise ValueError(f"Invalid merged anchor index {idx} from active name {name}")
        prev_save_dir = steps_meta[idx].get("save_dir", None)
        if prev_save_dir is None:
            raise ValueError(f"Step {idx+1} has no save_dir; cannot resolve merged anchor")
        return ("merged_anchor", prev_save_dir)

    raise KeyError(f"Cannot resolve active object name: {name}")


def _append_jsonl(path, records):
    if not records:
        return
    with open(path, "a", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


class DeltaStore:
    def __init__(self, base_model_path, cache_dir, model_dtype="bfloat16"):
        self.base_model_path = base_model_path
        self.cache_dir = cache_dir
        self.model_dtype = model_dtype
        os.makedirs(self.cache_dir, exist_ok=True)
        self.base_sd = _load_selected_state_dict(base_model_path, dtype=model_dtype)

    def _cache_file(self, key):
        return os.path.join(self.cache_dir, f"{_safe_name(key)}.pt")

    def load_or_build_delta(self, key, model_path):
        cache_f = self._cache_file(key)
        if os.path.exists(cache_f):
            return torch.load(cache_f, map_location="cpu")
        delta = _build_delta_dict(model_path, self.base_sd)
        torch.save(delta, cache_f)
        return delta


def _sym(A):
    return 0.5 * (A + A.T)


def _eye(n, like, scale=1.0):
    return scale * torch.eye(n, device=like.device, dtype=like.dtype)


def _matrix_sqrt_psd(A, neg_tol=1e-12, zero_rel_tol=1e-10):
    """
    PSD matrix square root in float64.

    neg_tol:
        tolerate tiny negative eigenvalues from numerical error;
        values in [-neg_tol, 0) are treated as 0.

    zero_rel_tol:
        optional relative threshold: eigenvalues much smaller than max eigenvalue
        are treated as 0 to stabilize low-rank PSD roots.
    """
    A = _sym(A.to(dtype=torch.float64))
    evals, evecs = torch.linalg.eigh(A)

    # Only remove tiny negative numerical noise; do NOT floor all eigvals to eps.
    evals = torch.where(
        (evals < 0) & (evals > -neg_tol),
        torch.zeros_like(evals),
        evals,
    )
    evals = torch.clamp(evals, min=0.0)

    if zero_rel_tol is not None and evals.numel() > 0:
        max_eval = float(evals.max().item())
        if max_eval > 0:
            rel_cut = zero_rel_tol * max_eval
            evals = torch.where(evals < rel_cut, torch.zeros_like(evals), evals)

    return evecs @ torch.diag(torch.sqrt(evals)) @ evecs.T


def _truncate_rank_from_energy(evals, rank, energy_threshold=0.95):
    if evals.numel() == 0:
        return 1
    if energy_threshold is None:
        return max(1, min(rank, evals.numel()))
    energy = torch.cumsum(evals, dim=0) / (torch.sum(evals) + 1e-12)
    target = torch.tensor(energy_threshold, device=energy.device, dtype=energy.dtype)
    r = int(torch.searchsorted(energy, target).item()) + 1
    r = max(1, min(r, rank, evals.numel()))
    return r


def _top_right_basis_from_gram(delta, rank=32, energy_threshold=0.95, reg=1e-6):
    delta = delta.to(dtype=torch.float64)
    d_in = delta.shape[1]
    G = _sym(delta.T @ delta)
    G = G + reg * torch.eye(d_in, device=delta.device, dtype=G.dtype)

    evals, evecs = torch.linalg.eigh(G)
    idx = torch.argsort(evals, descending=True)
    evals = evals[idx]
    evecs = evecs[:, idx]
    evals = torch.clamp(evals, min=0.0)

    positive = evals > reg
    if positive.sum() == 0:
        V_r = torch.zeros((d_in, 1), device=delta.device, dtype=delta.dtype)
        V_r[0, 0] = 1.0
        S_r = torch.ones((1,), device=delta.device, dtype=delta.dtype) * (reg ** 0.5)
        fro_norm = torch.norm(delta, p="fro")
        return V_r, S_r, fro_norm

    evals = evals[positive]
    evecs = evecs[:, positive]
    r = _truncate_rank_from_energy(evals, rank=rank, energy_threshold=energy_threshold)
    V_r = evecs[:, :r].contiguous()
    S_r = torch.sqrt(evals[:r]).contiguous()
    fro_norm = torch.norm(delta, p="fro")
    return V_r, S_r, fro_norm


def _build_union_basis(bases, eps=1e-10):
    cat = torch.cat([b.to(dtype=torch.float64) for b in bases], dim=1)
    if cat.numel() == 0:
        raise ValueError("Empty basis list.")
    Q, R = torch.linalg.qr(cat, mode="reduced")
    diag = torch.abs(torch.diag(R))
    if diag.numel() == 0:
        return Q[:, :1]
    keep = diag > eps
    if keep.sum() == 0:
        return Q[:, :1]
    return Q[:, keep]

def _project_covariance_to_union_basis(Q, V_r, S_r, reg=1e-6):
    Q = Q.to(dtype=torch.float64)
    V_r = V_r.to(device=Q.device, dtype=torch.float64)
    S_r = S_r.to(device=Q.device, dtype=torch.float64)

    QTV = Q.T @ V_r
    B = QTV @ torch.diag(S_r ** 2) @ QTV.T
    B = _sym(B) + reg * _eye(Q.shape[1], B)
    return B


def _bures_distance_sq(A, B, eps=1e-12):
    A = _sym(A.to(dtype=torch.float64))
    B = _sym(B.to(dtype=torch.float64))

    A_sqrt = _matrix_sqrt_psd(A, eps=eps)
    inner = _sym(A_sqrt @ B @ A_sqrt)
    inner_sqrt = _matrix_sqrt_psd(inner, eps=eps)

    val = torch.trace(A) + torch.trace(B) - 2.0 * torch.trace(inner_sqrt)
    # Keep the clamp, but only after doing the entire path in float64.
    return torch.clamp(val, min=0.0)

def _bures_distance_sq_cpu64(A, B, neg_tol=1e-12, zero_rel_tol=1e-10):
    """
    Stable Bures distance on CPU float64.
    Returns (clamped_value, raw_value_before_clamp).
    """
    A_cpu = _sym(A.detach().to("cpu", dtype=torch.float64))
    B_cpu = _sym(B.detach().to("cpu", dtype=torch.float64))

    A_sqrt = _matrix_sqrt_psd(A_cpu, neg_tol=neg_tol, zero_rel_tol=zero_rel_tol)
    inner = _sym(A_sqrt @ B_cpu @ A_sqrt)
    inner_sqrt = _matrix_sqrt_psd(inner, neg_tol=neg_tol, zero_rel_tol=zero_rel_tol)

    raw = torch.trace(A_cpu) + torch.trace(B_cpu) - 2.0 * torch.trace(inner_sqrt)
    d2 = torch.clamp(raw, min=0.0)
    return d2, raw

class SummaryStore:
    def __init__(self, target_keys, rank=32, energy_threshold=0.95, cov_reg=1e-6, linalg_device="cpu"):
        self.target_keys = target_keys
        self.rank = rank
        self.energy_threshold = energy_threshold
        self.cov_reg = cov_reg
        self.linalg_device = _resolve_device(linalg_device, what="linalg_device")
        self.object_layer_cache = {}

    def _compute_layer_summary(self, delta, layer_key):
        x = delta[layer_key].to(self.linalg_device, dtype=torch.float64, non_blocking=True)
        V_r, S_r, fro_norm = _top_right_basis_from_gram(
            x,
            rank=self.rank,
            energy_threshold=self.energy_threshold,
            reg=self.cov_reg,
        )
        cov_trace = torch.sum(S_r ** 2).item() + self.cov_reg * x.shape[1]
        summary = {
            "fro_norm": float(fro_norm.item()),
            "retained_rank": int(V_r.shape[1]),
            "singular_values": S_r.detach().cpu().tolist(),
            "cov_trace": float(cov_trace),
            "basis": V_r.detach(),
            "svals": S_r.detach(),
        }
        return summary

    def get_object_summaries(self, obj_key, delta):
        if obj_key in self.object_layer_cache:
            return self.object_layer_cache[obj_key]
        layer_map = {}
        layer_pbar = tqdm(self.target_keys, desc=f"summaries::{obj_key}", leave=False)
        for layer_key in layer_pbar:
            layer_map[layer_key] = self._compute_layer_summary(delta, layer_key)
        self.object_layer_cache[obj_key] = layer_map
        return layer_map


def _sar_from_cached(delta_src_layer, basis_trg, src_norm):
    src_norm = max(float(src_norm), 1e-12)
    if delta_src_layer.device != basis_trg.device:
        delta_src_layer = delta_src_layer.to(basis_trg.device, dtype=torch.float64, non_blocking=True)
    else:
        delta_src_layer = delta_src_layer.to(dtype=torch.float64)
    basis_trg = basis_trg.to(dtype=torch.float64)
    val = torch.norm(delta_src_layer @ basis_trg, p="fro") / src_norm
    return float(val.item())


def _pair_metrics_from_cache(delta_i_layer, delta_j_layer, info_i, info_j, cov_reg=1e-6):
    device = info_i["basis"].device

    basis_i = info_i["basis"].to(device=device, dtype=torch.float64, non_blocking=True)
    basis_j = info_j["basis"].to(device=device, dtype=torch.float64, non_blocking=True)
    svals_i = info_i["svals"].to(device=device, dtype=torch.float64, non_blocking=True)
    svals_j = info_j["svals"].to(device=device, dtype=torch.float64, non_blocking=True)

    if delta_i_layer.device != device:
        delta_i_layer = delta_i_layer.to(device=device, dtype=torch.float64, non_blocking=True)
    else:
        delta_i_layer = delta_i_layer.to(dtype=torch.float64)

    if delta_j_layer.device != device:
        delta_j_layer = delta_j_layer.to(device=device, dtype=torch.float64, non_blocking=True)
    else:
        delta_j_layer = delta_j_layer.to(dtype=torch.float64)

    sar_i_to_j = _sar_from_cached(delta_i_layer, basis_j, info_i["fro_norm"])
    sar_j_to_i = _sar_from_cached(delta_j_layer, basis_i, info_j["fro_norm"])
    sar_sym = 0.5 * (sar_i_to_j + sar_j_to_i)

    Q = _build_union_basis([basis_i, basis_j])
    B_i = _project_covariance_to_union_basis(Q, basis_i, svals_i, reg=cov_reg)
    B_j = _project_covariance_to_union_basis(Q, basis_j, svals_j, reg=cov_reg)
    d2, raw = _bures_distance_sq_cpu64(B_i, B_j, neg_tol=1e-12, zero_rel_tol=1e-10)
    trace_i = float(torch.trace(B_i).detach().cpu().item())
    trace_j = float(torch.trace(B_j).detach().cpu().item())
    denom = max(trace_i + trace_j + cov_reg, 1e-12)
    gc = float(d2.item() / denom)

    return {
        "sar_i_to_j": sar_i_to_j,
        "sar_j_to_i": sar_j_to_i,
        "sar_sym": sar_sym,
        "geometry_conflict": gc,
        "bures_d2": float(d2.item()),
        "bures_raw_before_clamp": float(raw.item()),
        "bures_negative_before_clamp": bool(raw.item() < 0),
        "union_rank": int(Q.shape[1]),
        "trace_i": trace_i,
        "trace_j": trace_j,
        "retained_rank_i": info_i["retained_rank"],
        "retained_rank_j": info_j["retained_rank"],
        "delta_norm_i": info_i["fro_norm"],
        "delta_norm_j": info_j["fro_norm"],
        "singular_values_i": info_i["singular_values"],
        "singular_values_j": info_j["singular_values"],
    }


def _global_geometry_summary_from_cache(object_names, layer_infos, target_keys, cov_reg=1e-6, save_full_tensors=False):
    layer_summaries = {}
    full_tensors = {}

    for key in tqdm(target_keys, desc="global-geometry", leave=False):
        bases = []
        svds = []
        object_meta = {}
        for name in object_names:
            info = layer_infos[name][key]
            basis = info["basis"].to(dtype=torch.float64)
            svals = info["svals"].to(dtype=torch.float64)
            bases.append(basis)
            svds.append((basis, svals))
            object_meta[name] = {
                "retained_rank": info["retained_rank"],
                "fro_norm": info["fro_norm"],
                "cov_trace": info["cov_trace"],
                "singular_values": info["singular_values"],
            }

        Q = _build_union_basis(bases)
        proj_covs = []
        for _, (V_r, S_r) in zip(object_names, svds):
            B = _project_covariance_to_union_basis(Q, V_r, S_r, reg=cov_reg)
            proj_covs.append(B)

        conflicts = []
        pair_conf = {}
        for i, j in combinations(range(len(object_names)), 2):
            A = proj_covs[i]
            B = proj_covs[j]
            d2, raw = _bures_distance_sq_cpu64(A, B, neg_tol=1e-12, zero_rel_tol=1e-10)
            trace_i = float(torch.trace(A).detach().cpu().item())
            trace_j = float(torch.trace(B).detach().cpu().item())
            denom = max(trace_i + trace_j + cov_reg, 1e-12)
            gc = float(d2.item() / denom)
            pair_key = f"{object_names[i]}|||{object_names[j]}"
            pair_conf[pair_key] = {
                "bures_d2": float(d2.item()),
                "bures_raw_before_clamp": float(raw.item()),
                "bures_negative_before_clamp": bool(raw.item() < 0),
                "geometry_conflict": gc,
                "trace_i": trace_i,
                "trace_j": trace_j,
            }
            conflicts.append(gc)

        layer_summaries[key] = {
            "union_rank": int(Q.shape[1]),
            "mean_pairwise_geometry_conflict": float(sum(conflicts) / max(len(conflicts), 1)),
            "object_meta": object_meta,
            "pairwise_conflicts": pair_conf,
        }

        if save_full_tensors:
            full_tensors[key] = {
                "Q": Q.detach().cpu(),
                "projected_covs": {name: B.detach().cpu() for name, B in zip(object_names, proj_covs)},
            }

    mean_conf = sum(v["mean_pairwise_geometry_conflict"] for v in layer_summaries.values()) / max(len(layer_summaries), 1)
    out = {
        "mean_pairwise_geometry_conflict": float(mean_conf),
        "layers": layer_summaries,
    }
    if save_full_tensors:
        out["full_tensors"] = full_tensors
    return out


def _select_target_keys(first_delta, layer_filter="all", max_layers=None):
    keys = [k for k, v in first_delta.items() if _is_target_key(k, v)]
    if layer_filter == "attention_only":
        keys = [k for k in keys if any(x in k for x in ("q_proj.weight", "k_proj.weight", "v_proj.weight", "o_proj.weight"))]
    elif layer_filter == "mlp_only":
        keys = [k for k in keys if any(x in k for x in ("gate_proj.weight", "up_proj.weight", "down_proj.weight"))]
    elif layer_filter == "all":
        pass
    else:
        raise ValueError(f"Unsupported layer_filter: {layer_filter}")

    if max_layers is not None and max_layers > 0 and len(keys) > max_layers:
        idxs = torch.linspace(0, len(keys) - 1, steps=max_layers).long().tolist()
        keys = [keys[i] for i in idxs]
    return keys


def _slice_steps(steps_meta, step_start=None, step_end=None, max_steps=None):
    selected = steps_meta
    if step_start is not None or step_end is not None:
        lo = 1 if step_start is None else step_start
        hi = len(steps_meta) if step_end is None else step_end
        selected = [s for s in steps_meta if lo <= int(s["step"]) <= hi]
    if max_steps is not None and max_steps > 0:
        selected = selected[:max_steps]
    return selected


def _load_prev_merged_delta_for_first_selected(selected_steps_meta, all_steps_meta, delta_store):
    if len(selected_steps_meta) == 0:
        return None
    first_step_num = int(selected_steps_meta[0]["step"])
    if first_step_num <= 1:
        return None
    prev_step = all_steps_meta[first_step_num - 2]
    prev_new_task = prev_step.get("new_task", f"step_{first_step_num-1}")
    prev_save_dir = prev_step.get("save_dir", None)
    if prev_save_dir is None:
        return None
    prev_obj_key = f"merged_step::{first_step_num-1}::{prev_new_task}"
    return delta_store.load_or_build_delta(prev_obj_key, prev_save_dir)


def run_bridge_analysis(
    base_model,
    expert_models,
    continual_stats_path,
    output_dir,
    task_labels=None,
    rank=32,
    energy_threshold=0.95,
    cov_reg=1e-6,
    save_full_tensors=False,
    device="cpu",
    linalg_device=None,
    model_dtype="bfloat16",
    max_steps=None,
    max_layers=None,
    layer_filter="all",
    skip_merged_vs_active=False,
    skip_global_with_merged=False,
    step_start=None,
    step_end=None,
):
    os.makedirs(output_dir, exist_ok=True)

    compute_device = _resolve_device(device, what="device")
    linalg_device = _resolve_device(device if linalg_device is None else linalg_device, what="linalg_device")

    if linalg_device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        try:
            torch.set_float32_matmul_precision("highest")
        except Exception:
            pass
            
    # Preload CUDA linalg libs and fail fast if the runtime is broken.
    if compute_device.type == "cuda" or linalg_device.type == "cuda":
        _preload_torch_cuda_libraries()
        # Warm up on the linalg device, since eigh/qr/bures all depend on it.
        _cuda_linalg_smoke_test(linalg_device)

    stats = _load_stats(continual_stats_path)
    all_steps_meta = stats["steps"]
    steps_meta = _slice_steps(all_steps_meta, step_start=step_start, step_end=step_end, max_steps=max_steps)

    if task_labels is None:
        stats_task_names = stats.get("task_names", None)
        if stats_task_names is not None and len(stats_task_names) == len(expert_models):
            task_labels = stats_task_names
        else:
            task_labels = [os.path.basename(x.rstrip("/")) for x in expert_models]

    label_to_path = _build_label_to_path(expert_models, task_labels)
    delta_store = DeltaStore(base_model, os.path.join(output_dir, "delta_cache"), model_dtype=model_dtype)

    first_label = task_labels[0]
    first_delta = delta_store.load_or_build_delta(f"expert::{first_label}", label_to_path[first_label])
    target_keys = _select_target_keys(first_delta, layer_filter=layer_filter, max_layers=max_layers)

    summary_store = SummaryStore(
        target_keys=target_keys,
        rank=rank,
        energy_threshold=energy_threshold,
        cov_reg=cov_reg,
        linalg_device=str(linalg_device),
    )

    step_metrics_path = os.path.join(output_dir, "step_metrics.jsonl")
    pairwise_metrics_path = os.path.join(output_dir, "pairwise_layer_metrics.jsonl")
    for p in (step_metrics_path, pairwise_metrics_path):
        if os.path.exists(p):
            os.remove(p)

    finished_steps = []
    failed_steps = []
    prev_merged_delta = _load_prev_merged_delta_for_first_selected(steps_meta, all_steps_meta, delta_store)

    step_bar = tqdm(steps_meta, desc="bridge-analyzer steps")
    for step in step_bar:
        step_idx = int(step["step"]) - 1
        step_id = step_idx + 1
        new_task = step.get("new_task", f"step_{step_id}")
        step_save_dir = step.get("save_dir", None)
        error_path = os.path.join(output_dir, f"step_{step_id:02d}_error.txt")

        if os.path.exists(error_path):
            os.remove(error_path)

        if step_save_dir is None:
            msg = f"Step {step_id} has no save_dir; analyzer expects saved checkpoints."
            with open(error_path, "w", encoding="utf-8") as f:
                f.write(msg + "\n")
            failed_steps.append(step_id)
            print(f"[ERROR] {msg}")
            continue

        step_pairwise_records = []

        try:
            step_bar.set_postfix({"new_task": new_task, "stage": "load_delta"})
            merged_obj_key = f"merged_step::{step_id}::{new_task}"
            merged_step_delta = delta_store.load_or_build_delta(merged_obj_key, step_save_dir)
            merged_summaries = summary_store.get_object_summaries(merged_obj_key, merged_step_delta)

            active_names = step.get("active_task_names", [])
            active_delta_dicts = []
            active_object_types = []
            active_layer_infos = {}

            for name in active_names:
                obj_type, obj_path = _resolve_active_object(name, step_idx, all_steps_meta, label_to_path)
                cache_key = f"{obj_type}::{name}"
                delta = delta_store.load_or_build_delta(cache_key, obj_path)
                active_delta_dicts.append(delta)
                active_object_types.append(obj_type)
                active_layer_infos[name] = summary_store.get_object_summaries(cache_key, delta)

            merged_total_norm = 0.0
            increment_total_norm = 0.0
            for key in target_keys:
                merged_total_norm += float(torch.norm(merged_step_delta[key].float(), p="fro").item())
                if prev_merged_delta is None:
                    increment = merged_step_delta[key]
                else:
                    increment = merged_step_delta[key] - prev_merged_delta[key]
                increment_total_norm += float(torch.norm(increment.float(), p="fro").item())

            step_bar.set_postfix({"new_task": new_task, "stage": "active_pair"})
            step_pair_sars = []
            step_pair_gcs = []
            layer_bar = tqdm(target_keys, desc=f"active_pair::{step_id}", leave=False)
            for key in layer_bar:
                layer_active = [d[key].to(linalg_device, dtype=torch.float64, non_blocking=True) for d in active_delta_dicts]
                merged_layer = merged_step_delta[key].to(linalg_device, dtype=torch.float64, non_blocking=True)
                for i, j in combinations(range(len(active_names)), 2):
                    name_i = active_names[i]
                    name_j = active_names[j]
                    pm = _pair_metrics_from_cache(
                        layer_active[i],
                        layer_active[j],
                        active_layer_infos[name_i][key],
                        active_layer_infos[name_j][key],
                        cov_reg=cov_reg,
                    )
                    step_pairwise_records.append({
                        "step": step_id,
                        "context": "active_pair",
                        "layer_key": key,
                        "object_i": name_i,
                        "object_j": name_j,
                        "object_type_i": active_object_types[i],
                        "object_type_j": active_object_types[j],
                        "new_task": new_task,
                        **pm,
                    })
                    step_pair_sars.append(pm["sar_sym"])
                    step_pair_gcs.append(pm["geometry_conflict"])

                merged_vs_active_sars = [] if key == target_keys[0] else merged_vs_active_sars
                merged_vs_active_gcs = [] if key == target_keys[0] else merged_vs_active_gcs
                if not skip_merged_vs_active:
                    for i, name_i in enumerate(active_names):
                        pm = _pair_metrics_from_cache(
                            merged_layer,
                            layer_active[i],
                            merged_summaries[key],
                            active_layer_infos[name_i][key],
                            cov_reg=cov_reg,
                        )
                        step_pairwise_records.append({
                            "step": step_id,
                            "context": "merged_vs_active",
                            "layer_key": key,
                            "object_i": f"merged_step_{step_id}",
                            "object_j": name_i,
                            "object_type_i": "merged_step",
                            "object_type_j": active_object_types[i],
                            "new_task": new_task,
                            **pm,
                        })
                        merged_vs_active_sars.append(pm["sar_sym"])
                        merged_vs_active_gcs.append(pm["geometry_conflict"])

            step_bar.set_postfix({"new_task": new_task, "stage": "global_active"})
            global_active_geom = _global_geometry_summary_from_cache(
                object_names=active_names,
                layer_infos=active_layer_infos,
                target_keys=target_keys,
                cov_reg=cov_reg,
                save_full_tensors=save_full_tensors,
            )

            if not skip_global_with_merged:
                step_bar.set_postfix({"new_task": new_task, "stage": "global_with_merged"})
                merged_active_names = [f"merged_step_{step_id}"] + list(active_names)
                merged_active_infos = {f"merged_step_{step_id}": merged_summaries}
                merged_active_infos.update(active_layer_infos)
                global_with_merged_geom = _global_geometry_summary_from_cache(
                    object_names=merged_active_names,
                    layer_infos=merged_active_infos,
                    target_keys=target_keys,
                    cov_reg=cov_reg,
                    save_full_tensors=save_full_tensors,
                )
            else:
                global_with_merged_geom = None

            step_rec = {
                "step": step_id,
                "new_task": new_task,
                "num_active_tasks": len(active_names),
                "active_task_names": active_names,
                "active_task_weights": step.get("active_task_weights", None),
                "step_coef": step.get("step_coef", None),
                "save_dir": step_save_dir,
                "num_target_layers": len(target_keys),
                "layer_filter": layer_filter,
                "merged_total_norm": merged_total_norm,
                "increment_total_norm": increment_total_norm,
                "mean_active_pair_sar": float(sum(step_pair_sars) / max(len(step_pair_sars), 1)),
                "mean_active_pair_geometry_conflict": float(sum(step_pair_gcs) / max(len(step_pair_gcs), 1)),
                "mean_merged_vs_active_sar": float(sum(merged_vs_active_sars) / max(len(merged_vs_active_sars), 1)) if len(merged_vs_active_sars) > 0 else None,
                "mean_merged_vs_active_geometry_conflict": float(sum(merged_vs_active_gcs) / max(len(merged_vs_active_gcs), 1)) if len(merged_vs_active_gcs) > 0 else None,
                "global_active_geometry": {
                    "mean_pairwise_geometry_conflict": global_active_geom["mean_pairwise_geometry_conflict"],
                },
                "global_with_merged_geometry": None if global_with_merged_geom is None else {
                    "mean_pairwise_geometry_conflict": global_with_merged_geom["mean_pairwise_geometry_conflict"],
                },
            }

            _append_jsonl(pairwise_metrics_path, step_pairwise_records)
            _append_jsonl(step_metrics_path, [step_rec])

            step_detail = {
                "step_summary": step_rec,
                "global_active_geometry": global_active_geom,
                "global_with_merged_geometry": global_with_merged_geom,
            }
            with open(os.path.join(output_dir, f"step_{step_id:02d}_bridge_detail.json"), "w", encoding="utf-8") as f:
                json.dump(step_detail, f, indent=2)

            if save_full_tensors:
                tensor_blob = {
                    "global_active_geometry": global_active_geom.get("full_tensors", {}),
                    "global_with_merged_geometry": {} if global_with_merged_geom is None else global_with_merged_geom.get("full_tensors", {}),
                }
                torch.save(tensor_blob, os.path.join(output_dir, f"step_{step_id:02d}_bridge_tensors.pt"))

            finished_steps.append(step_id)
            prev_merged_delta = merged_step_delta

        except Exception:
            tb = traceback.format_exc()
            with open(error_path, "w", encoding="utf-8") as f:
                f.write(tb)
            failed_steps.append(step_id)
            print(f"[ERROR] step {step_id:02d} failed; see {error_path}")

        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    summary = {
        "base_model": base_model,
        "continual_stats_path": continual_stats_path,
        "expert_models": expert_models,
        "task_labels": task_labels,
        "rank": rank,
        "energy_threshold": energy_threshold,
        "cov_reg": cov_reg,
        "device": str(compute_device),
        "linalg_device": str(linalg_device),
        "layer_filter": layer_filter,
        "max_layers": max_layers,
        "max_steps": max_steps,
        "step_start": step_start,
        "step_end": step_end,
        "skip_merged_vs_active": skip_merged_vs_active,
        "skip_global_with_merged": skip_global_with_merged,
        "num_steps": len(finished_steps),
        "finished_steps": finished_steps,
        "failed_steps": failed_steps,
        "num_target_layers": len(target_keys),
        "step_metrics_path": step_metrics_path,
        "pairwise_metrics_path": pairwise_metrics_path,
    }
    with open(os.path.join(output_dir, "bridge_analysis_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


def build_parser():
    p = argparse.ArgumentParser("bridge_analyzer_fast_sharded")
    p.add_argument("--base-model", type=str, required=True)
    p.add_argument("--expert-models", type=str, required=True,
                   help="Comma-separated expert checkpoint paths in the same order as continual stats task_names")
    p.add_argument("--continual-stats", type=str, required=True,
                   help="Path to continual_*_stats.json or run directory containing it")
    p.add_argument("--output-dir", type=str, required=True)
    p.add_argument("--task-labels", type=str, default=None,
                   help="Comma-separated labels matching --expert-models; defaults to stats.task_names or basenames")
    p.add_argument("--rank", type=int, default=32)
    p.add_argument("--energy-threshold", type=float, default=0.95)
    p.add_argument("--cov-reg", type=float, default=1e-6)
    p.add_argument("--save-full-tensors", action="store_true")
    p.add_argument("--device", type=str, default=("cuda:0" if torch.cuda.is_available() else "cpu"))
    p.add_argument("--linalg-device", type=str, default=None,
                   help="Device for eigh/qr/bures linear algebra. Default: same as --device.")
    p.add_argument("--model-dtype", type=str, default="bfloat16", choices=["float16", "bfloat16", "float32"])
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--max-layers", type=int, default=None)
    p.add_argument("--layer-filter", type=str, default="all", choices=["all", "attention_only", "mlp_only"])
    p.add_argument("--skip-merged-vs-active", action="store_true")
    p.add_argument("--skip-global-with-merged", action="store_true")
    p.add_argument("--step-start", type=int, default=None, help="1-indexed inclusive step start for sharding")
    p.add_argument("--step-end", type=int, default=None, help="1-indexed inclusive step end for sharding")
    return p


def main():
    args = build_parser().parse_args()
    expert_models = _parse_csv(args.expert_models)
    task_labels = _parse_csv(args.task_labels)
    continual_stats_path = _infer_stats_path(args.continual_stats)

    print(f"[INFO] requested device={args.device}, linalg_device={args.linalg_device}")
    print(f"[INFO] torch.cuda.is_available()={torch.cuda.is_available()}")

    summary = run_bridge_analysis(
        base_model=args.base_model,
        expert_models=expert_models,
        continual_stats_path=continual_stats_path,
        output_dir=args.output_dir,
        task_labels=task_labels,
        rank=args.rank,
        energy_threshold=args.energy_threshold,
        cov_reg=args.cov_reg,
        save_full_tensors=args.save_full_tensors,
        device=args.device,
        linalg_device=args.linalg_device,
        model_dtype=args.model_dtype,
        max_steps=args.max_steps,
        max_layers=args.max_layers,
        layer_filter=args.layer_filter,
        skip_merged_vs_active=args.skip_merged_vs_active,
        skip_global_with_merged=args.skip_global_with_merged,
        step_start=args.step_start,
        step_end=args.step_end,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

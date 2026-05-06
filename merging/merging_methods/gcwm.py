import os
import json
from collections import OrderedDict

import torch
from tqdm import tqdm

from merging_methods.merger import Merger


TARGET_KEYWORDS = (
    "q_proj.weight",
    "k_proj.weight",
    "v_proj.weight",
    "o_proj.weight",
    "gate_proj.weight",
    "up_proj.weight",
    "down_proj.weight",
)

DENSE_LOCAL_KEYWORDS = (
    "q_proj.weight",
    "k_proj.weight",
    "v_proj.weight",
    "o_proj.weight",
)


def _is_dense_local_key(key):
    return any(x in key for x in DENSE_LOCAL_KEYWORDS)


# =========================================================
# State dict / task vectors
# =========================================================

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


def _build_task_vector_dict(ft_model, base_model):
    ft_sd = _select_trainable_state_dict(ft_model)
    base_sd = _select_trainable_state_dict(base_model)

    tv = OrderedDict()
    for k in base_sd:
        if k not in ft_sd:
            continue
        tv[k] = ft_sd[k] - base_sd[k]
    return tv


def _is_target_key(key, tensor):
    if tensor is None or tensor.ndim != 2:
        return False
    return any(x in key for x in TARGET_KEYWORDS)


def _apply_delta_to_model(delta_dict, pretrained_model, scaling_coef=1.0):
    with torch.no_grad():
        model_sd = pretrained_model.state_dict()
        new_sd = OrderedDict()

        for k, v in model_sd.items():
            if k in delta_dict:
                new_sd[k] = v + scaling_coef * delta_dict[k].to(v.device, dtype=v.dtype)
            else:
                new_sd[k] = v

        pretrained_model.load_state_dict(new_sd, strict=False)

    return pretrained_model


# =========================================================
# Linear algebra helpers
# =========================================================

def _sym(A):
    return 0.5 * (A + A.T)


def _eye(n, like, scale=1.0):
    return scale * torch.eye(n, device=like.device, dtype=like.dtype)


def _matrix_sqrt_psd(A, eps=1e-10):
    A = _sym(A.float())
    evals, evecs = torch.linalg.eigh(A)
    evals = torch.clamp(evals, min=eps)
    return evecs @ torch.diag(torch.sqrt(evals)) @ evecs.T


def _matrix_inv_sqrt_psd(A, eps=1e-10):
    A = _sym(A.float())
    evals, evecs = torch.linalg.eigh(A)
    evals = torch.clamp(evals, min=eps)
    return evecs @ torch.diag(torch.rsqrt(evals)) @ evecs.T


def _truncate_rank_from_energy(S, rank, energy_threshold=0.9):
    if energy_threshold is None:
        return max(1, min(rank, S.numel()))
    energy = torch.cumsum(S ** 2, dim=0) / (torch.sum(S ** 2) + 1e-12)
    target = torch.tensor(energy_threshold, device=energy.device, dtype=energy.dtype)
    r = int(torch.searchsorted(energy, target).item()) + 1
    r = max(1, min(r, rank, S.numel()))
    return r


def _truncated_right_svd(delta, rank=8, energy_threshold=0.9):
    _, S, Vh = torch.linalg.svd(delta.float(), full_matrices=False)
    r = _truncate_rank_from_energy(S, rank, energy_threshold)
    V_r = Vh[:r, :].T.contiguous()
    S_r = S[:r].contiguous()
    return V_r, S_r


def _build_union_basis(bases, eps=1e-8):
    cat = torch.cat(bases, dim=1).float()
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


# =========================================================
# Capability geometry
# =========================================================

def _estimate_capability_covariance_dense(delta, reg=1e-6):
    delta = delta.float()
    d_in = delta.shape[1]
    Sigma = delta.T @ delta
    Sigma = _sym(Sigma) + reg * torch.eye(d_in, device=delta.device, dtype=Sigma.dtype)
    return Sigma


def _estimate_capability_covariance_lowrank(delta, rank=8, energy_threshold=0.9):
    V_r, S_r = _truncated_right_svd(delta, rank=rank, energy_threshold=energy_threshold)
    return V_r, S_r


def _project_covariance_to_union_basis(Q, V_r, S_r, reg=1e-6):
    QTV = Q.T @ V_r
    B = QTV @ torch.diag(S_r ** 2) @ QTV.T
    B = _sym(B) + reg * _eye(Q.shape[1], B)
    return B


def _full_metric_apply_right(delta, Q, B, reg=1e-6, inverse_sqrt=False):
    delta = delta.float()
    Q = Q.float()
    B = _sym(B.float())

    if inverse_sqrt:
        B_half = _matrix_inv_sqrt_psd(B, eps=reg)
        scalar = reg ** (-0.5)
    else:
        B_half = _matrix_sqrt_psd(B, eps=reg)
        scalar = reg ** 0.5

    delta_Q = delta @ Q
    parallel = delta_Q @ B_half @ Q.T
    orth = delta - delta_Q @ Q.T
    return parallel + scalar * orth


def _relation_metric_whiten_dense(vectors, Sigma_shared, reg=1e-6):
    Sigma_inv_sqrt = _matrix_inv_sqrt_psd(Sigma_shared, eps=reg)
    whitened = [vectors[t].float() @ Sigma_inv_sqrt for t in range(vectors.shape[0])]
    return torch.stack(whitened, dim=0)


def _relation_metric_recolor_dense(merged_whitened, Sigma_shared, reg=1e-6):
    Sigma_sqrt = _matrix_sqrt_psd(Sigma_shared, eps=reg)
    return merged_whitened.float() @ Sigma_sqrt


# =========================================================
# Gaussian W2 barycenter on projected SPD matrices
# =========================================================

def _gaussian_w2_barycenter_standard(covs, weights=None, max_iter=50, tol=1e-5, eps=1e-8, damping=1.0):
    n = len(covs)
    if n == 0:
        raise ValueError("No covariance matrices.")

    device = covs[0].device
    dtype = covs[0].dtype

    if weights is None:
        weights = [1.0 / n] * n
    weights = torch.tensor(weights, device=device, dtype=dtype)
    weights = weights / weights.sum()

    d = covs[0].shape[0]
    reg_eye = eps * torch.eye(d, device=device, dtype=dtype)
    covs = [_sym(C.float()) + reg_eye for C in covs]
    B = sum(w * C for w, C in zip(weights, covs))
    B = _sym(B) + reg_eye

    for _ in range(max_iter):
        B_sqrt = _matrix_sqrt_psd(B, eps=eps)
        B_inv_sqrt = _matrix_inv_sqrt_psd(B, eps=eps)
        T = torch.zeros_like(B)
        for w, C in zip(weights, covs):
            inner = _sym(B_sqrt @ C @ B_sqrt)
            T = T + w * _matrix_sqrt_psd(inner, eps=eps)
        B_next = _sym(B_inv_sqrt @ (T @ T) @ B_inv_sqrt) + reg_eye
        if damping < 1.0:
            B_next = _sym(damping * B_next + (1.0 - damping) * B)
        rel_diff = torch.norm(B_next - B, p="fro") / (torch.norm(B, p="fro") + eps)
        B = B_next
        if rel_diff.item() < tol:
            break
    return _sym(B)


# =========================================================
# Conflict / weighting helpers
# =========================================================

def _bures_distance_sq(A, B, eps=1e-8):
    A = _sym(A.float())
    B = _sym(B.float())
    A_sqrt = _matrix_sqrt_psd(A, eps=eps)
    inner = _sym(A_sqrt @ B @ A_sqrt)
    inner_sqrt = _matrix_sqrt_psd(inner, eps=eps)
    val = torch.trace(A) + torch.trace(B) - 2.0 * torch.trace(inner_sqrt)
    return torch.clamp(val, min=0.0)


def _normalize_task_weights(task_weights, T, device, dtype):
    if task_weights is None:
        w = torch.ones(T, device=device, dtype=dtype)
    else:
        if len(task_weights) != T:
            raise ValueError(f"task_weights length {len(task_weights)} != number of tasks {T}")
        w = torch.tensor(task_weights, device=device, dtype=dtype)
        w = torch.clamp(w, min=0.0)
        if torch.sum(w) <= 0:
            w = torch.ones(T, device=device, dtype=dtype)
    return w


def _compute_layer_conflict_score(projected_covs, eps=1e-8, task_weights=None):
    n = len(projected_covs)
    if n <= 1:
        return 0.0

    if task_weights is None:
        pair_weights = None
    else:
        tw = torch.tensor(task_weights, dtype=torch.float32)
        tw = torch.clamp(tw, min=0.0)
        if tw.sum() <= 0:
            tw = torch.ones_like(tw)
        pair_weights = []

    vals = []
    pw = []
    for i in range(n):
        for j in range(i + 1, n):
            A = projected_covs[i]
            B = projected_covs[j]
            d2 = _bures_distance_sq(A, B, eps=eps)
            denom = torch.trace(A) + torch.trace(B) + eps
            vals.append((d2 / denom).item())
            if pair_weights is not None:
                pw.append(float((tw[i] * tw[j]).item()))

    if len(vals) == 0:
        return 0.0
    if pair_weights is None:
        return float(sum(vals) / len(vals))
    total = sum(pw)
    if total <= 0:
        return float(sum(vals) / len(vals))
    return float(sum(v * w for v, w in zip(vals, pw)) / total)


def _blend_factor_from_conflict(conflict_score, gate_threshold=0.12, gate_sharpness=12.0,
                                gate_min_alpha=0.15, gate_max_alpha=0.95):
    z = gate_sharpness * (conflict_score - gate_threshold)
    alpha = 1.0 / (1.0 + torch.exp(torch.tensor(-z))).item()
    alpha = gate_min_alpha + (gate_max_alpha - gate_min_alpha) * alpha
    alpha = max(gate_min_alpha, min(gate_max_alpha, alpha))
    return float(alpha)


# =========================================================
# Inner optimizer (weighted)
# =========================================================

def _optimize_relation_merge_vector_weighted(key, vectors, iter_num=300, lr=1e-5, task_weights=None, device="cuda"):
    if "cuda" in str(device).lower() and torch.cuda.is_available():
        device = torch.device(device)
    else:
        device = torch.device("cpu")

    vectors = vectors.to(device, non_blocking=True).float()
    T = vectors.shape[0]
    weights = _normalize_task_weights(task_weights, T, device, vectors.dtype)

    merging_vector = torch.nn.Parameter(torch.sum(vectors, dim=0))
    optimizer = torch.optim.Adam([merging_vector], lr=lr, weight_decay=0.0)

    l2_norms = torch.square(torch.norm(vectors.reshape(T, -1), p=2, dim=-1)) + 1e-12

    for _ in tqdm(range(iter_num), desc=f"GCWM-opt {key}", leave=False):
        disturbing_vectors = merging_vector.unsqueeze(0) - vectors
        inner_product = torch.matmul(disturbing_vectors, vectors.transpose(1, 2))
        per_task_loss = torch.sum(torch.square(inner_product) / l2_norms.unsqueeze(-1).unsqueeze(-1), dim=(1, 2))
        loss = torch.sum(weights * per_task_loss)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return merging_vector.detach()


# =========================================================
# GCWM layer merge (weighted)
# =========================================================

@torch.no_grad()
def _prepare_relation_metric(vectors, rank=8, energy_threshold=0.9, metric_mode="barycenter",
                             bary_max_iter=30, bary_tol=1e-5, bary_damping=1.0, reg=1e-6,
                             task_weights=None):
    T = vectors.shape[0]
    weights = _normalize_task_weights(task_weights, T, vectors.device, vectors.dtype)
    weights = weights / torch.clamp(weights.sum(), min=1e-12)
    weights_list = [float(x.item()) for x in weights]

    if metric_mode == "dense_mean":
        dense_covs = [
            _estimate_capability_covariance_dense(vectors[t], reg=reg)
            for t in range(T)
        ]
        Sigma_shared = sum(w * C for w, C in zip(weights, dense_covs))
        Sigma_shared = _sym(Sigma_shared)
        conflict_score = _compute_layer_conflict_score(dense_covs, eps=reg, task_weights=weights_list)
        return {
            "repr": "dense",
            "Sigma_shared": Sigma_shared,
            "conflict_score": conflict_score,
        }

    bases = []
    svds = []
    for t in range(T):
        V_r, S_r = _estimate_capability_covariance_lowrank(vectors[t], rank=rank, energy_threshold=energy_threshold)
        bases.append(V_r)
        svds.append((V_r, S_r))

    Q = _build_union_basis(bases)
    proj_covs = [
        _project_covariance_to_union_basis(Q, V_r, S_r, reg=reg)
        for (V_r, S_r) in svds
    ]

    if metric_mode == "mean":
        B_shared = sum(w * C for w, C in zip(weights, proj_covs))
        B_shared = _sym(B_shared)
    elif metric_mode == "barycenter":
        B_shared = _gaussian_w2_barycenter_standard(
            proj_covs,
            weights=weights_list,
            max_iter=bary_max_iter,
            tol=bary_tol,
            eps=reg,
            damping=bary_damping,
        )
    else:
        raise ValueError(f"Unsupported metric_mode: {metric_mode}")

    conflict_score = _compute_layer_conflict_score(proj_covs, eps=reg, task_weights=weights_list)
    return {
        "repr": "lowrank",
        "Q": Q,
        "B_shared": B_shared,
        "conflict_score": conflict_score,
    }


@torch.no_grad()
def _relation_metric_whiten(vectors, Q, B_shared, reg=1e-6):
    whitened = []
    for t in range(vectors.shape[0]):
        w = _full_metric_apply_right(vectors[t], Q, B_shared, reg=reg, inverse_sqrt=True)
        whitened.append(w)
    return torch.stack(whitened, dim=0)


@torch.no_grad()
def _relation_metric_recolor(merged_whitened, Q, B_shared, reg=1e-6):
    return _full_metric_apply_right(merged_whitened, Q, B_shared, reg=reg, inverse_sqrt=False)


def _gcwm_merge_one_layer(key, vectors, iter_num=300, lr=1e-5, rank=8, energy_threshold=0.9,
                          metric_mode="barycenter", bary_max_iter=30, bary_tol=1e-5,
                          bary_damping=1.0, cov_reg=1e-6, gate_mode="conflict",
                          gate_threshold=0.12, gate_sharpness=12.0, gate_min_alpha=0.15,
                          gate_max_alpha=0.95, gate_skip_tol=0.05, task_weights=None,
                          device="cuda"):
    if "cuda" in str(device).lower() and torch.cuda.is_available():
        device = torch.device(device)
    else:
        device = torch.device("cpu")

    vectors = vectors.to(device, non_blocking=True).float()
    T = vectors.shape[0]
    weights = _normalize_task_weights(task_weights, T, device, vectors.dtype)
    weights_list = [float(x.item()) for x in weights]

    effective_metric_mode = metric_mode
    if metric_mode == "dense_mean" and (not _is_dense_local_key(key)):
        effective_metric_mode = "mean"

    metric_info = _prepare_relation_metric(
        vectors=vectors,
        rank=rank,
        energy_threshold=energy_threshold,
        metric_mode=effective_metric_mode,
        bary_max_iter=bary_max_iter,
        bary_tol=bary_tol,
        bary_damping=bary_damping,
        reg=cov_reg,
        task_weights=weights_list,
    )

    conflict_score = metric_info["conflict_score"]

    if gate_mode == "none":
        alpha = 1.0
    elif gate_mode == "conflict":
        alpha = _blend_factor_from_conflict(
            conflict_score=conflict_score,
            gate_threshold=gate_threshold,
            gate_sharpness=gate_sharpness,
            gate_min_alpha=gate_min_alpha,
            gate_max_alpha=gate_max_alpha,
        )
    else:
        raise ValueError(f"Unsupported gate_mode: {gate_mode}")

    if alpha <= gate_skip_tol:
        merged_plain = _optimize_relation_merge_vector_weighted(
            key=key,
            vectors=vectors,
            iter_num=iter_num,
            lr=lr,
            task_weights=weights_list,
            device=device,
        )
        return merged_plain.detach().cpu(), {
            "alpha": float(alpha),
            "conflict_score": float(conflict_score),
            "union_rank": -1 if metric_info["repr"] == "dense" else int(metric_info["Q"].shape[1]),
            "dense_local": bool(metric_info["repr"] == "dense"),
            "mode": "plain_only",
        }

    if metric_info["repr"] == "dense":
        whitened_vectors = _relation_metric_whiten_dense(vectors=vectors, Sigma_shared=metric_info["Sigma_shared"], reg=cov_reg)
    else:
        whitened_vectors = _relation_metric_whiten(vectors=vectors, Q=metric_info["Q"], B_shared=metric_info["B_shared"], reg=cov_reg)

    merged_whitened = _optimize_relation_merge_vector_weighted(
        key=key + "_relation",
        vectors=whitened_vectors,
        iter_num=iter_num,
        lr=lr,
        task_weights=weights_list,
        device=device,
    )

    if metric_info["repr"] == "dense":
        merged_relation = _relation_metric_recolor_dense(merged_whitened=merged_whitened, Sigma_shared=metric_info["Sigma_shared"], reg=cov_reg)
    else:
        merged_relation = _relation_metric_recolor(merged_whitened=merged_whitened, Q=metric_info["Q"], B_shared=metric_info["B_shared"], reg=cov_reg)

    if alpha >= 1.0 - gate_skip_tol:
        return merged_relation.detach().cpu(), {
            "alpha": float(alpha),
            "conflict_score": float(conflict_score),
            "union_rank": -1 if metric_info["repr"] == "dense" else int(metric_info["Q"].shape[1]),
            "dense_local": bool(metric_info["repr"] == "dense"),
            "mode": "gcwm_only",
            "merged_whitened_norm": float(torch.norm(merged_whitened).item()),
            "merged_relation_norm": float(torch.norm(merged_relation).item()),
        }

    merged_plain = _optimize_relation_merge_vector_weighted(
        key=key + "_plain",
        vectors=vectors,
        iter_num=iter_num,
        lr=lr,
        task_weights=weights_list,
        device=device,
    )
    merged = alpha * merged_relation + (1.0 - alpha) * merged_plain

    stats = {
        "alpha": float(alpha),
        "conflict_score": float(conflict_score),
        "union_rank": -1 if metric_info["repr"] == "dense" else int(metric_info["Q"].shape[1]),
        "dense_local": bool(metric_info["repr"] == "dense"),
        "mode": "blended",
        "merged_whitened_norm": float(torch.norm(merged_whitened).item()),
        "merged_relation_norm": float(torch.norm(merged_relation).item()),
        "merged_plain_norm": float(torch.norm(merged_plain).item()),
        "merged_final_norm": float(torch.norm(merged).item()),
    }
    return merged.detach().cpu(), stats


# =========================================================
# Continual helpers
# =========================================================

def _infer_task_names_from_models(models):
    task_names = []
    for m in models:
        name_or_path = None
        if hasattr(m, "name_or_path") and isinstance(m.name_or_path, str):
            name_or_path = m.name_or_path
        elif hasattr(m, "config") and hasattr(m.config, "_name_or_path"):
            name_or_path = m.config._name_or_path
        elif hasattr(m, "config") and hasattr(m.config, "name_or_path"):
            name_or_path = m.config.name_or_path
        if not isinstance(name_or_path, str):
            task_names.append("unknown")
            continue
        base = os.path.basename(name_or_path.rstrip("/"))
        task_names.append(base.split("_")[-1])
    return task_names


def _get_common_target_keys(task_vectors):
    common_keys = set(task_vectors[0].keys())
    for tv in task_vectors[1:]:
        common_keys &= set(tv.keys())
    target_keys = [k for k in task_vectors[0].keys() if k in common_keys and _is_target_key(k, task_vectors[0][k])]
    return target_keys


def _compute_increment_dict(new_delta, old_delta):
    inc = OrderedDict()
    if old_delta is None:
        for k, v in new_delta.items():
            inc[k] = v
        return inc
    for k, v in new_delta.items():
        prev = old_delta.get(k, None)
        inc[k] = v if prev is None else v - prev
    return inc


class GCWM(Merger):
    """
    MergeBench-compatible GCWM with continual-v2 support.

    Batch mode:
        relation-conditioned Wasserstein merging over a batch of task deltas.

    Continual mode:
        - all task vectors are always base-relative
        - old/new task weights are explicit
        - each step applies a step-specific outer coefficient
        - supports all_history / current_anchor memory
        - saves per-step models and GCWM diagnostics for analysis
    """

    def __init__(self, base_model, ft_models, save_path):
        super().__init__(base_model, ft_models, save_path)

    def _build_active_set(self, history_task_vectors, prev_merged_delta, new_task_vector, task_names, step_idx,
                          memory_mode="all_history", memory_size=-1, old_weight=1.0, new_weight=1.0):
        if memory_mode == "all_history":
            history_task_vectors.append(new_task_vector)
            if memory_size is not None and memory_size > 0:
                active_task_vectors = history_task_vectors[-memory_size:]
                active_task_names = task_names[max(0, step_idx + 1 - len(active_task_vectors)): step_idx + 1]
            else:
                active_task_vectors = history_task_vectors
                active_task_names = task_names[:step_idx + 1]

            if len(active_task_vectors) == 1:
                active_task_weights = [new_weight]
            else:
                active_task_weights = [old_weight] * (len(active_task_vectors) - 1) + [new_weight]

        elif memory_mode == "current_anchor":
            if prev_merged_delta is None:
                active_task_vectors = [new_task_vector]
                active_task_names = [task_names[step_idx]]
                active_task_weights = [new_weight]
            else:
                active_task_vectors = [prev_merged_delta, new_task_vector]
                active_task_names = [f"merged_until_{step_idx}", task_names[step_idx]]
                active_task_weights = [old_weight, new_weight]
        else:
            raise ValueError(f"Unsupported memory_mode: {memory_mode}")

        return active_task_vectors, active_task_names, active_task_weights

    def _merge_batch(self, **kwargs):
        iter_num = kwargs.get("iter_num", kwargs.get("iter", 300))
        scaling_coef = kwargs.get("scaling_coef", 1.0)
        inner_lr = kwargs.get("gcwm_lr", kwargs.get("inner_lr", 1e-5))
        rank = kwargs.get("rank", 16)
        energy_threshold = kwargs.get("energy_threshold", 0.9)

        metric_mode = kwargs.get("metric_mode", "barycenter")
        bary_max_iter = kwargs.get("bary_max_iter", 30)
        bary_tol = kwargs.get("bary_tol", 1e-5)
        bary_damping = kwargs.get("bary_damping", 1.0)
        cov_reg = kwargs.get("cov_reg", 1e-6)

        gate_mode = kwargs.get("gate_mode", "conflict")
        gate_threshold = kwargs.get("gate_threshold", 0.12)
        gate_sharpness = kwargs.get("gate_sharpness", 12.0)
        gate_min_alpha = kwargs.get("gate_min_alpha", 0.15)
        gate_max_alpha = kwargs.get("gate_max_alpha", 0.95)
        gate_skip_tol = kwargs.get("gate_skip_tol", 0.05)
        task_weights = kwargs.get("task_weights", None)

        save_stats = kwargs.get("save_stats", True)
        save_layer_stats = kwargs.get("save_layer_stats", False)
        device = kwargs.get("device", "cuda" if torch.cuda.is_available() else "cpu")

        os.makedirs(self.save_path, exist_ok=True)
        task_vectors = [_build_task_vector_dict(ft_model, self.base_model) for ft_model in self.ft_ckpts]
        target_keys = _get_common_target_keys(task_vectors)

        merged_delta = OrderedDict()
        layer_stats = {} if (save_stats and save_layer_stats) else None
        for key in tqdm(target_keys, desc="GCWM decompose"):
            values = torch.stack([tv[key] for tv in task_vectors], dim=0).clone()
            merged_delta[key], stats = _gcwm_merge_one_layer(
                key=key,
                vectors=values,
                iter_num=iter_num,
                lr=inner_lr,
                rank=rank,
                energy_threshold=energy_threshold,
                metric_mode=metric_mode,
                bary_max_iter=bary_max_iter,
                bary_tol=bary_tol,
                bary_damping=bary_damping,
                cov_reg=cov_reg,
                gate_mode=gate_mode,
                gate_threshold=gate_threshold,
                gate_sharpness=gate_sharpness,
                gate_min_alpha=gate_min_alpha,
                gate_max_alpha=gate_max_alpha,
                gate_skip_tol=gate_skip_tol,
                task_weights=task_weights,
                device=device,
            )
            if layer_stats is not None:
                layer_stats[key] = stats

        merged_model = _apply_delta_to_model(
            delta_dict=merged_delta,
            pretrained_model=self.base_model,
            scaling_coef=scaling_coef,
        )
        merged_model.save_pretrained(self.save_path)
        self.tokenizer.save_pretrained(self.save_path)

        if layer_stats is not None:
            with open(os.path.join(self.save_path, "gcwm_layer_stats.json"), "w") as f:
                json.dump(layer_stats, f, indent=2)

    def _merge_continual(self, **kwargs):
        iter_num = kwargs.get("iter_num", kwargs.get("iter", 200))
        scaling_coef = kwargs.get("scaling_coef", 1.0)
        continual_step_coef = kwargs.get("continual_step_coef", scaling_coef)
        step_decay = kwargs.get("step_decay", 1.0)
        old_weight = kwargs.get("old_weight", 1.0)
        new_weight = kwargs.get("new_weight", 1.0)
        inner_lr = kwargs.get("gcwm_lr", kwargs.get("inner_lr", 1e-5))
        device = kwargs.get("device", "cuda" if torch.cuda.is_available() else "cpu")

        rank = kwargs.get("rank", 16)
        energy_threshold = kwargs.get("energy_threshold", 0.9)
        metric_mode = kwargs.get("metric_mode", "barycenter")
        bary_max_iter = kwargs.get("bary_max_iter", 30)
        bary_tol = kwargs.get("bary_tol", 1e-5)
        bary_damping = kwargs.get("bary_damping", 1.0)
        cov_reg = kwargs.get("cov_reg", 1e-6)
        gate_mode = kwargs.get("gate_mode", "conflict")
        gate_threshold = kwargs.get("gate_threshold", 0.12)
        gate_sharpness = kwargs.get("gate_sharpness", 12.0)
        gate_min_alpha = kwargs.get("gate_min_alpha", 0.15)
        gate_max_alpha = kwargs.get("gate_max_alpha", 0.95)
        gate_skip_tol = kwargs.get("gate_skip_tol", 0.05)

        memory_mode = kwargs.get("memory_mode", "all_history")
        memory_size = kwargs.get("memory_size", -1)
        save_each_step = kwargs.get("save_each_step", True)
        save_stats = kwargs.get("save_stats", True)
        task_order_list = kwargs.get("task_order_list", None)
        save_step_layer_stats = kwargs.get("save_step_layer_stats", False)

        os.makedirs(self.save_path, exist_ok=True)

        all_task_vectors = [
            _build_task_vector_dict(ft_model, self.base_model)
            for ft_model in self.ft_ckpts
        ]
        target_keys = _get_common_target_keys(all_task_vectors)

        inferred_task_names = _infer_task_names_from_models(self.ft_ckpts)
        if task_order_list is None or len(task_order_list) != len(inferred_task_names):
            task_names = inferred_task_names
        else:
            task_names = task_order_list

        history_task_vectors = []
        prev_merged_delta = None
        continual_stats = {
            "mode": "gcwm",
            "memory_mode": memory_mode,
            "memory_size": memory_size,
            "task_names": task_names,
            "continual_step_coef": continual_step_coef,
            "step_decay": step_decay,
            "old_weight": old_weight,
            "new_weight": new_weight,
            "steps": [],
        }

        for step_idx, new_task_vector in enumerate(all_task_vectors):
            task_name = task_names[step_idx] if step_idx < len(task_names) else f"task_{step_idx}"
            step_coef = continual_step_coef * (step_decay ** step_idx)

            active_task_vectors, active_task_names, active_task_weights = self._build_active_set(
                history_task_vectors=history_task_vectors,
                prev_merged_delta=prev_merged_delta,
                new_task_vector=new_task_vector,
                task_names=task_names,
                step_idx=step_idx,
                memory_mode=memory_mode,
                memory_size=memory_size,
                old_weight=old_weight,
                new_weight=new_weight,
            )

            merged_step_delta = OrderedDict()
            step_layer_stats = {} if (save_stats and save_step_layer_stats and save_each_step) else None

            for key in tqdm(target_keys, desc=f"Continual GCWM v2 step {step_idx+1}/{len(all_task_vectors)} [{task_name}]"):
                values = torch.stack([tv[key] for tv in active_task_vectors], dim=0).clone()
                merged_delta_k, stats = _gcwm_merge_one_layer(
                    key=key,
                    vectors=values,
                    iter_num=iter_num,
                    lr=inner_lr,
                    rank=rank,
                    energy_threshold=energy_threshold,
                    metric_mode=metric_mode,
                    bary_max_iter=bary_max_iter,
                    bary_tol=bary_tol,
                    bary_damping=bary_damping,
                    cov_reg=cov_reg,
                    gate_mode=gate_mode,
                    gate_threshold=gate_threshold,
                    gate_sharpness=gate_sharpness,
                    gate_min_alpha=gate_min_alpha,
                    gate_max_alpha=gate_max_alpha,
                    gate_skip_tol=gate_skip_tol,
                    task_weights=active_task_weights,
                    device=device,
                )
                merged_step_delta[key] = merged_delta_k

                if step_layer_stats is not None:
                    step_layer_stats[key] = stats

            delta_increment = _compute_increment_dict(merged_step_delta, prev_merged_delta)
            self.base_model = _apply_delta_to_model(
                delta_dict=delta_increment,
                pretrained_model=self.base_model,
                scaling_coef=step_coef,
            )

            step_dir = None
            if save_each_step:
                step_dir = os.path.join(self.save_path, f"step_{step_idx+1:02d}_{task_name}")
                os.makedirs(step_dir, exist_ok=True)
                self.base_model.save_pretrained(step_dir)
                self.tokenizer.save_pretrained(step_dir)

                if step_layer_stats is not None:
                    with open(os.path.join(step_dir, "gcwm_layer_stats.json"), "w") as f:
                        json.dump(step_layer_stats, f, indent=2)

            continual_stats["steps"].append({
                "step": step_idx + 1,
                "new_task": task_name,
                "active_task_names": active_task_names,
                "active_task_weights": active_task_weights,
                "num_active_tasks": len(active_task_vectors),
                "step_coef": step_coef,
                "save_dir": step_dir,
            })

            prev_merged_delta = merged_step_delta

        self.base_model.save_pretrained(self.save_path)
        self.tokenizer.save_pretrained(self.save_path)

        if save_stats:
            stats_path = os.path.join(self.save_path, "continual_gcwm_stats.json")
            with open(stats_path, "w") as f:
                json.dump(continual_stats, f, indent=2)

    def merge(self, **kwargs):
        continual = kwargs.get("continual", False)
        if continual:
            return self._merge_continual(**kwargs)
        return self._merge_batch(**kwargs)

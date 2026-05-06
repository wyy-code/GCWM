'''
CUDA_VISIBLE_DEVICES=1 python analysis/bridge_analyzer_grad.py \
  --base-model /path/to/base/model \
  --continual-stats /path/to/merged_model/continual_gcwm_stats.json \
  --dataset-file /path/to/mmlupro.parquet \
  --output-dir /path/to/bridge_grad_analysis \
  --examples-per-task 32 \
  --batch-size 4 \
  --max-length 2048 \
  --model-dtype bfloat16 \
  --grad-store-dtype float16 \
  --device cuda:0 \
  --layer-filter all \
  --task-map-json /path/to/task_map.json \
  --step-start 1 \
  --step-end 4
'''
import os
import re
import json
import math
import argparse
import random
from typing import Dict, List, Any, Optional, Tuple
import ast

import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


TARGET_KEYWORDS = (
    "q_proj.weight",
    "k_proj.weight",
    "v_proj.weight",
    "o_proj.weight",
    "gate_proj.weight",
    "up_proj.weight",
    "down_proj.weight",
)


# ============================
# Utilities
# ============================

def _safe_name(s: str) -> str:
    s = str(s)
    s = s.replace("/", "__").replace("\\", "__")
    s = re.sub(r"[^A-Za-z0-9_.\-]+", "_", s)
    return s[:220]


def _parse_csv(s: Optional[str]):
    if s is None:
        return None
    parts = [x.strip() for x in s.split(",")]
    return [x for x in parts if len(x) > 0]


def _infer_stats_path(run_dir_or_file: str) -> str:
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


def _load_stats(stats_path: str) -> Dict[str, Any]:
    with open(stats_path, "r") as f:
        return json.load(f)


def _write_jsonl(path: str, records: List[Dict[str, Any]]):
    with open(path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")


def _is_target_param_name(name: str, tensor: torch.Tensor) -> bool:
    if tensor is None or tensor.ndim != 2:
        return False
    return any(x in name for x in TARGET_KEYWORDS)


def _select_target_param_names(model: torch.nn.Module, layer_filter: str = "all") -> List[str]:
    names = []
    for n, p in model.named_parameters():
        if not _is_target_param_name(n, p):
            continue
        names.append(n)

    if layer_filter == "attention_only":
        names = [n for n in names if any(x in n for x in ("q_proj.weight", "k_proj.weight", "v_proj.weight", "o_proj.weight"))]
    elif layer_filter == "mlp_only":
        names = [n for n in names if any(x in n for x in ("gate_proj.weight", "up_proj.weight", "down_proj.weight"))]
    elif layer_filter == "all":
        pass
    else:
        raise ValueError(f"Unsupported layer_filter: {layer_filter}")
    return names


def _subsample_names(names: List[str], max_layers: Optional[int]) -> List[str]:
    if max_layers is None or max_layers <= 0 or len(names) <= max_layers:
        return names
    idxs = torch.linspace(0, len(names) - 1, steps=max_layers).long().tolist()
    return [names[i] for i in idxs]


# ============================
# Task dataset selection (MMLU-Pro style)
# ============================

class MMLUProDataset:
    """
    Minimal local-parquet loader for MMLU-Pro-style data.

    Expected columns (at least):
      - question
      - options
      - answer or answer_index
      - category and/or src
    """

    def __init__(
        self,
        dataset_file: str,
        task_filter_field: str = "category",
        task_map_json: Optional[str] = None,
        seed: int = 42,
    ):
        self.dataset_file = dataset_file
        self.task_filter_field = task_filter_field
        self.seed = seed
        self.df = pd.read_parquet(dataset_file)
        self.task_map = self._load_task_map(task_map_json)
        self.available_columns = set(self.df.columns)
        self.letters = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

    def _load_task_map(self, task_map_json: Optional[str]) -> Dict[str, Any]:
        if task_map_json is None:
            return {}
        with open(task_map_json, "r") as f:
            obj = json.load(f)
        return obj

    def _normalize_label(self, s: str) -> str:
        return str(s).strip().lower()

    def _auto_filter(self, task_label: str) -> pd.DataFrame:
        label = self._normalize_label(task_label)
        df = self.df

        # 1) exact match on requested field
        if self.task_filter_field in df.columns:
            series = df[self.task_filter_field].astype(str).str.lower()
            exact = df[series == label]
            if len(exact) > 0:
                return exact

        # 2) exact match on category / src
        for field in ["category", "src"]:
            if field in df.columns:
                series = df[field].astype(str).str.lower()
                exact = df[series == label]
                if len(exact) > 0:
                    return exact

        # 3) substring fallback on category / src
        for field in ["category", "src"]:
            if field in df.columns:
                series = df[field].astype(str).str.lower()
                sub = df[series.str.contains(label, na=False)]
                if len(sub) > 0:
                    return sub

        raise KeyError(
            f"Cannot auto-resolve task label '{task_label}' in dataset columns category/src; "
            f"please provide --task-map-json."
        )

    def _apply_filter_spec(self, spec: Dict[str, Any]) -> pd.DataFrame:
        df = self.df
        field = spec.get("field", self.task_filter_field)
        if field not in df.columns:
            raise KeyError(f"Filter field '{field}' not in dataset columns {list(df.columns)}")
        series = df[field].astype(str)
        out = df

        if "equals" in spec:
            vals = spec["equals"]
            if not isinstance(vals, list):
                vals = [vals]
            vals = [str(v).lower() for v in vals]
            out = out[out[field].astype(str).str.lower().isin(vals)]

        if "contains" in spec:
            vals = spec["contains"]
            if not isinstance(vals, list):
                vals = [vals]
            mask = pd.Series(False, index=out.index)
            for v in vals:
                mask = mask | out[field].astype(str).str.lower().str.contains(str(v).lower(), na=False)
            out = out[mask]

        return out

    def get_task_examples(self, task_label: str, num_examples: int) -> List[Dict[str, Any]]:
        if task_label in self.task_map:
            sub = self._apply_filter_spec(self.task_map[task_label])
        else:
            sub = self._auto_filter(task_label)

        if len(sub) == 0:
            raise ValueError(f"No examples found for task '{task_label}'")

        rng = random.Random(self.seed + hash(task_label) % 1000003)
        idxs = list(sub.index)
        rng.shuffle(idxs)
        idxs = idxs[: min(num_examples, len(idxs))]
        records = sub.loc[idxs].to_dict(orient="records")
        return records
    
    def _normalize_options(self, opts):
        """
        Make MMLU-Pro 'options' robust across parquet/pandas/pyarrow representations.

        Supports:
          - Python list / tuple
          - numpy / pandas / pyarrow objects with .tolist()
          - stringified list, e.g. '["A", "B", "C"]'
          - fallback iterable
        """
        if isinstance(opts, list):
            return [str(x) for x in opts]

        if isinstance(opts, tuple):
            return [str(x) for x in opts]

        # numpy / pandas / pyarrow-like objects
        if hasattr(opts, "tolist") and not isinstance(opts, str):
            try:
                x = opts.tolist()
                if isinstance(x, (list, tuple)):
                    return [str(v) for v in x]
            except Exception:
                pass

        # stringified list
        if isinstance(opts, str):
            s = opts.strip()
            if len(s) == 0:
                raise ValueError("MMLU-Pro record 'options' is an empty string")

            # try json-like parsing first
            try:
                x = json.loads(s)
                if isinstance(x, (list, tuple)):
                    return [str(v) for v in x]
            except Exception:
                pass

            # then python literal parsing
            try:
                x = ast.literal_eval(s)
                if isinstance(x, (list, tuple)):
                    return [str(v) for v in x]
            except Exception:
                pass

            raise ValueError(f"Could not parse MMLU-Pro 'options' string: {s[:120]}")

        # final fallback: generic iterable
        try:
            x = list(opts)
            return [str(v) for v in x]
        except Exception:
            raise ValueError(
                f"MMLU-Pro record 'options' has unsupported type: {type(opts)}"
            )

    def answer_letter(self, rec: Dict[str, Any]) -> str:
        ans = rec.get("answer", None)
        if isinstance(ans, str) and len(ans.strip()) == 1 and ans.strip().upper() in self.letters:
            return ans.strip().upper()
        ans_idx = rec.get("answer_index", None)
        if ans_idx is None:
            raise KeyError("Record has neither usable 'answer' letter nor 'answer_index'.")
        ans_idx = int(ans_idx)
        return self.letters[ans_idx]

    def build_prompt_target(self, rec: Dict[str, Any]) -> Tuple[str, str]:
        q = str(rec["question"])
        opts = self._normalize_options(rec["options"])

        if len(opts) == 0:
            raise ValueError("MMLU-Pro record 'options' is empty")

        if len(opts) > len(self.letters):
            raise ValueError(
                f"MMLU-Pro record has {len(opts)} options, exceeding supported letters"
            )

        letters = self.letters[: len(opts)]
        opt_lines = [f"{letters[i]}. {str(opt)}" for i, opt in enumerate(opts)]

        prompt = (
            "Answer the multiple-choice question by giving the correct option letter.\n\n"
            f"Question: {q}\n"
            "Options:\n" + "\n".join(opt_lines) + "\n"
            "Answer:"
        )
        target = " " + self.answer_letter(rec)
        return prompt, target


# ============================
# Tokenization / batches
# ============================

def _build_causal_lm_batch(tokenizer, prompts: List[str], targets: List[str], max_length: int, device: torch.device):
    input_ids_list = []
    labels_list = []
    attn_list = []

    for prompt, target in zip(prompts, targets):
        p_ids = tokenizer(prompt, add_special_tokens=False).input_ids
        t_ids = tokenizer(target, add_special_tokens=False).input_ids
        if len(t_ids) == 0:
            raise ValueError("Target tokenization produced empty ids; check tokenizer / answer formatting.")

        # Keep target intact; truncate prompt from the left if needed.
        max_prompt_len = max_length - len(t_ids)
        if max_prompt_len <= 0:
            t_ids = t_ids[-max_length:]
            p_ids = []
        elif len(p_ids) > max_prompt_len:
            p_ids = p_ids[-max_prompt_len:]

        ids = p_ids + t_ids
        labels = [-100] * len(p_ids) + t_ids
        attn = [1] * len(ids)

        input_ids_list.append(ids)
        labels_list.append(labels)
        attn_list.append(attn)

    pad_id = tokenizer.pad_token_id
    max_len = max(len(x) for x in input_ids_list)
    padded_input_ids = []
    padded_labels = []
    padded_attn = []
    for ids, labels, attn in zip(input_ids_list, labels_list, attn_list):
        pad_len = max_len - len(ids)
        padded_input_ids.append(ids + [pad_id] * pad_len)
        padded_labels.append(labels + [-100] * pad_len)
        padded_attn.append(attn + [0] * pad_len)

    batch = {
        "input_ids": torch.tensor(padded_input_ids, dtype=torch.long, device=device),
        "attention_mask": torch.tensor(padded_attn, dtype=torch.long, device=device),
        "labels": torch.tensor(padded_labels, dtype=torch.long, device=device),
    }
    return batch


# ============================
# Gradient cache / computation
# ============================

class GradientCache:
    def __init__(self, cache_dir: Optional[str] = None):
        self.cache_dir = cache_dir
        if cache_dir is not None:
            os.makedirs(cache_dir, exist_ok=True)

    def _cache_file(self, step: int, task_label: str) -> Optional[str]:
        if self.cache_dir is None:
            return None
        return os.path.join(self.cache_dir, f"step_{step:02d}__{_safe_name(task_label)}.pt")

    def maybe_load(self, step: int, task_label: str):
        path = self._cache_file(step, task_label)
        if path is None or not os.path.exists(path):
            return None
        return torch.load(path, map_location="cpu")

    def maybe_save(self, step: int, task_label: str, obj: Dict[str, torch.Tensor]):
        path = self._cache_file(step, task_label)
        if path is None:
            return
        torch.save(obj, path)


def _compute_task_gradients(
    model: torch.nn.Module,
    tokenizer,
    records: List[Dict[str, Any]],
    dataset_helper: MMLUProDataset,
    target_param_names: List[str],
    batch_size: int,
    max_length: int,
    device: torch.device,
    grad_store_dtype: str = "float16",
) -> Tuple[Dict[str, torch.Tensor], Dict[str, float]]:
    """
    Returns:
      grad_dict[name] = average gradient tensor on CPU in grad_store_dtype
      meta[name__norm] etc.
    """
    model.eval()
    name_to_param = {n: p for n, p in model.named_parameters() if n in target_param_names}
    params = [name_to_param[n] for n in target_param_names]

    grad_dtype = getattr(torch, grad_store_dtype)
    accum = {n: None for n in target_param_names}
    num_micro = 0
    num_examples = 0

    for start in range(0, len(records), batch_size):
        chunk = records[start:start + batch_size]
        prompts = []
        targets = []
        for rec in chunk:
            p, t = dataset_helper.build_prompt_target(rec)
            prompts.append(p)
            targets.append(t)

        batch = _build_causal_lm_batch(tokenizer, prompts, targets, max_length=max_length, device=device)
        outputs = model(**batch)
        loss = outputs.loss

        grads = torch.autograd.grad(
            loss,
            params,
            retain_graph=False,
            create_graph=False,
            allow_unused=True,
        )

        for n, g in zip(target_param_names, grads):
            if g is None:
                g_cpu = torch.zeros_like(name_to_param[n], device="cpu", dtype=grad_dtype)
            else:
                g_cpu = g.detach().to("cpu", dtype=grad_dtype)
            if accum[n] is None:
                accum[n] = g_cpu.clone()
            else:
                accum[n] += g_cpu

        num_micro += 1
        num_examples += len(chunk)
        del batch, outputs, loss, grads
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if num_micro == 0:
        raise ValueError("No microbatches were processed for gradient computation")

    for n in accum:
        accum[n] = accum[n] / float(num_micro)

    meta = {"num_examples": int(num_examples), "num_microbatches": int(num_micro)}
    return accum, meta


# ============================
# Pairwise grad metrics
# ============================

def _grad_cos_metrics(g_i: torch.Tensor, g_j: torch.Tensor, eps: float = 1e-12) -> Dict[str, Any]:
    x = g_i.float().reshape(-1)
    y = g_j.float().reshape(-1)
    dot = torch.dot(x, y)
    nx = torch.norm(x, p=2)
    ny = torch.norm(y, p=2)
    denom = torch.clamp(nx * ny, min=eps)
    cos = dot / denom
    neg = bool(cos.item() < 0)
    return {
        "grad_cos": float(cos.item()),
        "negative_conflict": neg,
        "grad_dot": float(dot.item()),
        "grad_norm_i": float(nx.item()),
        "grad_norm_j": float(ny.item()),
    }


# ============================
# Main analyzer
# ============================

def run_gradient_bridge_analysis(
    base_model: str,
    continual_stats_path: str,
    output_dir: str,
    dataset_file: str,
    task_labels: Optional[List[str]] = None,
    task_map_json: Optional[str] = None,
    task_filter_field: str = "category",
    examples_per_task: int = 16,
    batch_size: int = 4,
    max_length: int = 1024,
    model_dtype: str = "bfloat16",
    grad_store_dtype: str = "float16",
    device: str = "cpu",
    layer_filter: str = "all",
    max_layers: Optional[int] = None,
    step_start: Optional[int] = None,
    step_end: Optional[int] = None,
    seed: int = 42,
    grad_cache_dir: Optional[str] = None,
):
    os.makedirs(output_dir, exist_ok=True)
    stats = _load_stats(continual_stats_path)
    steps_meta = stats["steps"]

    # Step slicing: 1-indexed inclusive
    total_steps = len(steps_meta)
    if step_start is None:
        step_start = 1
    if step_end is None:
        step_end = total_steps
    if step_start < 1 or step_end > total_steps or step_start > step_end:
        raise ValueError(f"Invalid step range [{step_start}, {step_end}] for total_steps={total_steps}")
    sliced_steps = steps_meta[step_start - 1: step_end]

    stats_task_names = stats.get("task_names", None)
    if task_labels is None:
        task_labels = stats_task_names
    if task_labels is None:
        raise ValueError("task_labels could not be inferred; pass --task-labels or ensure stats has task_names.")

    dataset_helper = MMLUProDataset(
        dataset_file=dataset_file,
        task_filter_field=task_filter_field,
        task_map_json=task_map_json,
        seed=seed,
    )

    grad_cache = GradientCache(grad_cache_dir)

    pairwise_records = []
    step_records = []

    step_bar = tqdm(sliced_steps, desc="bridge-grad steps")
    for step in step_bar:
        step_idx = int(step["step"]) - 1
        new_task = step.get("new_task", f"step_{step_idx+1}")
        step_save_dir = step.get("save_dir", None)
        if step_save_dir is None:
            raise ValueError(f"Step {step_idx+1} has no save_dir; analyzer expects saved checkpoints.")

        step_bar.set_postfix({"new_task": new_task, "stage": "load_model"})
        model = AutoModelForCausalLM.from_pretrained(
            step_save_dir,
            torch_dtype=getattr(torch, model_dtype),
            low_cpu_mem_usage=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(step_save_dir)
        if tokenizer.pad_token_id is None:
            if tokenizer.eos_token_id is not None:
                tokenizer.pad_token = tokenizer.eos_token
            else:
                tokenizer.add_special_tokens({"pad_token": "<pad>"})
                model.resize_token_embeddings(len(tokenizer))

        dev = torch.device(device)
        model.to(dev)
        model.eval()

        all_target_param_names = _select_target_param_names(model, layer_filter=layer_filter)
        target_param_names = _subsample_names(all_target_param_names, max_layers=max_layers)
        if len(target_param_names) == 0:
            raise ValueError("No target parameter names selected for gradient analysis.")

        active_names = step.get("active_task_names", [])
        active_weights = step.get("active_task_weights", None)

        # Compute or load gradients for each active task at the current merged checkpoint
        task_gradients = {}
        task_grad_meta = {}
        task_bar = tqdm(active_names, desc=f"grad_tasks::{step_idx+1}", leave=False)
        for task_label in task_bar:
            task_bar.set_postfix({"task": task_label})
            cached = grad_cache.maybe_load(step_idx + 1, task_label)
            if cached is not None:
                task_gradients[task_label] = cached["grads"]
                task_grad_meta[task_label] = cached.get("meta", {})
                continue

            recs = dataset_helper.get_task_examples(task_label, num_examples=examples_per_task)
            grad_dict, grad_meta = _compute_task_gradients(
                model=model,
                tokenizer=tokenizer,
                records=recs,
                dataset_helper=dataset_helper,
                target_param_names=target_param_names,
                batch_size=batch_size,
                max_length=max_length,
                device=dev,
                grad_store_dtype=grad_store_dtype,
            )
            task_gradients[task_label] = grad_dict
            task_grad_meta[task_label] = grad_meta
            grad_cache.maybe_save(step_idx + 1, task_label, {"grads": grad_dict, "meta": grad_meta})

        # Pairwise layer metrics
        step_pair_cos = []
        step_neg_flags = []
        step_neg_only = []
        layer_bar = tqdm(target_param_names, desc=f"grad_pairs::{step_idx+1}", leave=False)
        for layer_key in layer_bar:
            for i in range(len(active_names)):
                for j in range(i + 1, len(active_names)):
                    name_i = active_names[i]
                    name_j = active_names[j]
                    gm = _grad_cos_metrics(
                        task_gradients[name_i][layer_key],
                        task_gradients[name_j][layer_key],
                    )
                    pairwise_records.append({
                        "step": step_idx + 1,
                        "new_task": new_task,
                        "context": "active_pair_grad",
                        "layer_key": layer_key,
                        "task_i": name_i,
                        "task_j": name_j,
                        "active_task_weights": active_weights,
                        **gm,
                    })
                    step_pair_cos.append(gm["grad_cos"])
                    step_neg_flags.append(1 if gm["negative_conflict"] else 0)
                    if gm["negative_conflict"]:
                        step_neg_only.append(gm["grad_cos"])

        step_record = {
            "step": step_idx + 1,
            "new_task": new_task,
            "num_active_tasks": len(active_names),
            "active_task_names": active_names,
            "active_task_weights": active_weights,
            "step_coef": step.get("step_coef", None),
            "save_dir": step_save_dir,
            "num_target_layers": len(target_param_names),
            "layer_filter": layer_filter,
            "mean_active_pair_grad_cos": float(sum(step_pair_cos) / max(len(step_pair_cos), 1)),
            "min_active_pair_grad_cos": float(min(step_pair_cos)) if len(step_pair_cos) > 0 else None,
            "max_active_pair_grad_cos": float(max(step_pair_cos)) if len(step_pair_cos) > 0 else None,
            "neg_conflict_ratio": float(sum(step_neg_flags) / max(len(step_neg_flags), 1)),
            "mean_negative_conflict_cos": float(sum(step_neg_only) / max(len(step_neg_only), 1)) if len(step_neg_only) > 0 else None,
            "task_grad_meta": task_grad_meta,
        }
        step_records.append(step_record)

        with open(os.path.join(output_dir, f"step_{step_idx+1:02d}_grad_detail.json"), "w") as f:
            json.dump(step_record, f, indent=2)

        del model
        if dev.type == "cuda":
            torch.cuda.empty_cache()

    _write_jsonl(os.path.join(output_dir, "pairwise_layer_grad_metrics.jsonl"), pairwise_records)
    _write_jsonl(os.path.join(output_dir, "step_grad_metrics.jsonl"), step_records)

    summary = {
        "base_model": base_model,
        "continual_stats_path": continual_stats_path,
        "dataset_file": dataset_file,
        "task_labels": task_labels,
        "task_filter_field": task_filter_field,
        "task_map_json": task_map_json,
        "examples_per_task": examples_per_task,
        "batch_size": batch_size,
        "max_length": max_length,
        "model_dtype": model_dtype,
        "grad_store_dtype": grad_store_dtype,
        "device": str(device),
        "layer_filter": layer_filter,
        "max_layers": max_layers,
        "step_start": step_start,
        "step_end": step_end,
        "num_steps": len(step_records),
        "pairwise_metrics_path": os.path.join(output_dir, "pairwise_layer_grad_metrics.jsonl"),
        "step_metrics_path": os.path.join(output_dir, "step_grad_metrics.jsonl"),
        "grad_cache_dir": grad_cache_dir,
    }
    with open(os.path.join(output_dir, "grad_analysis_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    return summary


# ============================
# CLI
# ============================

def build_parser():
    p = argparse.ArgumentParser("bridge_analyzer_grad")
    p.add_argument("--base-model", type=str, required=True)
    p.add_argument("--continual-stats", type=str, required=True,
                   help="Path to continual_*_stats.json or run directory containing it")
    p.add_argument("--output-dir", type=str, required=True)

    # Dataset / task mapping
    p.add_argument("--dataset-file", type=str, required=True,
                   help="Local parquet file for MMLU-Pro-style data (e.g., validation-00000-of-00001.parquet)")
    p.add_argument("--task-labels", type=str, default=None,
                   help="Comma-separated labels matching stats.task_names; usually omitted to use stats.task_names")
    p.add_argument("--task-map-json", type=str, default=None,
                   help="Optional JSON mapping task labels to dataset filter specs")
    p.add_argument("--task-filter-field", type=str, default="category")

    # Gradient probe config
    p.add_argument("--examples-per-task", type=int, default=16)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--max-length", type=int, default=1024)
    p.add_argument("--model-dtype", type=str, default="bfloat16", choices=["float16", "bfloat16", "float32"])
    p.add_argument("--grad-store-dtype", type=str, default="float16", choices=["float16", "bfloat16", "float32"])
    p.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))

    # Layer scope
    p.add_argument("--layer-filter", type=str, default="all", choices=["all", "attention_only", "mlp_only"])
    p.add_argument("--max-layers", type=int, default=None)

    # Sharding / cache
    p.add_argument("--step-start", type=int, default=None)
    p.add_argument("--step-end", type=int, default=None)
    p.add_argument("--grad-cache-dir", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)
    return p


def main():
    args = build_parser().parse_args()
    continual_stats_path = _infer_stats_path(args.continual_stats)
    task_labels = _parse_csv(args.task_labels)
    summary = run_gradient_bridge_analysis(
        base_model=args.base_model,
        continual_stats_path=continual_stats_path,
        output_dir=args.output_dir,
        dataset_file=args.dataset_file,
        task_labels=task_labels,
        task_map_json=args.task_map_json,
        task_filter_field=args.task_filter_field,
        examples_per_task=args.examples_per_task,
        batch_size=args.batch_size,
        max_length=args.max_length,
        model_dtype=args.model_dtype,
        grad_store_dtype=args.grad_store_dtype,
        device=args.device,
        layer_filter=args.layer_filter,
        max_layers=args.max_layers,
        step_start=args.step_start,
        step_end=args.step_end,
        seed=args.seed,
        grad_cache_dir=args.grad_cache_dir,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

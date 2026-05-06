import argparse


def create_parser():
    parser = argparse.ArgumentParser(description="Configuration for GCWM continual model merging")

    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--scaling-coef", default=1.0, type=float)
    parser.add_argument("--save_group", type=str, default=None)
    parser.add_argument("--task_names", type=str, default=None)
    parser.add_argument("--save-stats", action="store_true")

    parser.add_argument("--continual", action="store_true")
    parser.add_argument(
        "--task-order",
        type=str,
        default=None,
        help="Comma-separated task order. Items may be task names, model IDs, or explicit expert paths.",
    )
    parser.add_argument(
        "--memory-mode",
        type=str,
        default="all_history",
        choices=["all_history", "current_anchor"],
    )
    parser.add_argument(
        "--memory-size",
        type=int,
        default=-1,
        help="For all_history mode, keep only the most recent N tasks; -1 means unlimited.",
    )
    parser.add_argument("--save-each-step", action="store_true")
    parser.add_argument(
        "--continual-step-coef",
        type=float,
        default=None,
        help="Outer coefficient applied at each continual step; defaults to scaling_coef if omitted.",
    )
    parser.add_argument("--step-decay", type=float, default=1.0)
    parser.add_argument("--old-weight", type=float, default=1.0)
    parser.add_argument("--new-weight", type=float, default=1.0)

    parser.add_argument("--gcwm-lr", type=float, default=1e-5)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--energy-threshold", type=float, default=0.9)
    parser.add_argument(
        "--metric-mode",
        type=str,
        default="barycenter",
        choices=["mean", "barycenter", "dense_mean"],
    )
    parser.add_argument("--bary-max-iter", type=int, default=50)
    parser.add_argument("--bary-tol", type=float, default=1e-5)
    parser.add_argument("--bary-damping", type=float, default=1.0)
    parser.add_argument("--cov-reg", type=float, default=1e-6)
    parser.add_argument("--gate-mode", type=str, default="conflict", choices=["none", "conflict"])
    parser.add_argument("--gate-threshold", type=float, default=0.12)
    parser.add_argument("--gate-sharpness", type=float, default=12.0)
    parser.add_argument("--gate-min-alpha", type=float, default=0.15)
    parser.add_argument("--gate-max-alpha", type=float, default=0.95)
    parser.add_argument("--gate-skip-tol", type=float, default=0.05)
    parser.add_argument("--save-step-layer-stats", action="store_true")
    parser.add_argument("--save-layer-stats", action="store_true")

    return parser


def _parse_task_order(task_order_str):
    if task_order_str is None:
        return None
    parts = [x.strip() for x in task_order_str.split(",")]
    parts = [x for x in parts if len(x) > 0]
    return parts if len(parts) > 0 else None


def _inject_common_continual_kwargs(kwargs, params):
    kwargs["save_stats"] = params.save_stats
    kwargs["continual"] = params.continual
    kwargs["task_order_list"] = _parse_task_order(params.task_order)
    kwargs["memory_mode"] = params.memory_mode
    kwargs["memory_size"] = params.memory_size
    kwargs["save_each_step"] = params.save_each_step
    kwargs["continual_step_coef"] = params.continual_step_coef
    kwargs["step_decay"] = params.step_decay
    kwargs["old_weight"] = params.old_weight
    kwargs["new_weight"] = params.new_weight
    return kwargs


def prepare_args(params):
    if params.algo != "GCWM":
        raise ValueError(f"Unsupported merging method: {params.algo}")

    kwargs = {
        "scaling_coef": params.scaling_coef,
        "iter_num": params.iter_num,
        "gcwm_lr": params.gcwm_lr,
        "device": params.device,
        "rank": params.rank,
        "energy_threshold": params.energy_threshold,
        "metric_mode": params.metric_mode,
        "bary_max_iter": params.bary_max_iter,
        "bary_tol": params.bary_tol,
        "bary_damping": params.bary_damping,
        "cov_reg": params.cov_reg,
        "gate_mode": params.gate_mode,
        "gate_threshold": params.gate_threshold,
        "gate_sharpness": params.gate_sharpness,
        "gate_min_alpha": params.gate_min_alpha,
        "gate_max_alpha": params.gate_max_alpha,
        "gate_skip_tol": params.gate_skip_tol,
        "save_step_layer_stats": params.save_step_layer_stats,
        "save_layer_stats": params.save_layer_stats,
    }
    return _inject_common_continual_kwargs(kwargs, params)

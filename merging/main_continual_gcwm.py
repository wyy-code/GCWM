from prepare_args_gcwm import prepare_args, create_parser
import importlib
import time
import os
from pathlib import Path

DIR = os.environ.get("GCWM_ROOT", str(Path(__file__).resolve().parents[1]))


def _default_task_order():
    return ['instruction', 'math', 'coding', 'safety', 'multilingual']


def _looks_like_explicit_model_path(item: str) -> bool:
    if item is None:
        return False
    if '/' in item or '\\' in item:
        return True
    if os.path.exists(item):
        return True
    return False


def _safe_label_from_ckpt(ckpt: str) -> str:
    base = os.path.basename(ckpt.rstrip("/"))
    if len(base) == 0:
        base = ckpt.replace("/", "_").replace("\\", "_")
    return base


def get_ft_ckpts_and_labels(base_model, task_order=None):
    model_name = base_model.split('/')[-1]

    if task_order is None or len(task_order) == 0:
        task_order = _default_task_order()

    ft_ckpts = []
    task_labels = []

    for item in task_order:
        if _looks_like_explicit_model_path(item):
            ckpt = item
            label = _safe_label_from_ckpt(item)
        else:
            ckpt = f'{DIR}/models/{model_name}_{item}'
            label = item

        ft_ckpts.append(ckpt)
        task_labels.append(label)

    return ft_ckpts, task_labels


def parse_args():
    parser = create_parser()
    parser.add_argument('--base-model', default='meta-llama/Llama-3.2-3B', type=str)
    parser.add_argument('--algo', default='GCWM', type=str, choices=['GCWM'])
    parser.add_argument('--save-path', default='./merged_models/', type=str)
    parser.add_argument('--iter-num', type=int, default=100)
    parser.add_argument('--device', type=str, default='cuda')
    return parser.parse_args()


def main(args):
    kwargs = prepare_args(args)
    merger_module = importlib.import_module("merging_methods")

    requested_order = kwargs.get("task_order_list", None)
    ft_ckpts, task_labels = get_ft_ckpts_and_labels(args.base_model, task_order=requested_order)

    kwargs["task_order_list"] = task_labels

    print(f"Merging models: {ft_ckpts}")
    print(f"Task labels: {task_labels}")

    kwargs_str = "_".join(
        f"{key}_{value}"
        for key, value in kwargs.items()
        if key not in ['fisher_only', 'merge_only', 'save_group', 'task_names', 'keep_checkpoints', 'task_order_list']
    )

    if args.save_group:
        task_group = args.save_group
    elif args.task_names:
        task_group = args.task_names
    else:
        task_group = None

    base_label = os.path.basename(args.base_model.rstrip("/")) or "base_model"
    save_path = os.path.join(args.save_path, f"{base_label}_merged", args.algo)
    if kwargs.get("continual", False):
        save_path += "_continual_v2"

    if task_group:
        print("---------------------------------------------------")
        print("The first choice..")
        save_path += '_task_names_' + task_group
    if kwargs_str != '':
        print("---------------------------------------------------")
        print("The second choice..")
        save_path += '_' + kwargs_str[:48]
        print(kwargs_str[:48])

    start_time = time.time()

    merger = getattr(merger_module, args.algo)(args.base_model, ft_ckpts, save_path)
    print(args)
    print(kwargs)
    merger.merge(**kwargs)
    print(f"The total time cost is {time.time()-start_time}s.")


if __name__ == "__main__":
    args = parse_args()
    main(args)

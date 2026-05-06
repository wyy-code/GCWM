'''
python analysis/merge_bridge_step_outputs.py --run-dir /path/to/bridge_fast_outputs/run_name
'''

import json
import shutil
from pathlib import Path


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(obj, path: Path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def discover_step_dirs(run_path: Path):
    # Prefer top-level step_* dirs
    top_steps = sorted(
        [p for p in run_path.iterdir() if p.is_dir() and p.name.startswith("step_")],
        key=lambda x: x.name,
    )
    top_steps = [p for p in top_steps if (p / "step_metrics.jsonl").exists()]

    if top_steps:
        return top_steps, "top_level"

    # Fallback to parts/step_*
    parts_dir = run_path / "parts"
    if parts_dir.exists():
        part_steps = sorted(
            [p for p in parts_dir.iterdir() if p.is_dir() and p.name.startswith("step_")],
            key=lambda x: x.name,
        )
        part_steps = [p for p in part_steps if (p / "step_metrics.jsonl").exists()]
        if part_steps:
            return part_steps, "parts"

    raise RuntimeError(
        f"No valid step_* dirs with step_metrics.jsonl found under {run_path}"
    )


def merge_bridge_outputs(run_dir: str, output_subdir: str = "bridge_analysis_merged", overwrite: bool = True):
    run_path = Path(run_dir)
    step_dirs, source_mode = discover_step_dirs(run_path)

    out_dir = run_path / output_subdir
    if out_dir.exists() and overwrite:
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    merged_step_metrics = []
    merged_pairwise_lines = []
    merged_summaries = []

    num_target_layers = None
    layer_filter = None
    merged_steps = []

    for step_dir in step_dirs:
        summary_path = step_dir / "bridge_analysis_summary.json"
        step_metrics_path = step_dir / "step_metrics.jsonl"
        pairwise_path = step_dir / "pairwise_layer_metrics.jsonl"

        if not step_metrics_path.exists():
            print(f"[WARN] missing {step_metrics_path}, skip")
            continue

        if summary_path.exists():
            summary = load_json(summary_path)
            merged_summaries.append(summary)
            if num_target_layers is None:
                num_target_layers = summary.get("num_target_layers")
            if layer_filter is None:
                layer_filter = summary.get("layer_filter")

        with open(step_metrics_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    obj = json.loads(line)
                    merged_step_metrics.append(obj)
                    if "step" in obj:
                        merged_steps.append(obj["step"])

        if pairwise_path.exists():
            with open(pairwise_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        merged_pairwise_lines.append(line)

        for detail_file in step_dir.glob("step_*_bridge_detail.json"):
            shutil.copy2(detail_file, out_dir / detail_file.name)

    merged_step_metrics = sorted(merged_step_metrics, key=lambda x: x.get("step", 10**9))
    merged_steps = sorted(set(merged_steps))

    with open(out_dir / "step_metrics.jsonl", "w", encoding="utf-8") as f:
        for obj in merged_step_metrics:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    with open(out_dir / "pairwise_layer_metrics.jsonl", "w", encoding="utf-8") as f:
        for line in merged_pairwise_lines:
            f.write(line + "\n")

    final_summary = {
        "source_run_dir": str(run_path),
        "source_mode": source_mode,
        "merged_from_step_dirs": [str(p) for p in step_dirs],
        "num_step_dirs_found": len(step_dirs),
        "num_steps_in_step_metrics": len(merged_step_metrics),
        "step_ids": merged_steps,
        "num_target_layers": num_target_layers,
        "layer_filter": layer_filter,
    }
    dump_json(final_summary, out_dir / "bridge_analysis_summary.json")

    print(f"[OK] merged bridge outputs written to: {out_dir}")
    print(f"[OK] source_mode = {source_mode}")
    print(f"[OK] step_ids = {merged_steps}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=str, required=True)
    parser.add_argument("--output-subdir", type=str, default="bridge_analysis_merged")
    parser.add_argument("--no-overwrite", action="store_true")
    args = parser.parse_args()

    merge_bridge_outputs(
        run_dir=args.run_dir,
        output_subdir=args.output_subdir,
        overwrite=not args.no_overwrite,
    )

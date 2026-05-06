#!/usr/bin/env python3

'''
python generate_task_map_from_stats.py \
  --continual-stats /path/to/continual_gcwm_stats.json \
  --output-json /path/to/task_map.json

如果你的数据集过滤字段不是 category，可以改成：

python generate_task_map_from_stats.py \
  --continual-stats /path/to/continual_gcwm_stats.json \
  --output-json /path/to/task_map.json \
  --field src
'''

import argparse
import json
import os
import re
from typing import Dict, List


def load_json(path: str):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def dump_json(obj, path: str):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def strip_numeric_prefix(label: str) -> str:
    # 01_biology -> biology
    # step_01_01_biology -> biology (after basename handling if needed)
    s = os.path.basename(str(label).rstrip('/'))
    s = re.sub(r'^step_\d+_', '', s)
    s = re.sub(r'^\d+_', '', s)
    return s


def label_to_category(label: str) -> str:
    core = strip_numeric_prefix(label)
    core = core.replace('__', '_')
    core = core.replace('_', ' ')
    core = re.sub(r'\s+', ' ', core).strip().lower()
    return core


def build_task_map(task_names: List[str], field: str = 'category') -> Dict[str, Dict]:
    task_map = {}
    for name in task_names:
        category = label_to_category(name)
        task_map[name] = {
            'field': field,
            'equals': category,
        }
    return task_map


def main():
    ap = argparse.ArgumentParser(description='Generate task_map.json from continual_*_stats.json')
    ap.add_argument('--continual-stats', required=True, help='Path to continual_*_stats.json')
    ap.add_argument('--output-json', required=True, help='Output path for task_map.json')
    ap.add_argument('--field', default='category', help='Dataset field to filter on, default: category')
    args = ap.parse_args()

    stats = load_json(args.continual_stats)
    task_names = stats.get('task_names')
    if not isinstance(task_names, list) or len(task_names) == 0:
        raise ValueError('No valid task_names found in continual stats json')

    task_map = build_task_map(task_names, field=args.field)
    dump_json(task_map, args.output_json)
    print(f'Wrote task_map with {len(task_map)} tasks to: {args.output_json}')


if __name__ == '__main__':
    main()

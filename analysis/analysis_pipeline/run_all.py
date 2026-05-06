#!/usr/bin/env python3
import argparse
import subprocess
import sys
from pathlib import Path


def run(cmd):
    print("[RUN]", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)



def main():
    parser = argparse.ArgumentParser(description="Run full analysis pipeline.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    py = sys.executable
    run([py, str(root / "prepare_downstream.py"), "--config", args.config])
    run([py, str(root / "merge_mechanism_tables.py"), "--config", args.config])
    run([py, str(root / "build_analysis_dataset.py"), "--config", args.config])
    run([py, str(root / "plot_analysis_report.py"), "--config", args.config])


if __name__ == "__main__":
    main()

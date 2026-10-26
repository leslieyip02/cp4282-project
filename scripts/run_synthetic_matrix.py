"""Enumerate or run the 7-method x 3-seed array for one dataset path."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
METHODS = ("none", "downsample", "mean", "gaussian", "bilateral", "kuwahara", "unsharp")
SEEDS = (7, 17, 27)


def jobs():
    return [(method, seed) for method in METHODS for seed in SEEDS]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, help="One NeRF-synthetic dataset directory")
    parser.add_argument("--name", help="Output label; defaults to the dataset directory name")
    parser.add_argument("--config", type=Path, default=ROOT / "config/synthetic.yaml")
    parser.add_argument("--index", type=int, help="Run this zero-based array task")
    parser.add_argument("--list", action="store_true", help="Print the 21 method/seed pairs")
    parser.add_argument("--arch", choices=("cpu", "gpu"), default="gpu")
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs")
    args = parser.parse_args()
    matrix = jobs()
    if args.list:
        for index, (method, seed) in enumerate(matrix):
            print(f"{index}\t{method}\t{seed}")
        return
    if args.data is None or not args.data.is_dir():
        parser.error("--data must point to an existing scene directory")
    if args.index is None or not 0 <= args.index < len(matrix):
        parser.error(f"--index must be between 0 and {len(matrix) - 1}; use --list to inspect jobs")
    method, seed = matrix[args.index]
    command = [
        sys.executable, str(ROOT / "synthetic_experiment.py"),
        "--config", str(args.config),
        "--data", str(args.data),
        "--method", method, "--seed", str(seed),
        "--arch", args.arch,
        "--output-root", str(args.output_root),
    ]
    if args.name:
        command += ["--scene", args.name]
    print(" ".join(command), flush=True)
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()

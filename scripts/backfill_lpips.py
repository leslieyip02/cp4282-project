"""Add VGG LPIPS to an existing kernel run without retraining."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
from tempfile import TemporaryDirectory


ROOT = Path(__file__).resolve().parents[1]


def backfill(run_dir: Path, data: Path, arch: str) -> None:
    config = json.loads((run_dir / "run_config.json").read_text())
    summary_path = run_dir / "test_summary.json"
    metrics_path = run_dir / "test_metrics.csv"
    summary = json.loads(summary_path.read_text())
    with metrics_path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        fields = list(reader.fieldnames or ())
        rows = list(reader)
    if not rows or "frame" not in fields:
        raise ValueError(f"No per-view test metrics in {metrics_path}")
    if (summary.get("lpips_net") == "vgg" and summary.get("mean_lpips") is not None
            and "lpips" in fields and all(row.get("lpips") for row in rows)):
        print(f"Already has VGG LPIPS: {run_dir}")
        return
    with TemporaryDirectory(prefix="cp4282-lpips-") as directory:
        temporary = Path(directory)
        command = [
            sys.executable, str(ROOT / "3dgs_evaluator.py"),
            str(run_dir / "final.ply"), str(data),
            "--arch", arch,
            "--width", str(config["model"]["width"]),
            "--height", str(config["model"]["height"]),
            "--background", config["model"]["background"],
            "--split", "test", "--metrics", "lpips", "--lpips-net", "vgg",
            "--output-dir", str(temporary / "renders"),
            "--csv", str(temporary / "lpips.csv"),
        ]
        subprocess.run(command, check=True)
        with (temporary / "lpips.csv").open(newline="") as stream:
            lpips_rows = list(csv.DictReader(stream))
    if [str(row["frame"]) for row in rows] != [str(row["file"]) for row in lpips_rows]:
        raise ValueError(f"Test views differ from the original evaluation for {run_dir}")
    if "lpips" not in fields:
        fields.insert(fields.index("ssim") + 1 if "ssim" in fields else len(fields), "lpips")
    values = [float(row["lpips"]) for row in lpips_rows]
    for row, value in zip(rows, values):
        row["lpips"] = value
    pending_metrics = run_dir / "test_metrics.csv.tmp"
    with pending_metrics.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    summary["mean_lpips"] = statistics.mean(values)
    summary["lpips_net"] = "vgg"
    pending_summary = run_dir / "test_summary.json.tmp"
    pending_summary.write_text(json.dumps(summary, indent=2) + "\n")
    os.replace(pending_metrics, metrics_path)
    os.replace(pending_summary, summary_path)
    print(f"Updated {run_dir}: mean LPIPS (VGG)={summary['mean_lpips']:.5f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("data", type=Path)
    parser.add_argument("--arch", choices=("auto", "cpu", "gpu"), default="auto")
    args = parser.parse_args()
    backfill(args.run_dir.resolve(), args.data.resolve(), args.arch)


if __name__ == "__main__":
    main()

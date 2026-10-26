"""Summarize synthetic runs with same-scene, same-seed baseline comparisons."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import statistics

from scripts.run_synthetic_matrix import METHODS, SEEDS


def write_csv(path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=Path(__file__).resolve().parent / "outputs")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "outputs/analysis")
    parser.add_argument("--scenes", nargs="+", help="Dataset labels; defaults to discovered output directories")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    args = parser.parse_args()

    if args.scenes is not None:
        scenes = args.scenes
    elif args.input_root.exists():
        scenes = sorted(
            path.name for path in args.input_root.iterdir()
            if path.is_dir() and any((path / method).is_dir() for method in METHODS)
        )
    else:
        scenes = []

    results = {}
    curves = {}
    missing = []
    for scene in scenes:
        for method in METHODS:
            for seed in args.seeds:
                directory = args.input_root / scene / method / f"seed_{seed}"
                summary_path = directory / "test_summary.json"
                config_path = directory / "run_config.json"
                validation_path = directory / "validation_metrics.csv"
                if not all(path.exists() for path in (summary_path, config_path, validation_path)):
                    missing.append((scene, method, seed))
                    continue
                summary = json.loads(summary_path.read_text())
                config = json.loads(config_path.read_text())
                if summary.get("completed_iterations") != config["training"]["iterations"]:
                    missing.append((scene, method, seed))
                    continue
                results[(scene, method, seed)] = summary
                by_iteration = defaultdict(list)
                with validation_path.open(newline="") as stream:
                    for row in csv.DictReader(stream):
                        by_iteration[int(row["iteration"])].append(row)
                for iteration, rows in by_iteration.items():
                    curves[(scene, method, seed, iteration)] = {
                        "psnr_db": statistics.mean(float(row["psnr_db"]) for row in rows),
                        "ssim": statistics.mean(float(row["ssim"]) for row in rows),
                        "elapsed_seconds": float(rows[0]["elapsed_seconds"]),
                    }

    paired = []
    for (scene, method, seed), result in sorted(results.items()):
        baseline = results.get((scene, "none", seed))
        if baseline is None:
            continue
        paired.append({
            "scene": scene, "method": method, "seed": seed,
            "test_psnr_db": result["mean_psnr_db"],
            "delta_psnr_db": result["mean_psnr_db"] - baseline["mean_psnr_db"],
            "test_ssim": result["mean_ssim"],
            "delta_ssim": result["mean_ssim"] - baseline["mean_ssim"],
            "training_seconds": result["training_elapsed_seconds"],
            "final_gaussians": result["final_gaussians"],
        })
    fields = ("scene", "method", "seed", "test_psnr_db", "delta_psnr_db",
              "test_ssim", "delta_ssim", "training_seconds", "final_gaussians")
    write_csv(args.output_dir / "paired_runs.csv", fields, paired)

    grouped = defaultdict(list)
    for row in paired:
        grouped[(row["scene"], row["method"])].append(row)
    method_rows = []
    for (scene, method), rows in sorted(grouped.items()):
        method_rows.append({
            "scene": scene, "method": method, "seeds": len(rows),
            "mean_test_psnr_db": statistics.mean(row["test_psnr_db"] for row in rows),
            "mean_delta_psnr_db": statistics.mean(row["delta_psnr_db"] for row in rows),
            "sd_delta_psnr_db": statistics.stdev(row["delta_psnr_db"] for row in rows) if len(rows) > 1 else 0.0,
            "mean_test_ssim": statistics.mean(row["test_ssim"] for row in rows),
            "mean_delta_ssim": statistics.mean(row["delta_ssim"] for row in rows),
            "mean_training_seconds": statistics.mean(row["training_seconds"] for row in rows),
            "mean_final_gaussians": statistics.mean(row["final_gaussians"] for row in rows),
        })
    write_csv(args.output_dir / "method_summary.csv", (
        "scene", "method", "seeds", "mean_test_psnr_db", "mean_delta_psnr_db",
        "sd_delta_psnr_db", "mean_test_ssim", "mean_delta_ssim",
        "mean_training_seconds", "mean_final_gaussians",
    ), method_rows)

    by_curve = defaultdict(list)
    for (scene, method, seed, iteration), curve in curves.items():
        if (scene, "none", seed, iteration) in curves:
            by_curve[(scene, method, iteration)].append(curve)
    curve_rows = []
    for (scene, method, iteration), rows in sorted(by_curve.items()):
        curve_rows.append({
            "scene": scene, "method": method, "iteration": iteration,
            "seeds": len(rows),
            "mean_psnr_db": statistics.mean(row["psnr_db"] for row in rows),
            "mean_ssim": statistics.mean(row["ssim"] for row in rows),
            "mean_elapsed_seconds": statistics.mean(row["elapsed_seconds"] for row in rows),
        })
    write_csv(args.output_dir / "validation_curves.csv", (
        "scene", "method", "iteration", "seeds", "mean_psnr_db",
        "mean_ssim", "mean_elapsed_seconds",
    ), curve_rows)
    write_csv(args.output_dir / "missing_jobs.csv", ("scene", "method", "seed"), [
        {"scene": scene, "method": method, "seed": seed}
        for scene, method, seed in missing
    ])
    print(f"Summarized {len(results)} runs, {len(paired)} paired with a baseline; "
          f"{len(missing)} jobs missing or incomplete. CSVs in {args.output_dir}.")


if __name__ == "__main__":
    main()

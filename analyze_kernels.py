"""Summarize paired Gaussian-versus-linear runs without mixing old filter outputs."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import statistics

from scipy.stats import t

from scripts.run_kernel_matrix import KERNELS, SEEDS


def write_csv(path, fields, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=Path("outputs/kernels"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/kernel_analysis"))
    parser.add_argument("--scenes", nargs="+")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    args = parser.parse_args()
    scenes = args.scenes or (sorted(p.name for p in args.input_root.iterdir() if p.is_dir())
                             if args.input_root.exists() else [])
    results = {}
    configs = {}
    curves = defaultdict(list)
    missing = []
    for scene in scenes:
        for kernel in KERNELS:
            for seed in args.seeds:
                run_dir = args.input_root / scene / kernel / f"seed_{seed}"
                summary_path = run_dir / "test_summary.json"
                config_path = run_dir / "run_config.json"
                curve_path = run_dir / "validation_metrics.csv"
                if not all(p.exists() for p in (summary_path, config_path, curve_path)):
                    missing.append(dict(scene=scene, kernel=kernel, seed=seed,
                                        reason="missing files"))
                    continue
                summary = json.loads(summary_path.read_text())
                config = json.loads(config_path.read_text())
                if (summary.get("completed_iterations") != config["training"]["iterations"]
                        or config["experiment"]["kernel"] != kernel):
                    missing.append(dict(scene=scene, kernel=kernel, seed=seed,
                                        reason="incomplete or wrong kernel"))
                    continue
                results[(scene, kernel, seed)] = summary
                configs[(scene, kernel, seed)] = config
                with curve_path.open(newline="") as stream:
                    for row in csv.DictReader(stream):
                        curves[(scene, kernel, seed, int(row["iteration"]))].append(row)

    paired = []
    for (scene, kernel, seed), result in sorted(results.items()):
        baseline = results.get((scene, "gaussian", seed))
        if baseline is None:
            continue
        radius_fraction = configs[(scene, kernel, seed)].get(
            "projection", {}).get("max_radius_fraction", 0.0)
        baseline_fraction = configs[(scene, "gaussian", seed)].get(
            "projection", {}).get("max_radius_fraction", 0.0)
        if radius_fraction != baseline_fraction:
            missing.append(dict(scene=scene, kernel=kernel, seed=seed,
                                reason="projection setting differs from Gaussian baseline"))
            continue
        paired.append(dict(
            scene=scene, kernel=kernel, seed=seed,
            max_radius_fraction=radius_fraction,
            test_psnr_db=result["mean_psnr_db"],
            delta_psnr_db=result["mean_psnr_db"] - baseline["mean_psnr_db"],
            test_ssim=result["mean_ssim"],
            delta_ssim=result["mean_ssim"] - baseline["mean_ssim"],
            render_ms=1000 * result["mean_render_seconds"],
            tile_pairs=result["mean_tile_pairs"],
            training_seconds=result["training_elapsed_seconds"],
            final_splats=result["final_gaussians"],
        ))
    write_csv(args.output_dir / "paired_runs.csv", (
        "scene", "kernel", "seed", "max_radius_fraction", "test_psnr_db", "delta_psnr_db", "test_ssim",
        "delta_ssim", "render_ms", "tile_pairs", "training_seconds", "final_splats",
    ), paired)

    grouped = defaultdict(list)
    for row in paired:
        grouped[(row["scene"], row["kernel"], row["max_radius_fraction"])].append(row)
    summaries = []
    for (scene, kernel, radius_fraction), rows in sorted(grouped.items()):
        deltas = [row["delta_psnr_db"] for row in rows]
        mean = statistics.mean(deltas)
        if len(deltas) > 1 and statistics.stdev(deltas) > 0:
            sem = statistics.stdev(deltas) / len(deltas) ** 0.5
            low, high = t.interval(0.95, len(deltas) - 1, loc=mean, scale=sem)
        elif len(deltas) > 1:
            low = high = mean
        else:
            low = high = None
        summaries.append(dict(
            scene=scene, kernel=kernel, max_radius_fraction=radius_fraction,
            seeds=len(rows),
            mean_delta_psnr_db=mean, ci95_low_psnr_db=low, ci95_high_psnr_db=high,
            mean_delta_ssim=statistics.mean(r["delta_ssim"] for r in rows),
            mean_render_ms=statistics.mean(r["render_ms"] for r in rows),
            mean_tile_pairs=statistics.mean(r["tile_pairs"] for r in rows),
            mean_training_seconds=statistics.mean(r["training_seconds"] for r in rows),
            mean_final_splats=statistics.mean(r["final_splats"] for r in rows),
        ))
    write_csv(args.output_dir / "kernel_summary.csv", (
        "scene", "kernel", "max_radius_fraction", "seeds", "mean_delta_psnr_db", "ci95_low_psnr_db",
        "ci95_high_psnr_db", "mean_delta_ssim", "mean_render_ms",
        "mean_tile_pairs", "mean_training_seconds", "mean_final_splats",
    ), summaries)

    curve_rows = []
    for (scene, kernel, seed, iteration), rows in sorted(curves.items()):
        curve_rows.append(dict(scene=scene, kernel=kernel, seed=seed, iteration=iteration,
                               mean_psnr_db=statistics.mean(float(r["psnr_db"]) for r in rows),
                               mean_ssim=statistics.mean(float(r["ssim"]) for r in rows)))
    write_csv(args.output_dir / "validation_curves.csv", (
        "scene", "kernel", "seed", "iteration", "mean_psnr_db", "mean_ssim",
    ), curve_rows)
    write_csv(args.output_dir / "missing_jobs.csv", ("scene", "kernel", "seed", "reason"), missing)
    print(f"{len(results)} runs, {len(paired)} paired rows, {len(missing)} missing jobs")


if __name__ == "__main__":
    main()

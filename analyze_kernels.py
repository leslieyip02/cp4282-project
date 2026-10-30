"""Summarize paired kernel runs and available held-out image metrics."""

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


def mean_ci95(values):
    if not values:
        return None, None, None
    mean = statistics.mean(values)
    if len(values) == 1:
        return mean, None, None
    if statistics.stdev(values) == 0:
        return mean, mean, mean
    sem = statistics.stdev(values) / len(values) ** 0.5
    low, high = t.interval(0.95, len(values) - 1, loc=mean, scale=sem)
    return mean, low, high


def mean_optional(values):
    present = [value for value in values if value is not None]
    return statistics.mean(present) if present else None


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
        lpips_net = result.get("lpips_net")
        baseline_lpips_net = baseline.get("lpips_net")
        has_paired_lpips = (result.get("mean_lpips") is not None
                            and baseline.get("mean_lpips") is not None
                            and lpips_net is not None
                            and lpips_net == baseline_lpips_net)
        gpu_name = result.get("gpu_name")
        comparable_memory = (result.get("peak_train_gpu_memory_mib") is not None
                             and baseline.get("peak_train_gpu_memory_mib") is not None
                             and gpu_name is not None
                             and gpu_name == baseline.get("gpu_name"))
        paired.append(dict(
            scene=scene, kernel=kernel, seed=seed,
            max_radius_fraction=radius_fraction,
            test_psnr_db=result["mean_psnr_db"],
            delta_psnr_db=result["mean_psnr_db"] - baseline["mean_psnr_db"],
            test_ssim=result["mean_ssim"],
            delta_ssim=result["mean_ssim"] - baseline["mean_ssim"],
            lpips_net=lpips_net if has_paired_lpips else "",
            test_lpips=result["mean_lpips"] if has_paired_lpips else None,
            delta_lpips=(result["mean_lpips"] - baseline["mean_lpips"]
                         if has_paired_lpips else None),
            render_ms=1000 * result["mean_render_seconds"],
            tile_pairs=result["mean_tile_pairs"],
            training_seconds=result["training_elapsed_seconds"],
            final_splats=result["final_gaussians"],
            peak_train_gpu_memory_mib=result.get("peak_train_gpu_memory_mib"),
            peak_test_gpu_memory_mib=result.get("peak_test_gpu_memory_mib"),
            gpu_name=gpu_name,
            memory_comparable_to_gaussian=comparable_memory,
            delta_peak_train_gpu_memory_mib=(
                result["peak_train_gpu_memory_mib"] - baseline["peak_train_gpu_memory_mib"]
                if comparable_memory else None),
            delta_peak_test_gpu_memory_mib=(
                result["peak_test_gpu_memory_mib"] - baseline["peak_test_gpu_memory_mib"]
                if comparable_memory and result.get("peak_test_gpu_memory_mib") is not None
                and baseline.get("peak_test_gpu_memory_mib") is not None else None),
        ))
    write_csv(args.output_dir / "paired_runs.csv", (
        "scene", "kernel", "seed", "max_radius_fraction", "test_psnr_db", "delta_psnr_db", "test_ssim",
        "delta_ssim", "lpips_net", "test_lpips", "delta_lpips", "render_ms",
        "tile_pairs", "training_seconds", "final_splats",
        "peak_train_gpu_memory_mib", "peak_test_gpu_memory_mib", "gpu_name",
        "memory_comparable_to_gaussian", "delta_peak_train_gpu_memory_mib",
        "delta_peak_test_gpu_memory_mib",
    ), paired)

    grouped = defaultdict(list)
    for row in paired:
        grouped[(row["scene"], row["kernel"], row["max_radius_fraction"])].append(row)
    summaries = []
    for (scene, kernel, radius_fraction), rows in sorted(grouped.items()):
        deltas = [row["delta_psnr_db"] for row in rows]
        mean, low, high = mean_ci95(deltas)
        lpips_rows = [r for r in rows if r["delta_lpips"] is not None]
        lpips_nets = {r["lpips_net"] for r in lpips_rows}
        if len(lpips_nets) > 1:
            raise ValueError(f"Mixed LPIPS backbones for {scene}/{kernel}: {lpips_nets}")
        lpips_mean, lpips_low, lpips_high = mean_ci95(
            [r["delta_lpips"] for r in lpips_rows])
        summaries.append(dict(
            scene=scene, kernel=kernel, max_radius_fraction=radius_fraction,
            seeds=len(rows),
            mean_delta_psnr_db=mean, ci95_low_psnr_db=low, ci95_high_psnr_db=high,
            mean_delta_ssim=statistics.mean(r["delta_ssim"] for r in rows),
            lpips_net=next(iter(lpips_nets)) if lpips_nets else "",
            lpips_seeds=len(lpips_rows),
            mean_test_lpips=(statistics.mean(r["test_lpips"] for r in lpips_rows)
                             if lpips_rows else None),
            mean_delta_lpips=lpips_mean,
            ci95_low_lpips=lpips_low, ci95_high_lpips=lpips_high,
            mean_render_ms=statistics.mean(r["render_ms"] for r in rows),
            mean_tile_pairs=statistics.mean(r["tile_pairs"] for r in rows),
            mean_training_seconds=statistics.mean(r["training_seconds"] for r in rows),
            mean_final_splats=statistics.mean(r["final_splats"] for r in rows),
            memory_seeds=sum(r["memory_comparable_to_gaussian"] for r in rows),
            mean_peak_train_gpu_memory_mib=mean_optional(
                r["peak_train_gpu_memory_mib"] for r in rows
                if r["memory_comparable_to_gaussian"]),
            mean_peak_test_gpu_memory_mib=mean_optional(
                r["peak_test_gpu_memory_mib"] for r in rows
                if r["memory_comparable_to_gaussian"]),
            mean_delta_peak_train_gpu_memory_mib=mean_optional(
                r["delta_peak_train_gpu_memory_mib"] for r in rows),
            mean_delta_peak_test_gpu_memory_mib=mean_optional(
                r["delta_peak_test_gpu_memory_mib"] for r in rows),
        ))
    write_csv(args.output_dir / "kernel_summary.csv", (
        "scene", "kernel", "max_radius_fraction", "seeds", "mean_delta_psnr_db", "ci95_low_psnr_db",
        "ci95_high_psnr_db", "mean_delta_ssim", "lpips_net", "lpips_seeds",
        "mean_test_lpips", "mean_delta_lpips", "ci95_low_lpips", "ci95_high_lpips",
        "mean_render_ms",
        "mean_tile_pairs", "mean_training_seconds", "mean_final_splats",
        "memory_seeds", "mean_peak_train_gpu_memory_mib", "mean_peak_test_gpu_memory_mib",
        "mean_delta_peak_train_gpu_memory_mib", "mean_delta_peak_test_gpu_memory_mib",
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

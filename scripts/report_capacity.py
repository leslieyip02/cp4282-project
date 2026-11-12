"""Compare matched kernel runs at two splat capacities.

Example:
    python scripts/report_capacity.py --low-root <low-root>/kernels \
        --high-root <high-root>/kernels --scene bonsai
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import csv
import hashlib
import os
from pathlib import Path
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "cp4282-mpl"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "cp4282-cache"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from report_kernels import KERNELS, METRICS, LABELS, COLORS, ci95, load_run, metric_pair


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def fmt(value: float | None, digits: int = 3) -> str:
    return "—" if value is None else f"{value:+.{digits}f}"


def markdown_table(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |",
                      "| " + " | ".join("---" for _ in headers) + " |",
                      *("| " + " | ".join(str(x) for x in row) + " |" for row in rows)])


def first_at_capacity(run):
    with (run["directory"] / "validation_metrics.csv").open(newline="") as stream:
        steps = [int(row["iteration"]) for row in csv.DictReader(stream)
                 if int(row["gaussians"]) >= run["config"]["model"]["capacity"]]
    return min(steps) if steps else None


def compare_config(low, high):
    a, b = deepcopy(low["config"]), deepcopy(high["config"])
    low_cap, high_cap = a["model"].pop("capacity"), b["model"].pop("capacity")
    if low_cap >= high_cap:
        raise ValueError(f"Expected a higher capacity, got {low_cap} and {high_cap}")
    # Saved output and initialization paths change with the output root.
    a.pop("paths", None)
    b.pop("paths", None)
    changed = [key for key in a if a[key] != b.get(key)] + [key for key in b if key not in a]
    if changed:
        raise ValueError(f"Settings differ beyond capacity: {changed}")
    if low["split"] is None or low["split"] != high["split"]:
        raise ValueError("Train/validation/test splits differ or are missing")
    if set(low["views"]) != set(high["views"]):
        raise ValueError("Test view filenames differ")
    for key in ("iterations",):
        if low["config"]["training"][key] != high["config"]["training"][key]:
            raise ValueError(f"Training {key} differs")
    if (low["config"]["learning_rates"]["position_max_steps"] !=
            high["config"]["learning_rates"]["position_max_steps"]):
        raise ValueError("Position learning-rate schedule differs")
    if low["summary"].get("lpips_net") != high["summary"].get("lpips_net"):
        raise ValueError("LPIPS backbones differ")
    init_name = "colmap_init.ply"
    if not (low["directory"] / init_name).is_file() or not (high["directory"] / init_name).is_file():
        raise ValueError("Cannot verify the shared COLMAP initialization")
    if digest(low["directory"] / init_name) != digest(high["directory"] / init_name):
        raise ValueError("COLMAP initialization PLYs differ")
    return low_cap, high_cap


def plots(output, runs, seeds, low_cap, high_cap):
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), constrained_layout=True)
    for metric, ax in zip(METRICS, axes):
        for kernel in KERNELS:
            for seed in seeds:
                low, high = runs[("low", kernel, seed)], runs[("high", kernel, seed)]
                a = low["summary"].get({"psnr_db":"mean_psnr_db", "ssim":"mean_ssim", "lpips":"mean_lpips"}[metric])
                b = high["summary"].get({"psnr_db":"mean_psnr_db", "ssim":"mean_ssim", "lpips":"mean_lpips"}[metric])
                if a is not None and b is not None:
                    ax.plot([0, 1], [a, b], marker="o", color=COLORS[kernel],
                            alpha=0.65, label=kernel if seed == seeds[0] else None)
        ax.set(xticks=[0, 1], xticklabels=[f"{low_cap//1000}k", f"{high_cap//1000}k"],
               xlabel="Splat capacity", ylabel=f"Final test {LABELS[metric]}")
        ax.grid(alpha=0.2)
    axes[0].legend(fontsize=8)
    fig.suptitle("Capacity response at the same training budget")
    fig.savefig(output / "capacity_response.png", dpi=170)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4.5), constrained_layout=True)
    for kernel in KERNELS:
        for level, linestyle in (("low", "--"), ("high", "-")):
            curves = [runs[(level, kernel, seed)]["validation"] for seed in seeds]
            common = sorted(set.intersection(*(set(curve) for curve in curves)))
            ax.plot(common, [np.mean([curve[step]["psnr_db"] for curve in curves])
                             for step in common], color=COLORS[kernel], linestyle=linestyle,
                    label=f"{kernel} {low_cap//1000 if level == 'low' else high_cap//1000}k")
    ax.set(xlabel="Training step", ylabel="Mean validation PSNR (dB)")
    ax.grid(alpha=0.2)
    ax.legend(ncol=2, fontsize=8)
    fig.savefig(output / "validation_capacity.png", dpi=170)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--low-root", type=Path, required=True,
                        help="Directory containing <scene>/<kernel>/seed_<n>/ at lower capacity")
    parser.add_argument("--high-root", type=Path, required=True,
                        help="Directory containing matching higher-capacity runs")
    parser.add_argument("--scene", required=True)
    parser.add_argument("--seeds", nargs="+", type=int,
                        help="Defaults to seeds in the higher-capacity Gaussian directory")
    parser.add_argument("--output-dir", type=Path,
                        help="Defaults to capacity_report/<scene>/ beside high-root")
    args = parser.parse_args()
    seed_dir = args.high_root / args.scene / "gaussian"
    seeds = args.seeds or sorted(int(p.name[5:]) for p in seed_dir.glob("seed_*")
                                 if p.is_dir() and p.name[5:].isdigit())
    if not seeds:
        parser.error("No higher-capacity Gaussian seeds found")
    output = args.output_dir or args.high_root.parent / "capacity_report" / args.scene
    output.mkdir(parents=True, exist_ok=True)
    runs = {(level, kernel, seed): load_run(root, args.scene, kernel, seed)
            for level, root in (("low", args.low_root), ("high", args.high_root))
            for kernel in KERNELS for seed in seeds}
    missing = [f"{level}/{kernel}/seed_{seed}" for (level, kernel, seed), run in runs.items()
               if run is None]
    if missing:
        raise ValueError(f"Missing complete runs: {', '.join(missing)}")
    reference_split = runs[("low", "gaussian", seeds[0])]["split"]
    reference_views = set(runs[("low", "gaussian", seeds[0])]["views"])
    if any(run["split"] != reference_split or set(run["views"]) != reference_views
           for run in runs.values()):
        raise ValueError("Kernel/seed runs do not share the same split and test views")
    capacities = {compare_config(runs[("low", kernel, seed)], runs[("high", kernel, seed)])
                  for kernel in KERNELS for seed in seeds}
    if len(capacities) != 1:
        raise ValueError(f"Capacity pair varies across runs: {capacities}")
    low_cap, high_cap = capacities.pop()
    rows, view_rows = [], []
    for seed in seeds:
        for kernel in KERNELS:
            low, high = runs[("low", kernel, seed)], runs[("high", kernel, seed)]
            delta = {metric: metric_pair(high, low, metric) for metric in METRICS}
            base_delta = {metric: metric_pair(runs[("high", "gaussian", seed)],
                                              runs[("low", "gaussian", seed)], metric)
                          for metric in METRICS}
            same_gpu = (low["summary"].get("gpu_name") is not None and
                        low["summary"].get("gpu_name") == high["summary"].get("gpu_name"))
            row = {"seed": seed, "kernel": kernel, "low_capacity": low_cap,
                   "high_capacity": high_cap, "low_splats": low["summary"]["final_gaussians"],
                   "high_splats": high["summary"]["final_gaussians"],
                   "low_first_at_cap": first_at_capacity(low),
                   "high_first_at_cap": first_at_capacity(high),
                   "gpu_matched": same_gpu,
                   "low_gpu": low["summary"].get("gpu_name"),
                   "high_gpu": high["summary"].get("gpu_name")}
            for metric in METRICS:
                row[f"low_{metric}"] = low["summary"].get({"psnr_db":"mean_psnr_db", "ssim":"mean_ssim", "lpips":"mean_lpips"}[metric])
                row[f"high_{metric}"] = high["summary"].get({"psnr_db":"mean_psnr_db", "ssim":"mean_ssim", "lpips":"mean_lpips"}[metric])
                row[f"capacity_delta_{metric}"] = delta[metric]
                row[f"interaction_{metric}"] = (delta[metric] - base_delta[metric]
                                                  if delta[metric] is not None and
                                                  base_delta[metric] is not None else None)
            for field in ("training_elapsed_seconds", "peak_train_gpu_memory_mib",
                          "mean_render_seconds"):
                row[f"low_{field}"] = low["summary"].get(field)
                row[f"high_{field}"] = high["summary"].get(field)
            rows.append(row)
            for frame in sorted(low["views"]):
                a, b = low["views"][frame], high["views"][frame]
                view_rows.append({"seed": seed, "kernel": kernel, "frame": frame,
                                  **{f"capacity_delta_{metric}": (
                                      float(b[metric]) - float(a[metric])
                                      if a.get(metric) not in (None, "") and b.get(metric) not in (None, "")
                                      else None) for metric in METRICS}})
    with (output / "capacity_comparison.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    with (output / "per_view_capacity.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(view_rows[0]))
        writer.writeheader(); writer.writerows(view_rows)
    plots(output, runs, seeds, low_cap, high_cap)

    summary_rows = [[row["seed"], row["kernel"],
                     f"{row['low_psnr_db']:.2f} → {row['high_psnr_db']:.2f}",
                     fmt(row["capacity_delta_psnr_db"]),
                     fmt(row["capacity_delta_ssim"], 4),
                     fmt(row["capacity_delta_lpips"], 4),
                     fmt(row["interaction_psnr_db"] if row["kernel"] != "gaussian" else None),
                     f"{row['low_first_at_cap']} → {row['high_first_at_cap']}"]
                    for row in rows]
    aggregate_rows = []
    for kernel in KERNELS:
        group = [row for row in rows if row["kernel"] == kernel]
        cells = []
        for metric in METRICS:
            values = [row[f"capacity_delta_{metric}"] for row in group
                      if row[f"capacity_delta_{metric}"] is not None]
            center, low, high = ci95(values)
            digits = 3 if metric == "psnr_db" else 4
            cells.append(f"{fmt(center, digits)} [{fmt(low, digits)}, {fmt(high, digits)}]")
        aggregate_rows.append([kernel, *cells,
                               f"{sum(row['capacity_delta_psnr_db'] > 0 for row in group)}/{len(group)}"])
    gap_rows = []
    for kernel in ("linear", "polynomial"):
        group = [row for row in rows if row["kernel"] == kernel]
        low_gap = [row["low_psnr_db"] - next(base["low_psnr_db"] for base in rows
                    if base["kernel"] == "gaussian" and base["seed"] == row["seed"])
                   for row in group]
        high_gap = [row["high_psnr_db"] - next(base["high_psnr_db"] for base in rows
                     if base["kernel"] == "gaussian" and base["seed"] == row["seed"])
                    for row in group]
        interaction = [row["interaction_psnr_db"] for row in group]
        gap_rows.append([kernel, fmt(np.mean(low_gap)), fmt(np.mean(high_gap)),
                         fmt(np.mean(interaction)),
                         f"{sum(value > 0 for value in high_gap)}/{len(high_gap)}"])
    view_counts = []
    for seed in seeds:
        for kernel in KERNELS:
            group = [row for row in view_rows if row["seed"] == seed and row["kernel"] == kernel]
            view_counts.append([seed, kernel,
                f"{sum(row['capacity_delta_psnr_db'] > 0 for row in group)}/{len(group)}",
                f"{sum(row['capacity_delta_ssim'] > 0 for row in group)}/{len(group)}",
                f"{sum(row['capacity_delta_lpips'] < 0 for row in group if row['capacity_delta_lpips'] is not None)}/"
                f"{sum(row['capacity_delta_lpips'] is not None for row in group)}"])
    convergence = []
    for seed in seeds:
        for kernel in KERNELS:
            curve = runs[("high", kernel, seed)]["validation"]
            last = max(curve)
            best = max(curve, key=lambda step: curve[step]["psnr_db"])
            convergence.append([seed, kernel, f"{curve[last]['psnr_db']:.2f}",
                                f"{curve[best]['psnr_db']:.2f} @ {best:,}",
                                fmt(curve[last]["psnr_db"] - curve[best]["psnr_db"], 2)])
    cost_rows = []
    for row in rows:
        same = row["gpu_matched"]
        cost_rows.append([row["seed"], row["kernel"],
                          row["low_gpu"] if same else f"{row['low_gpu']} → {row['high_gpu']}",
                          (f"{row['high_training_elapsed_seconds']/row['low_training_elapsed_seconds']:.2f}×"
                           if same else "—"),
                          (fmt(row["high_peak_train_gpu_memory_mib"] - row["low_peak_train_gpu_memory_mib"], 0)
                           if same and row["low_peak_train_gpu_memory_mib"] is not None
                           and row["high_peak_train_gpu_memory_mib"] is not None else "—")])
    lines = [f"# {args.scene} capacity comparison", "",
             f"{low_cap:,} versus {high_cap:,} splats, both trained for "
             f"{runs[('low','gaussian',seeds[0])]['config']['training']['iterations']:,} steps. "
             f"Matched configurations, split, test views, LPIPS backbone, and COLMAP initializations verified.", "",
             "Positive PSNR and SSIM, negative LPIPS favor higher capacity. The interaction "
             "is the change in an alternative kernel's advantage over Gaussian; a negative "
             "PSNR interaction means Gaussian gained more from the extra capacity.", "",
             markdown_table(["Seed", "Kernel", "PSNR low → high", "ΔPSNR", "ΔSSIM", "ΔLPIPS",
                             "PSNR interaction", "First at cap"], summary_rows), "",
             "## Across-seed summary", "",
             "Means and 95% t intervals across paired seeds. Three seeds give very wide "
             "intervals; these describe stochastic variation on this one scene.", "",
             markdown_table(["Kernel", "Mean ΔPSNR [95% CI]", "Mean ΔSSIM [95% CI]",
                             "Mean ΔLPIPS [95% CI]", "PSNR gains"], aggregate_rows), "",
             "The PSNR interaction is (alternative − Gaussian) at high capacity minus the "
             "same gap at low capacity. Negative values mean Gaussian gained more from the "
             "additional splats.", "",
             markdown_table(["Kernel", f"{low_cap//1000}k gap vs Gaussian",
                             f"{high_cap//1000}k gap vs Gaussian",
                             "Mean interaction", f"{high_cap//1000}k PSNR wins"], gap_rows), "",
             "![Capacity response](capacity_response.png)", "",
             "![Validation curves](validation_capacity.png)", "",
             "## Held-out view consistency", "",
             "A view is counted as improved when the higher-capacity run beats its own "
             "lower-capacity run on that camera.", "",
             markdown_table(["Seed", "Kernel", "PSNR views better", "SSIM views better",
                             "LPIPS views better"], view_counts), "",
             "## Higher-capacity convergence", "",
             markdown_table(["Seed", "Kernel", "Final validation PSNR", "Best validation PSNR",
                             "Final − best"], convergence), "",
             "## Cost on matched GPU models", "",
             "Training-time ratios and memory differences are omitted when GPU models differ. "
             "Memory is a 250 ms sampled per-process peak.", "",
             markdown_table(["Seed", "Kernel", "GPU", "Train time high/low", "Δtrain peak MiB"],
                            cost_rows), "",
             f"This is a {len(seeds)}-seed capacity comparison on one scene. "
             f"The {len(reference_views)} held-out views are repeated measurements of the same scene, not independent "
             "scene replicates. Inspect matched image crops before attributing artifacts to capacity.", "",
             "Data: [run and interaction values](capacity_comparison.csv), "
             "[per-view changes](per_view_capacity.csv).", ""]
    (output / "report.md").write_text("\n".join(lines))
    print(f"Wrote {output / 'report.md'}")


if __name__ == "__main__":
    main()

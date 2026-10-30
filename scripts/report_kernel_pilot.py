"""Report a longer-training kernel pilot against Gaussian and an earlier budget.

Example:
    python scripts/report_kernel_pilot.py --scene bonsai \
        --pilot-root outputs-pilot/kernels --reference-root outputs/kernels
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
import os
from pathlib import Path
import statistics
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "cp4282-mpl"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "cp4282-cache"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import t


KERNELS = ("gaussian", "linear", "polynomial")
METRICS = ("psnr_db", "ssim", "lpips")
SUMMARY_KEYS = {"psnr_db": "mean_psnr_db", "ssim": "mean_ssim", "lpips": "mean_lpips"}
COLORS = {"gaussian": "#4b5563", "linear": "#2563eb", "polynomial": "#db651b"}
LABELS = {"psnr_db": "PSNR (dB)", "ssim": "SSIM", "lpips": "LPIPS (VGG)"}


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def number(value):
    return None if value in (None, "") else float(value)


def mean(values):
    values = list(values)
    return statistics.mean(values) if values else None


def ci95(values):
    values = list(values)
    center = mean(values)
    if len(values) < 2:
        return center, None, None
    spread = statistics.stdev(values)
    if spread == 0:
        return center, center, center
    low, high = t.interval(0.95, len(values) - 1,
                           loc=center, scale=spread / len(values) ** 0.5)
    return center, low, high


def fmt(value, digits=3):
    return "—" if value is None else f"{value:+.{digits}f}"


def table(headers, rows):
    return "\n".join([
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
        *("| " + " | ".join(str(cell) for cell in row) + " |" for row in rows),
    ])


def load_run(root: Path, scene: str, kernel: str, seed: int):
    directory = root / scene / kernel / f"seed_{seed}"
    required = ("run_config.json", "test_summary.json", "test_metrics.csv",
                "validation_metrics.csv")
    if not all((directory / name).exists() for name in required):
        return None
    config = json.loads((directory / "run_config.json").read_text())
    summary = json.loads((directory / "test_summary.json").read_text())
    if (summary.get("completed_iterations") != config["training"]["iterations"]
            or config["experiment"]["kernel"] != kernel
            or config["runtime"]["seed"] != seed):
        raise ValueError(f"Incomplete or mislabeled run: {directory}")
    view_rows = read_csv(directory / "test_metrics.csv")
    views = {row["frame"]: row for row in view_rows}
    if not views or len(views) != len(view_rows) or summary.get("views") != len(view_rows):
        raise ValueError(f"Missing or duplicated test views: {directory}")
    curves = defaultdict(list)
    for row in read_csv(directory / "validation_metrics.csv"):
        curves[int(row["iteration"])].append(row)
    validation = {
        step: {metric: mean(number(row[metric]) for row in rows)
               for metric in ("psnr_db", "ssim")}
        for step, rows in curves.items()
    }
    if not validation or max(validation) != config["training"]["iterations"]:
        raise ValueError(f"Validation curve is incomplete: {directory}")
    split_path = directory / "split.json"
    return dict(directory=directory, config=config, summary=summary,
                views=views, validation=validation,
                split=json.loads(split_path.read_text()) if split_path.exists() else None)


def settings(run):
    config = run["config"]
    model = config["model"]
    return {
        "width": model["width"], "height": model["height"],
        "capacity": model["capacity"], "initial_splats": model["initial_splats"],
        "scene_radius": model["scene_radius"], "background": model["background"],
        "learning_rates_except_schedule": {
            key: value for key, value in config["learning_rates"].items()
            if key != "position_max_steps"
        },
        "projection": config.get("projection"),
        "compact_box": config.get("compact_box"),
        "densification": config.get("densification"),
        "multiview_adc": config.get("multiview_adc"),
        "sparse": config.get("sparse"),
        "adaptive": config.get("adaptive"),
        "convergence": config.get("convergence"),
        "validation_views": config["reporting"]["eval_views"],
    }


def metric_pair(a, b, metric):
    if metric == "lpips":
        if (a["summary"].get("lpips_net") != b["summary"].get("lpips_net")
                or a["summary"].get("lpips_net") is None):
            return None
    av = a["summary"].get(SUMMARY_KEYS[metric])
    bv = b["summary"].get(SUMMARY_KEYS[metric])
    return float(av) - float(bv) if av is not None and bv is not None else None


def plot_validation(path: Path, pilot, reference, seeds):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for metric, ax in zip(("psnr_db", "ssim"), axes):
        for kernel in KERNELS:
            for source, linestyle, alpha in ((reference, "--", 0.7), (pilot, "-", 1.0)):
                runs = [source.get((kernel, seed)) for seed in seeds]
                runs = [run for run in runs if run]
                if not runs:
                    continue
                common_steps = sorted(set.intersection(
                    *(set(run["validation"]) for run in runs)))
                if not common_steps:
                    continue
                y = [mean(run["validation"][step][metric] for run in runs)
                     for step in common_steps]
                label = f"{kernel} ({'pilot' if source is pilot else 'earlier'})"
                ax.plot(common_steps, y, color=COLORS[kernel], linestyle=linestyle,
                        alpha=alpha, linewidth=2 if source is pilot else 1.5, label=label)
                if source is pilot:
                    for run in runs:
                        steps = sorted(run["validation"])
                        ax.plot(steps, [run["validation"][step][metric] for step in steps],
                                color=COLORS[kernel], alpha=0.16, linewidth=0.8)
        ax.set(xlabel="Training step", ylabel=f"Mean validation {LABELS[metric]}")
        ax.grid(alpha=0.2)
    axes[0].legend(fontsize=8, ncol=2)
    fig.suptitle("Validation trajectories: thick lines average seeds; faint lines are pilot seeds")
    fig.savefig(path, dpi=170)
    plt.close(fig)


def plot_paired(path: Path, paired_rows):
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), constrained_layout=True)
    for metric, ax in zip(METRICS, axes):
        key = f"delta_{metric}"
        for kernel, offset in (("linear", -0.08), ("polynomial", 0.08)):
            rows = [row for row in paired_rows if row["kernel"] == kernel
                    and row[key] is not None]
            ax.scatter([row["seed"] + offset for row in rows],
                       [row[key] for row in rows], color=COLORS[kernel], s=45,
                       label=kernel)
            if rows:
                ax.plot([row["seed"] + offset for row in rows],
                        [row[key] for row in rows], color=COLORS[kernel], alpha=0.35)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set(xlabel="Seed", ylabel=f"Test {LABELS[metric]} − Gaussian")
        ax.set_xticks(sorted({row["seed"] for row in paired_rows}))
        ax.grid(axis="y", alpha=0.2)
    axes[0].legend(fontsize=8)
    fig.suptitle("Paired final-test differences; lower LPIPS is better")
    fig.savefig(path, dpi=170)
    plt.close(fig)


def plot_views(path: Path, per_view_rows):
    fig, ax = plt.subplots(figsize=(11, 4.5), constrained_layout=True)
    frames = []
    for kernel in ("linear", "polynomial"):
        grouped = defaultdict(list)
        for row in per_view_rows:
            if row["kernel"] == kernel:
                grouped[row["frame"]].append(row["delta_psnr_db"])
        if not grouped:
            continue
        frames = sorted(grouped)
        values = [mean(grouped[frame]) for frame in frames]
        ax.plot(range(len(frames)), values, color=COLORS[kernel], marker=".",
                markersize=4, label=kernel)
        ax.fill_between(range(len(frames)),
                        [min(grouped[frame]) for frame in frames],
                        [max(grouped[frame]) for frame in frames],
                        color=COLORS[kernel], alpha=0.1)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set(xlabel="Held-out view (filename order)",
           ylabel="Mean paired PSNR difference (dB)")
    ax.set_xticks(range(0, len(frames), max(1, len(frames) // 10)))
    ax.grid(axis="y", alpha=0.2)
    ax.legend()
    fig.suptitle("Per-view differences; shading spans the three seed values")
    fig.savefig(path, dpi=170)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", default="bonsai")
    parser.add_argument("--pilot-root", type=Path, default=Path("outputs-pilot/kernels"))
    parser.add_argument("--reference-root", type=Path, default=Path("outputs/kernels"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("outputs-pilot/pilot_report"))
    parser.add_argument("--seeds", nargs="+", type=int, default=[7, 17, 27])
    args = parser.parse_args()
    output = args.output_dir / args.scene
    output.mkdir(parents=True, exist_ok=True)
    pilot = {(kernel, seed): load_run(args.pilot_root, args.scene, kernel, seed)
             for kernel in KERNELS for seed in args.seeds}
    reference = {(kernel, seed): load_run(args.reference_root, args.scene, kernel, seed)
                 for kernel in KERNELS for seed in args.seeds}
    missing = [f"{kernel}/seed_{seed}" for (kernel, seed), run in pilot.items() if run is None]
    if missing:
        raise ValueError(f"Pilot runs missing: {', '.join(missing)}")
    exemplar = pilot[("gaussian", args.seeds[0])]
    pilot_settings = settings(exemplar)
    pilot_split = exemplar["split"]
    pilot_steps = exemplar["config"]["training"]["iterations"]
    pilot_schedule = exemplar["config"]["learning_rates"]["position_max_steps"]
    pilot_lpips_net = exemplar["summary"].get("lpips_net")
    warnings = []
    for (kernel, seed), run in pilot.items():
        if settings(run) != pilot_settings:
            warnings.append(f"Pilot settings differ: {kernel}/seed_{seed}")
        if run["split"] != pilot_split:
            warnings.append(f"Pilot split differs: {kernel}/seed_{seed}")
        if (run["config"]["training"]["iterations"] != pilot_steps
                or run["config"]["learning_rates"]["position_max_steps"] != pilot_schedule):
            warnings.append(f"Pilot training budget or schedule differs: {kernel}/seed_{seed}")
        if run["summary"].get("lpips_net") != pilot_lpips_net:
            warnings.append(f"Pilot LPIPS backbone differs: {kernel}/seed_{seed}")
    for (kernel, seed), run in reference.items():
        if run is None:
            continue
        if settings(run) != pilot_settings:
            warnings.append(f"Earlier and pilot settings differ: {kernel}/seed_{seed}")
        if run["split"] != pilot_split:
            warnings.append(f"Earlier and pilot split differ: {kernel}/seed_{seed}")

    paired, per_view, budget = [], [], []
    for seed in args.seeds:
        gaussian = pilot[("gaussian", seed)]
        for kernel in KERNELS:
            run = pilot[(kernel, seed)]
            old = reference[(kernel, seed)]
            if old:
                budget.append(dict(kernel=kernel, seed=seed,
                                   old_steps=old["config"]["training"]["iterations"],
                                   pilot_steps=run["config"]["training"]["iterations"],
                                   **{f"delta_{metric}": metric_pair(run, old, metric)
                                      for metric in METRICS}))
            if kernel == "gaussian":
                continue
            if set(run["views"]) != set(gaussian["views"]):
                raise ValueError(f"Test views differ: {kernel}/seed_{seed}")
            paired.append(dict(kernel=kernel, seed=seed,
                               **{f"delta_{metric}": metric_pair(run, gaussian, metric)
                                  for metric in METRICS},
                               render_ms=1000 * run["summary"]["mean_render_seconds"],
                               gaussian_render_ms=1000 * gaussian["summary"]["mean_render_seconds"],
                               tile_pairs=run["summary"]["mean_tile_pairs"],
                               gaussian_tile_pairs=gaussian["summary"]["mean_tile_pairs"],
                               peak_train_gpu_memory_mib=run["summary"].get("peak_train_gpu_memory_mib"),
                               gaussian_peak_train_gpu_memory_mib=gaussian["summary"].get(
                                   "peak_train_gpu_memory_mib"),
                               peak_test_gpu_memory_mib=run["summary"].get("peak_test_gpu_memory_mib"),
                               gpu_name=run["summary"].get("gpu_name"),
                               gaussian_gpu_name=gaussian["summary"].get("gpu_name")))
            for frame in sorted(run["views"]):
                row, base = run["views"][frame], gaussian["views"][frame]
                matching_lpips = (run["summary"].get("lpips_net")
                                  == gaussian["summary"].get("lpips_net"))
                per_view.append(dict(kernel=kernel, seed=seed, frame=frame,
                                     **{f"delta_{metric}": (
                                         number(row.get(metric)) - number(base.get(metric))
                                         if number(row.get(metric)) is not None
                                         and number(base.get(metric)) is not None
                                         and (metric != "lpips" or matching_lpips) else None)
                                        for metric in METRICS}))

    write_csv(output / "paired_test_deltas.csv",
              ["kernel", "seed", "delta_psnr_db", "delta_ssim", "delta_lpips",
               "render_ms", "gaussian_render_ms", "tile_pairs", "gaussian_tile_pairs",
               "peak_train_gpu_memory_mib", "gaussian_peak_train_gpu_memory_mib",
               "peak_test_gpu_memory_mib", "gpu_name", "gaussian_gpu_name"], paired)
    write_csv(output / "per_view_deltas.csv",
              ["kernel", "seed", "frame", "delta_psnr_db", "delta_ssim", "delta_lpips"], per_view)
    write_csv(output / "budget_changes.csv",
              ["kernel", "seed", "old_steps", "pilot_steps", "delta_psnr_db",
               "delta_ssim", "delta_lpips"], budget)
    plot_validation(output / "validation_curves.png", pilot, reference, args.seeds)
    plot_paired(output / "paired_test_deltas.png", paired)
    plot_views(output / "per_view_psnr.png", per_view)

    lines = [f"# {args.scene} kernel pilot", "",
             f"Pilot: {args.pilot_root / args.scene}; earlier runs: {args.reference_root / args.scene}.",
             "", "## Coverage and comparability", "",
             f"All {len(pilot)} pilot kernel/seed runs completed. "
             f"Pilot budget: {exemplar['config']['training']['iterations']:,} steps; "
             f"position learning-rate schedule: {exemplar['config']['learning_rates']['position_max_steps']:,} steps. "
             f"Resolution: {pilot_settings['width']}×{pilot_settings['height']}; "
             f"capacity: {pilot_settings['capacity']:,} splats.", ""]
    if warnings:
        lines += ["**Comparability warnings:**", ""] + [f"- {warning}" for warning in warnings] + [""]
    else:
        lines += ["Kernel/seed settings and splits match within the pilot and against available earlier runs.", ""]

    lines += ["## At a glance", ""]
    for kernel in ("linear", "polynomial"):
        rows = [row for row in paired if row["kernel"] == kernel]
        score = [sum(row[f"delta_{metric}"] * direction > 0 for row in rows
                     if row[f"delta_{metric}"] is not None)
                 for metric, direction in (("psnr_db", 1), ("ssim", 1), ("lpips", -1))]
        lines.append(f"- **{kernel}:** improves PSNR in {score[0]}/{len(rows)} seeds, "
                     f"SSIM in {score[1]}/{len(rows)}, and LPIPS in {score[2]}/{len(rows)}.")
    lines += ["", "Absolute final-test values:", "",
              table(["Kernel", "Seed", "PSNR dB", "SSIM", "LPIPS"], [
                  [kernel, seed,
                   f"{pilot[(kernel, seed)]['summary']['mean_psnr_db']:.2f}",
                   f"{pilot[(kernel, seed)]['summary']['mean_ssim']:.4f}",
                   (f"{pilot[(kernel, seed)]['summary']['mean_lpips']:.4f}"
                    if pilot[(kernel, seed)]["summary"].get("mean_lpips") is not None else "—")]
                  for seed in args.seeds for kernel in KERNELS]), ""]

    lines += ["## Paired final-test differences from Gaussian", "",
              "Positive PSNR/SSIM and negative LPIPS favor the alternative kernel. "
              "These are seed pairs; test views are not independent replicates.", "",
              table(["Kernel", "Seed", "ΔPSNR dB", "ΔSSIM", "ΔLPIPS"], [
                  [row["kernel"], row["seed"], fmt(row["delta_psnr_db"]),
                   fmt(row["delta_ssim"], 4), fmt(row["delta_lpips"], 4)]
                  for row in paired]), "",
              "Across-seed means and 95% t intervals (three seeds give wide intervals):", ""]
    aggregate = []
    for kernel in ("linear", "polynomial"):
        rows = [row for row in paired if row["kernel"] == kernel]
        stats = [ci95(row[f"delta_{metric}"] for row in rows
                      if row[f"delta_{metric}"] is not None) for metric in METRICS]
        aggregate.append([kernel] + [
            f"{fmt(center, 4 if metric != 'psnr_db' else 3)} "
            f"[{fmt(low, 4 if metric != 'psnr_db' else 3)}, "
            f"{fmt(high, 4 if metric != 'psnr_db' else 3)}]"
            for metric, (center, low, high) in zip(METRICS, stats)])
    lines += [table(["Kernel", "ΔPSNR mean [95% CI]", "ΔSSIM mean [95% CI]",
                     "ΔLPIPS mean [95% CI]"], aggregate), "",
              "![Paired final-test differences](paired_test_deltas.png)", ""]

    lines += ["## Test-view consistency", ""]
    consistency = []
    for kernel in ("linear", "polynomial"):
        for seed in args.seeds:
            rows = [row for row in per_view if row["kernel"] == kernel and row["seed"] == seed]
            consistency.append([kernel, seed,
                                f"{sum(row['delta_psnr_db'] > 0 for row in rows)}/{len(rows)}",
                                f"{sum(row['delta_ssim'] > 0 for row in rows)}/{len(rows)}",
                                (f"{sum(row['delta_lpips'] < 0 for row in rows if row['delta_lpips'] is not None)}/"
                                 f"{sum(row['delta_lpips'] is not None for row in rows)}")])
    lines += [table(["Kernel", "Seed", "PSNR views better", "SSIM views better",
                     "LPIPS views better"], consistency), "",
              "![Per-view PSNR differences](per_view_psnr.png)", ""]
    worst_views = []
    for kernel in ("linear", "polynomial"):
        grouped = defaultdict(list)
        for row in per_view:
            if row["kernel"] == kernel:
                grouped[row["frame"]].append(row["delta_psnr_db"])
        worst_views.extend((kernel, frame, mean(values), min(values), max(values))
                           for frame, values in grouped.items())
    lines += ["Views with the smallest mean PSNR change across seeds are good candidates "
              "for matched image crops:", "",
              table(["Kernel", "View", "Mean ΔPSNR", "Seed range"], [
                  [kernel, frame, fmt(center, 2), f"{fmt(low, 2)} to {fmt(high, 2)}"]
                  for kernel in ("linear", "polynomial")
                  for _, frame, center, low, high in sorted(
                      (row for row in worst_views if row[0] == kernel),
                      key=lambda row: row[2])[:3]]), ""]

    lines += ["## Validation convergence", "",
              "The validation plot averages the same reserved views at each checkpoint. "
              "Faint lines show individual pilot seeds; dashed lines show earlier runs.", "",
              "![Validation curves](validation_curves.png)", ""]
    convergence = []
    for kernel in KERNELS:
        for seed in args.seeds:
            curve = pilot[(kernel, seed)]["validation"]
            last = max(curve)
            peak = max(curve, key=lambda step: curve[step]["psnr_db"])
            earlier = [step for step in curve if step <= last - 2000]
            prior = max(earlier) if earlier else None
            convergence.append([kernel, seed, last,
                                f"{curve[last]['psnr_db']:.2f}",
                                f"{curve[peak]['psnr_db']:.2f} @ {peak:,}",
                                fmt(curve[last]["psnr_db"] - curve[prior]["psnr_db"]
                                    if prior is not None else None, 2)])
    lines += [table(["Kernel", "Seed", "Final step", "Final val PSNR",
                     "Best val PSNR", "Last 2k ΔPSNR"], convergence), ""]

    if budget:
        lines += ["## Change from the earlier training budget", "",
                  "The position learning-rate schedule changed with the step budget, so "
                  "these differences measure the combined budget and schedule change.", "",
                  table(["Kernel", "Seed", "Steps", "ΔPSNR dB", "ΔSSIM", "ΔLPIPS"], [
                      [row["kernel"], row["seed"], f"{row['old_steps']:,}→{row['pilot_steps']:,}",
                       fmt(row["delta_psnr_db"]), fmt(row["delta_ssim"], 4),
                       fmt(row["delta_lpips"], 4)] for row in budget]), ""]

    cost = []
    for kernel in ("linear", "polynomial"):
        rows = [row for row in paired if row["kernel"] == kernel]
        memory_rows = [row for row in rows if row["peak_train_gpu_memory_mib"] is not None
                       and row["gaussian_peak_train_gpu_memory_mib"] is not None
                       and row["gpu_name"] is not None
                       and row["gpu_name"] == row["gaussian_gpu_name"]]
        memory_delta = (mean(row["peak_train_gpu_memory_mib"] /
                             row["gaussian_peak_train_gpu_memory_mib"] - 1
                             for row in memory_rows) * 100 if memory_rows else None)
        cost.append([kernel,
                     f"{mean(row['render_ms'] / row['gaussian_render_ms'] - 1 for row in rows) * 100:+.1f}%",
                     f"{mean(row['tile_pairs'] / row['gaussian_tile_pairs'] - 1 for row in rows) * 100:+.1f}%",
                     f"{mean(pilot[(kernel, seed)]['summary']['training_elapsed_seconds'] /
                                  pilot[('gaussian', seed)]['summary']['training_elapsed_seconds'] - 1
                                  for seed in args.seeds) * 100:+.1f}%",
                     (f"{mean(row['peak_train_gpu_memory_mib'] for row in memory_rows):.0f}"
                      if memory_rows else "—"),
                     (f"{mean(row['peak_test_gpu_memory_mib'] for row in memory_rows if row['peak_test_gpu_memory_mib'] is not None):.0f}"
                      if any(row["peak_test_gpu_memory_mib"] is not None for row in memory_rows) else "—"),
                     f"{memory_delta:+.1f}%" if memory_delta is not None else "—"])
    lines += ["## Rendering cost", "",
              "Paired mean relative to Gaussian at the same seed. GPU memory is a "
              "250 ms sampled per-process peak and includes CUDA context and all live allocations:", "",
              table(["Kernel", "Render time", "Tile pairs", "Training time",
                     "Train peak MiB", "Test peak MiB", "Train peak change"], cost), "",
              "## Reading the result", "",
              "Look first for agreement across seeds and across PSNR, SSIM, and LPIPS. "
              "Use the view counts to see whether a mean is driven by a few cameras. "
              "Use the final validation slope and the best-checkpoint gap to assess whether "
              "the fixed-step endpoint is stable. Treat the three-seed intervals as exploratory; "
              "this single scene cannot establish a result across real scenes.", "",
              "Data: [paired test deltas](paired_test_deltas.csv), "
              "[per-view deltas](per_view_deltas.csv), "
              "[budget changes](budget_changes.csv).", ""]
    report = output / "report.md"
    report.write_text("\n".join(lines))
    print(f"Wrote {report} and three plots")
    if warnings:
        print(f"{len(warnings)} comparability warning(s); read the report before interpreting deltas")


if __name__ == "__main__":
    main()

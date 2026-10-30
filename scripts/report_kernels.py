"""Report one or more kernel scenes from raw run files, optionally versus another budget.

Example:
    python scripts/report_kernels.py --input-root outputs-20k/kernels \
        --reference-root outputs/kernels --scenes bonsai bicycle counter
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


def plot_validation(path: Path, current, reference, seeds):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for metric, ax in zip(("psnr_db", "ssim"), axes):
        for kernel in KERNELS:
            for source, linestyle, alpha in ((reference, "--", 0.7), (current, "-", 1.0)):
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
                label = f"{kernel} ({'current' if source is current else 'earlier'})"
                ax.plot(common_steps, y, color=COLORS[kernel], linestyle=linestyle,
                        alpha=alpha, linewidth=2 if source is current else 1.5, label=label)
                if source is current:
                    for run in runs:
                        steps = sorted(run["validation"])
                        ax.plot(steps, [run["validation"][step][metric] for step in steps],
                                color=COLORS[kernel], alpha=0.16, linewidth=0.8)
        ax.set(xlabel="Training step", ylabel=f"Mean validation {LABELS[metric]}")
        ax.grid(alpha=0.2)
    axes[0].legend(fontsize=8, ncol=2)
    fig.suptitle("Validation trajectories: thick lines average seeds; faint lines are current seeds")
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
    fig.suptitle("Per-view differences; shading spans the selected seeds")
    fig.savefig(path, dpi=170)
    plt.close(fig)


def report_scene(args):
    output = args.output_dir / args.scene
    output.mkdir(parents=True, exist_ok=True)
    current = {(kernel, seed): load_run(args.input_root, args.scene, kernel, seed)
             for kernel in KERNELS for seed in args.seeds}
    reference = {(kernel, seed): load_run(args.reference_root, args.scene, kernel, seed)
                 if args.reference_root is not None else None
                 for kernel in KERNELS for seed in args.seeds}
    missing = [f"{kernel}/seed_{seed}" for (kernel, seed), run in current.items() if run is None]
    if missing:
        raise ValueError(f"Current runs missing: {', '.join(missing)}")
    exemplar = current[("gaussian", args.seeds[0])]
    current_settings = settings(exemplar)
    current_split = exemplar["split"]
    current_steps = exemplar["config"]["training"]["iterations"]
    current_schedule = exemplar["config"]["learning_rates"]["position_max_steps"]
    current_lpips_net = exemplar["summary"].get("lpips_net")
    warnings = []
    for (kernel, seed), run in current.items():
        if settings(run) != current_settings:
            warnings.append(f"Current settings differ: {kernel}/seed_{seed}")
        if run["split"] != current_split:
            warnings.append(f"Current split differs: {kernel}/seed_{seed}")
        if (run["config"]["training"]["iterations"] != current_steps
                or run["config"]["learning_rates"]["position_max_steps"] != current_schedule):
            warnings.append(f"Current training budget or schedule differs: {kernel}/seed_{seed}")
        if run["summary"].get("lpips_net") != current_lpips_net:
            warnings.append(f"Current LPIPS backbone differs: {kernel}/seed_{seed}")
    for (kernel, seed), run in reference.items():
        if run is None:
            continue
        changed = [key for key, value in current_settings.items()
                   if settings(run)[key] != value]
        if changed or run["split"] != current_split:
            detail = ", ".join(changed + (["split"] if run["split"] != current_split else []))
            warnings.append(f"Reference excluded from budget comparison: "
                            f"{kernel}/seed_{seed} differs in {detail}")
            reference[(kernel, seed)] = None

    paired, per_view, budget = [], [], []
    for seed in args.seeds:
        gaussian = current[("gaussian", seed)]
        for kernel in KERNELS:
            run = current[(kernel, seed)]
            old = reference[(kernel, seed)]
            if old:
                budget.append(dict(kernel=kernel, seed=seed,
                                   old_steps=old["config"]["training"]["iterations"],
                                   current_steps=run["config"]["training"]["iterations"],
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
                               gaussian_peak_test_gpu_memory_mib=gaussian["summary"].get(
                                   "peak_test_gpu_memory_mib"),
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
               "peak_test_gpu_memory_mib", "gaussian_peak_test_gpu_memory_mib",
               "gpu_name", "gaussian_gpu_name"], paired)
    write_csv(output / "per_view_deltas.csv",
              ["kernel", "seed", "frame", "delta_psnr_db", "delta_ssim", "delta_lpips"], per_view)
    write_csv(output / "budget_changes.csv",
              ["kernel", "seed", "old_steps", "current_steps", "delta_psnr_db",
               "delta_ssim", "delta_lpips"], budget)
    plot_validation(output / "validation_curves.png", current, reference, args.seeds)
    plot_paired(output / "paired_test_deltas.png", paired)
    plot_views(output / "per_view_psnr.png", per_view)

    reference_text = (f"; reference: {args.reference_root / args.scene}"
                      if args.reference_root is not None else "")
    lines = [f"# {args.scene} kernel report", "",
             f"Input: {args.input_root / args.scene}{reference_text}.",
             "", "## Coverage and comparability", "",
             f"All {len(current)} kernel/seed runs completed. "
             f"Training budget: {exemplar['config']['training']['iterations']:,} steps; "
             f"position learning-rate schedule: {exemplar['config']['learning_rates']['position_max_steps']:,} steps. "
             f"Resolution: {current_settings['width']}×{current_settings['height']}; "
             f"capacity: {current_settings['capacity']:,} splats.", ""]
    if warnings:
        lines += ["**Comparability warnings:**", ""] + [f"- {warning}" for warning in warnings] + [""]
    else:
        lines += ["Kernel/seed settings and splits match within this input"
                  + (" and against available reference runs." if args.reference_root else "."), ""]

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
                   f"{current[(kernel, seed)]['summary']['mean_psnr_db']:.2f}",
                   f"{current[(kernel, seed)]['summary']['mean_ssim']:.4f}",
                   (f"{current[(kernel, seed)]['summary']['mean_lpips']:.4f}"
                    if current[(kernel, seed)]["summary"].get("mean_lpips") is not None else "—")]
                  for seed in args.seeds for kernel in KERNELS]), ""]

    lines += ["## Paired final-test differences from Gaussian", "",
              "Positive PSNR/SSIM and negative LPIPS favor the alternative kernel. "
              "These are seed pairs; test views are not independent replicates.", "",
              table(["Kernel", "Seed", "ΔPSNR dB", "ΔSSIM", "ΔLPIPS"], [
                  [row["kernel"], row["seed"], fmt(row["delta_psnr_db"]),
                   fmt(row["delta_ssim"], 4), fmt(row["delta_lpips"], 4)]
                  for row in paired]), "",
              f"Across-seed means and 95% t intervals ({len(args.seeds)} selected seeds; "
              "small samples give wide intervals):", ""]
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
              "Faint lines show individual seeds; dashed lines show reference runs when supplied.", "",
              "![Validation curves](validation_curves.png)", ""]
    convergence = []
    for kernel in KERNELS:
        for seed in args.seeds:
            curve = current[(kernel, seed)]["validation"]
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
                      [row["kernel"], row["seed"], f"{row['old_steps']:,}→{row['current_steps']:,}",
                       fmt(row["delta_psnr_db"]), fmt(row["delta_ssim"], 4),
                       fmt(row["delta_lpips"], 4)] for row in budget]), ""]

    cost = []
    for kernel in ("linear", "polynomial"):
        rows = [row for row in paired if row["kernel"] == kernel]
        hardware_rows = [row for row in rows if row["gpu_name"] is not None
                         and row["gpu_name"] == row["gaussian_gpu_name"]]
        memory_rows = [row for row in rows if row["peak_train_gpu_memory_mib"] is not None
                       and row["gaussian_peak_train_gpu_memory_mib"] is not None
                       and row["gpu_name"] is not None
                       and row["gpu_name"] == row["gaussian_gpu_name"]]
        memory_delta = (mean(row["peak_train_gpu_memory_mib"] /
                             row["gaussian_peak_train_gpu_memory_mib"] - 1
                             for row in memory_rows) * 100 if memory_rows else None)
        render_change = mean(row["render_ms"] / row["gaussian_render_ms"] - 1
                             for row in hardware_rows)
        train_change = mean(current[(kernel, row["seed"])]["summary"]["training_elapsed_seconds"] /
                            current[("gaussian", row["seed"])]["summary"]["training_elapsed_seconds"] - 1
                            for row in hardware_rows)
        cost.append([kernel, f"{len(hardware_rows)}/{len(rows)}",
                     f"{render_change * 100:+.1f}%" if render_change is not None else "—",
                     f"{mean(row['tile_pairs'] / row['gaussian_tile_pairs'] - 1 for row in rows) * 100:+.1f}%",
                     f"{train_change * 100:+.1f}%" if train_change is not None else "—",
                     (f"{mean(row['peak_train_gpu_memory_mib'] for row in memory_rows):.0f}"
                      if memory_rows else "—"),
                     (f"{mean(row['peak_test_gpu_memory_mib'] for row in memory_rows if row['peak_test_gpu_memory_mib'] is not None):.0f}"
                      if any(row["peak_test_gpu_memory_mib"] is not None for row in memory_rows) else "—"),
                     f"{memory_delta:+.1f}%" if memory_delta is not None else "—"])
    lines += ["## Rendering cost", "",
              "Paired mean relative to Gaussian at the same seed. GPU memory is a "
              "250 ms sampled per-process peak and includes CUDA context and all live allocations:", "",
              table(["Kernel", "GPU-matched pairs", "Render time", "Tile pairs", "Training time",
                     "Train peak MiB", "Test peak MiB", "Train peak change"], cost), "",
              "## Reading the result", "",
              "Look first for agreement across seeds and across PSNR, SSIM, and LPIPS. "
              "Use the view counts to see whether a mean is driven by a few cameras. "
              "Use the final validation slope and the best-checkpoint gap to assess whether "
              "the fixed-step endpoint is stable. Treat small-seed intervals as exploratory; "
              "this single scene cannot establish a result across real scenes.", "",
              "Data: [paired test deltas](paired_test_deltas.csv), "
              "[per-view deltas](per_view_deltas.csv), "
              "[budget changes](budget_changes.csv).", ""]
    report = output / "report.md"
    report.write_text("\n".join(lines))
    print(f"Wrote {report} and three plots")
    if warnings:
        print(f"{len(warnings)} comparability warning(s); read the report before interpreting deltas")
    return paired


def write_overview(output: Path, scene_rows: dict[str, list[dict]]):
    rows = []
    for scene, paired in sorted(scene_rows.items()):
        for kernel in ("linear", "polynomial"):
            group = [row for row in paired if row["kernel"] == kernel]
            hardware = [row for row in group if row["gpu_name"] is not None
                        and row["gpu_name"] == row["gaussian_gpu_name"]]
            comparable = [row for row in group
                          if row["peak_train_gpu_memory_mib"] is not None
                          and row["gaussian_peak_train_gpu_memory_mib"] is not None
                          and row["gpu_name"] is not None
                          and row["gpu_name"] == row["gaussian_gpu_name"]]
            rows.append(dict(
                scene=scene, kernel=kernel, seeds=len(group),
                mean_delta_psnr_db=mean(row["delta_psnr_db"] for row in group),
                mean_delta_ssim=mean(row["delta_ssim"] for row in group),
                mean_delta_lpips=mean(row["delta_lpips"] for row in group
                                      if row["delta_lpips"] is not None),
                psnr_positive_seeds=sum(row["delta_psnr_db"] > 0 for row in group),
                ssim_positive_seeds=sum(row["delta_ssim"] > 0 for row in group),
                lpips_better_seeds=sum(row["delta_lpips"] < 0 for row in group
                                       if row["delta_lpips"] is not None),
                memory_matched_seeds=len(comparable),
                mean_delta_train_gpu_memory_mib=mean(
                    row["peak_train_gpu_memory_mib"]
                    - row["gaussian_peak_train_gpu_memory_mib"] for row in comparable),
                mean_delta_test_gpu_memory_mib=mean(
                    row["peak_test_gpu_memory_mib"]
                    - row["gaussian_peak_test_gpu_memory_mib"] for row in comparable
                    if row["peak_test_gpu_memory_mib"] is not None
                    and row["gaussian_peak_test_gpu_memory_mib"] is not None),
                mean_render_time_change_pct=100 * mean(
                    row["render_ms"] / row["gaussian_render_ms"] - 1 for row in hardware)
                    if hardware else None,
            ))
    fields = list(rows[0])
    write_csv(output / "overview.csv", fields, rows)

    fig, axes = plt.subplots(1, 3, figsize=(13, max(3.5, 0.55 * len(scene_rows) + 1.8)),
                             constrained_layout=True)
    scenes = sorted(scene_rows)
    for metric, ax in zip(METRICS, axes):
        for kernel, offset in (("linear", -0.12), ("polynomial", 0.12)):
            group = [row for row in rows if row["kernel"] == kernel
                     and row[f"mean_delta_{metric}"] is not None]
            positions = [scenes.index(row["scene"]) + offset for row in group]
            ax.scatter([row[f"mean_delta_{metric}"] for row in group], positions,
                       color=COLORS[kernel], s=48, label=kernel)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set(yticks=range(len(scenes)), yticklabels=scenes,
               xlabel=f"Mean paired test {LABELS[metric]} − Gaussian")
        ax.grid(axis="x", alpha=0.2)
    axes[0].legend(fontsize=8)
    fig.suptitle("Scene means across paired seeds; each scene is one unit")
    fig.savefig(output / "scene_deltas.png", dpi=170)
    plt.close(fig)

    report = ["# Kernel report across scenes", "",
              "Each point is a scene's mean across paired seeds. Positive PSNR/SSIM "
              f"and negative LPIPS favor the alternative. {len(scene_rows)} scenes are exploratory; "
              "test views are not independent scene replicates.", "",
              "![Scene-level test differences](scene_deltas.png)", "",
              table(["Scene", "Kernel", "ΔPSNR dB", "ΔSSIM", "ΔLPIPS",
                     "PSNR wins", "LPIPS wins", "GPU-matched seeds", "Δtrain GPU MiB"], [
                  [row["scene"], row["kernel"], fmt(row["mean_delta_psnr_db"]),
                   fmt(row["mean_delta_ssim"], 4), fmt(row["mean_delta_lpips"], 4),
                   f"{row['psnr_positive_seeds']}/{row['seeds']}",
                   f"{row['lpips_better_seeds']}/{row['seeds']}",
                   f"{row['memory_matched_seeds']}/{row['seeds']}",
                   fmt(row["mean_delta_train_gpu_memory_mib"], 1)] for row in rows]), "",
              "Memory differences use only pairs trained on the same GPU model. "
              "Render and training time also depend on GPU model, so compare them "
              "within matched hardware before drawing cost conclusions.", "",
              "## Scene details", ""]
    report += [f"- [{scene}]({scene}/report.md)" for scene in scenes]
    report += ["", "[Machine-readable overview](overview.csv).", ""]
    (output / "overview.md").write_text("\n".join(report))
    print(f"Wrote {output / 'overview.md'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True,
                        help="Directory containing <scene>/<kernel>/seed_<n>/ runs")
    parser.add_argument("--reference-root", type=Path,
                        help="Optional earlier budget with the same layout")
    parser.add_argument("--output-dir", type=Path,
                        help="Defaults to a reports/ directory beside input-root")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--scene", help="Report one scene")
    selection.add_argument("--scenes", nargs="+", help="Report selected scenes")
    parser.add_argument("--seeds", nargs="+", type=int,
                        help="Defaults to all Gaussian seed directories in each scene")
    args = parser.parse_args()
    if not args.input_root.is_dir():
        parser.error(f"Input root does not exist: {args.input_root}")
    args.output_dir = args.output_dir or args.input_root.parent / "reports"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    scenes = args.scenes or ([args.scene] if args.scene else sorted(
        path.name for path in args.input_root.iterdir() if path.is_dir()))
    if not scenes:
        parser.error("No scenes found")
    selected_seeds = args.seeds
    scene_rows = {}
    for scene in scenes:
        seed_root = args.input_root / scene / "gaussian"
        if not seed_root.is_dir():
            raise ValueError(f"No Gaussian runs for {scene}")
        args.scene = scene
        args.seeds = selected_seeds or sorted(
            int(path.name.removeprefix("seed_")) for path in seed_root.glob("seed_*")
            if path.is_dir() and path.name.removeprefix("seed_").isdigit())
        if not args.seeds:
            raise ValueError(f"No seeds found for {scene}")
        scene_rows[scene] = report_scene(args)
    write_overview(args.output_dir, scene_rows)


if __name__ == "__main__":
    main()

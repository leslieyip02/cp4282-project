"""Export per-scene seed-variability figures for the two main experiment matrices.

Run analyze_kernels.py for each matrix first, then pass its kernel_analysis/
directory. The figures use sample SD across three final-test seed means.
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "cp4282-mpl"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import LogLocator, FuncFormatter


METRICS = {
    "psnr_db": ("PSNR", "dB"),
    "ssim": ("SSIM", ""),
    "lpips": ("LPIPS (VGG)", ""),
}
SERIES = (
    ("Gaussian run SD", "gaussian", "#566574", "o", -0.22),
    ("Linear paired Δ SD", "linear", "#2563a6", "s", 0.0),
    ("Quadratic paired Δ SD", "polynomial", "#c3652e", "^", 0.22),
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def load_matrix(directory: Path) -> list[dict]:
    baselines = read_csv(directory / "gaussian_seed_baseline.csv")
    summaries = read_csv(directory / "kernel_summary.csv")
    by_kernel = {(row["scene"], row["kernel"]): row for row in summaries}
    if not baselines:
        raise ValueError(f"No scenes in {directory}")
    scenes = []
    for baseline in sorted(baselines, key=lambda row: row["scene"]):
        scene = baseline["scene"]
        if int(baseline["gaussian_seeds"]) != 3:
            raise ValueError(f"Expected three Gaussian seeds for {scene} in {directory}")
        if int(baseline["gaussian_lpips_seeds"]) != 3 or baseline["lpips_net"] != "vgg":
            raise ValueError(f"Expected three VGG LPIPS values for {scene} in {directory}")
        values = {"scene": scene, "gaussian": {}, "linear": {}, "polynomial": {}}
        for metric in METRICS:
            values["gaussian"][metric] = float(baseline[f"gaussian_sd_{metric}"])
        for kernel in ("linear", "polynomial"):
            row = by_kernel.get((scene, kernel))
            if row is None or int(row["seeds"]) != 3 or int(row["lpips_seeds"]) != 3:
                raise ValueError(f"Expected three paired seeds for {scene}/{kernel} in {directory}")
            if row["lpips_net"] != baseline["lpips_net"]:
                raise ValueError(f"LPIPS backbone mismatch for {scene}/{kernel}")
            for metric in METRICS:
                values[kernel][metric] = float(row[f"sd_delta_{metric}"])
        if any(values[kernel][metric] <= 0 for _, kernel, *_ in SERIES for metric in METRICS):
            raise ValueError(f"Log-scale plot requires positive SDs for {scene}")
        scenes.append(values)
    return scenes


def plot_metric(metric: str, synthetic: list[dict], real: list[dict], output: Path) -> None:
    name, unit = METRICS[metric]
    fig, axes = plt.subplots(1, 2, figsize=(13.4, 6.7), sharex=True)
    fig.subplots_adjust(left=0.10, right=0.98, top=0.78, bottom=0.20, wspace=0.25)
    all_values = [scene[kernel][metric] for matrix in (synthetic, real)
                  for scene in matrix for _, kernel, *_ in SERIES]
    for ax, title, matrix in zip(axes, ("NeRF-synthetic · 8k steps", "Mip-NeRF 360 · 20k steps"),
                                 (synthetic, real)):
        for index, scene in enumerate(matrix):
            ax.axhspan(index - 0.48, index + 0.48,
                       color="#f2f5f7" if index % 2 == 0 else "white", zorder=0)
        for label, kernel, color, marker, offset in SERIES:
            ax.scatter([scene[kernel][metric] for scene in matrix],
                       [index + offset for index in range(len(matrix))],
                       s=52, marker=marker, color=color, edgecolor="white",
                       linewidth=0.7, zorder=3, label=label)
        ax.set_yticks(range(len(matrix)), [scene["scene"].capitalize() for scene in matrix])
        ax.set_ylim(len(matrix) - 0.45, -0.55)
        ax.set_xscale("log")
        ax.set_xlim(min(all_values) * 0.68, max(all_values) * 1.48)
        ax.xaxis.set_major_locator(LogLocator(base=10, subs=(1,)))
        ax.xaxis.set_minor_locator(LogLocator(base=10, subs=(2, 5)))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
        ax.grid(axis="x", which="major", color="#cbd6de", linewidth=0.9)
        ax.grid(axis="x", which="minor", color="#e6ebef", linewidth=0.6)
        ax.set_axisbelow(True)
        ax.set_title(title, loc="left", fontsize=12, weight="bold")
        ax.set_xlabel(f"Sample SD {f'({unit})' if unit else ''} · logarithmic scale")
        ax.tick_params(axis="y", length=0)
        ax.spines[["top", "right", "left"]].set_visible(False)
    handles = [Line2D([0], [0], marker=marker, linestyle="", markersize=8,
                      markerfacecolor=color, markeredgecolor="white", label=label)
               for label, _, color, marker, _ in SERIES]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.90),
               ncol=3, frameon=False)
    fig.suptitle(f"Random-seed variability by scene · {name}", fontsize=16,
                 weight="bold", y=0.98)
    fig.text(0.5, 0.05,
             "Gaussian: SD of final-test run means. Linear/quadratic: SD of paired kernel − Gaussian differences. "
             "Three seeds per scene; test views are averaged within each run.",
             ha="center", va="center", fontsize=9, color="#45545f")
    for extension in ("png", "pdf"):
        fig.savefig(output / f"seed_variability_{metric}.{extension}",
                    dpi=240, bbox_inches="tight", pad_inches=0.18)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthetic-analysis", type=Path,
                        default=Path("outputs/kernel_analysis"))
    parser.add_argument("--real-analysis", type=Path,
                        default=Path("outputs-cap200k/kernel_analysis"))
    parser.add_argument("--output-dir", type=Path, default=Path("docs/figures"))
    args = parser.parse_args()
    synthetic = load_matrix(args.synthetic_analysis)
    real = load_matrix(args.real_analysis)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for metric in METRICS:
        plot_metric(metric, synthetic, real, args.output_dir)
        print(f"Wrote {args.output_dir / f'seed_variability_{metric}.png'} and PDF")


if __name__ == "__main__":
    main()

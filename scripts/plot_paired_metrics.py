"""Plot paired final-test SSIM and LPIPS differences by scene and seed.

Requires paired_runs.csv from analyze_kernels.py for the synthetic and real
matrices. Each circle is one paired seed; each diamond is the scene mean.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import os
from pathlib import Path
import statistics
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "cp4282-mpl"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


KERNELS = {"linear": "#2563eb", "polynomial": "#d65f0c"}
MATRICES = (("NeRF-synthetic · 8k steps · 100k cap", "synthetic"),
            ("Mip-NeRF 360 · 20k steps · 200k cap", "real"))
METRICS = {
    "ssim": ("delta_ssim", "Final-test SSIM minus paired Gaussian (higher is better)"),
    "lpips": ("delta_lpips", "Final-test LPIPS minus paired Gaussian (VGG; lower is better)"),
}


def load_paired(path: Path, metric: str) -> dict[str, dict[str, list[float]]]:
    field, _ = METRICS[metric]
    grouped = defaultdict(lambda: defaultdict(list))
    with path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            kernel = row["kernel"]
            if kernel not in KERNELS:
                continue
            if metric == "lpips" and row["lpips_net"] != "vgg":
                raise ValueError(f"Expected VGG LPIPS for {row['scene']}/{kernel}")
            if not row[field]:
                raise ValueError(f"Missing {field} for {row['scene']}/{kernel}/seed_{row['seed']}")
            grouped[row["scene"]][kernel].append(float(row[field]))
    if not grouped:
        raise ValueError(f"No paired results in {path}")
    for scene, kernels in grouped.items():
        for kernel in KERNELS:
            if len(kernels[kernel]) != 3:
                raise ValueError(f"Expected three paired seeds for {scene}/{kernel} in {path}")
    return grouped


def plot(metric: str, matrices: dict[str, dict], output: Path) -> None:
    _, xlabel = METRICS[metric]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.4), layout="constrained")
    for ax, (title, key) in zip(axes, MATRICES):
        grouped = matrices[key]
        scenes = sorted(grouped)
        values = []
        for index, scene in enumerate(scenes):
            for kernel, color in KERNELS.items():
                deltas = grouped[scene][kernel]
                center = statistics.mean(deltas)
                y = index + (-0.08 if kernel == "linear" else 0.08)
                ax.scatter(deltas, [y] * len(deltas), s=26, color=color,
                           alpha=0.42, edgecolors=color, linewidths=0.7, zorder=2)
                ax.scatter([center], [y], s=65, marker="D", color=color,
                           edgecolors="white", linewidths=0.7, zorder=3)
                values.extend(deltas)
        ax.axvline(0, color="#334155", linewidth=1)
        ax.set(yticks=range(len(scenes)), yticklabels=scenes, ylim=(len(scenes)-0.5, -0.5),
               xlabel=xlabel, title=title)
        low, high = min(values), max(values)
        pad = max((high - low) * 0.055, max(abs(low), abs(high)) * 0.025)
        ax.set_xlim(min(low - pad, -pad), max(high + pad, pad))
        ax.grid(axis="x", color="#e6e6e6", linewidth=0.8)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    for extension in ("png", "pdf"):
        fig.savefig(output / f"paired_{metric}_by_scene.{extension}", dpi=190)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthetic-analysis", type=Path,
                        default=Path("outputs/kernel_analysis"))
    parser.add_argument("--real-analysis", type=Path,
                        default=Path("outputs-cap200k/kernel_analysis"))
    parser.add_argument("--output-dir", type=Path, default=Path("docs/figures"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for metric in METRICS:
        matrices = {
            "synthetic": load_paired(args.synthetic_analysis / "paired_runs.csv", metric),
            "real": load_paired(args.real_analysis / "paired_runs.csv", metric),
        }
        plot(metric, matrices, args.output_dir)
        print(f"Wrote {args.output_dir / f'paired_{metric}_by_scene.png'} and PDF")


if __name__ == "__main__":
    main()

"""Render matched train/validation cameras from snapshot PLYs for the comparison site.

Run on the cluster, where the dataset and a CUDA GPU are available. Rerunning skips
existing images. Example:
    python scripts/export_kernel_views.py --input-root outputs-20k/kernels \
        --scene bonsai --data data/bonsai --seeds 17 --arch gpu
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
KERNELS = ("gaussian", "linear", "polynomial")


def selected_names(split: dict, train_count: int, val_count: int) -> dict[str, list[str]]:
    train = split["train"]
    if train_count:
        indices = sorted({round(i * (len(train) - 1) / max(train_count - 1, 1))
                          for i in range(train_count)})
        train = [train[index] for index in indices]
    else:
        train = []
    return {"train": train, "val": split["validation"][:val_count]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-root", type=Path,
                        help="Defaults to view_renders/ beside input-root")
    parser.add_argument("--seeds", nargs="+", type=int, default=[7, 17, 27])
    parser.add_argument("--steps", nargs="+", type=int,
                        help="Defaults to every saved step from 6k onward")
    parser.add_argument("--train-views", type=int, default=3)
    parser.add_argument("--validation-views", type=int, default=8)
    parser.add_argument("--arch", choices=("auto", "cpu", "gpu"), default="gpu")
    args = parser.parse_args()
    if args.train_views < 0 or args.validation_views < 0:
        parser.error("View counts must be nonnegative")
    output = args.output_root or args.input_root.parent / "view_renders"
    info = json.loads((args.data / "nb-info.json").read_text())
    image_root = args.data / info["loader_kwargs"]["images_path"]
    for seed in args.seeds:
        exemplar = args.input_root / args.scene / "gaussian" / f"seed_{seed}"
        split = json.loads((exemplar / "split.json").read_text())
        names = selected_names(split, args.train_views, args.validation_views)
        for kernel in KERNELS:
            run = args.input_root / args.scene / kernel / f"seed_{seed}"
            config = json.loads((run / "run_config.json").read_text())
            width, height = config["model"]["width"], config["model"]["height"]
            available = {int(path.stem.split("_")[-1]): path
                         for path in (run / "snapshots").glob("out_*.ply")}
            steps = args.steps if args.steps is not None else sorted(
                step for step in available if step >= 6000)
            missing = sorted(set(steps) - set(available))
            if missing:
                raise ValueError(f"Missing snapshots in {run}: {missing}")
            for split_name, frames in names.items():
                for frame in frames:
                    source = image_root / frame
                    destination = output / args.scene / "reference" / split_name / f"{Path(frame).stem}.png"
                    if not destination.exists():
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        with Image.open(source) as image:
                            image.convert("RGB").resize((width, height), Image.Resampling.LANCZOS).save(destination)
                if not frames:
                    continue
                for step in steps:
                    target = output / args.scene / kernel / f"seed_{seed}" / split_name
                    pending = [frame for frame in frames if not (
                        target / Path(frame).stem / f"out_{step:06d}.png").exists()]
                    if not pending:
                        continue
                    with tempfile.TemporaryDirectory(prefix="kernel-views-") as temp:
                        temp_path = Path(temp)
                        command = [sys.executable, str(ROOT / "3dgs_evaluator.py"),
                                   str(available[step]), str(args.data),
                                   "--split", split_name, "--frames", *pending,
                                   "--width", str(width), "--height", str(height),
                                   "--metrics", "psnr", "ssim", "--arch", args.arch,
                                   "--output-dir", str(temp_path / "renders"),
                                   "--csv", str(temp_path / "metrics.csv")]
                        print(f"{args.scene}/{kernel}/seed_{seed} {split_name} step {step}: "
                              f"{len(pending)} views", flush=True)
                        subprocess.run(command, check=True)
                        for frame in pending:
                            destination = target / Path(frame).stem / f"out_{step:06d}.png"
                            destination.parent.mkdir(parents=True, exist_ok=True)
                            shutil.copy2(temp_path / "renders" / f"{Path(frame).stem}.png", destination)


if __name__ == "__main__":
    main()

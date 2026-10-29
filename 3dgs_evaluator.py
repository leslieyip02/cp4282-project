"""Evaluate a trained 3DGS PLY against a held-out view set and report PSNR, SSIM and LPIPS.

Usage:
    python 3dgs_evaluator.py trained.ply data/lego
    python 3dgs_evaluator.py trained.ply data/lego --arch gpu --metrics psnr ssim

For every camera in the manifest this renders the PLY with the same tiled Warp pipeline training
uses, saves the render as a PNG, and writes one CSV row per view.

Three things are decided rather than duplicated into separate scripts:

Device. `--arch auto`, the default, uses CUDA when Warp reports it and CPU otherwise, so the same
command works on a laptop and on a GPU node. `--arch cpu` or `--arch gpu` forces the choice, and
forcing gpu without CUDA is an error rather than a silent fallback -- a run that quietly took
forty times longer than expected is worse than one that refused to start.

Appearance. A model trained with spherical harmonics writes a `<ply-stem>.sh.npz` sidecar. If one
is next to the PLY it is loaded and the evaluation renders degree-2 view-dependent colour;
otherwise the PLY's degree-0 colour is used. `--sh-degree 2` demands the sidecar and fails if it
is absent, `--sh-degree 0` ignores it, and `--sh-npz` points at one kept elsewhere.

Metrics. `--metrics` selects any of psnr, ssim and lpips; all three run by default. LPIPS needs
PyTorch, which is slow to import and not always installed, so it is imported only when asked for.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import math
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import warp as wp

_here = Path(__file__).resolve().parent
if str(_here) not in sys.path:
    sys.path.insert(0, str(_here))
# Shared modules sit beside this file in the assignment repository.
_shared = _here / "shared"
if str(_shared) not in sys.path:
    sys.path.insert(0, str(_shared))

from image_metrics import LPIPSMetric, ssim_rgb
from gaussian_set import GaussianSet
from real_scene import RealScene

# The base trainer is `3dgs_trainer_v1` in the course repo and `3dgs_trainer` in the assignment
# repo, where there is only one. Probing for both keeps this file identical in the two trees, the
# same way the `shared/` lookup above does.
for _candidate_module in ("3dgs_trainer_v1", "3dgs_trainer"):
    try:
        _base_trainer = importlib.import_module(_candidate_module)
        break
    except ModuleNotFoundError:
        continue
else:
    raise ModuleNotFoundError(
        "No trainer module found. This script expects 3dgs_trainer_v1.py or 3dgs_trainer.py "
        "beside it."
    )

DEFAULT_RESOLUTION = _base_trainer.DEFAULT_RESOLUTION
WarpImageTrainer = _base_trainer.WarpImageTrainer
load_views = _base_trainer.load_views

METRICS = ("psnr", "ssim", "lpips")


def psnr_from_mse(mse: float) -> float:
    return float("inf") if mse <= 0.0 else 10.0 * math.log10(1.0 / mse)


def resolve_device(arch: str) -> str:
    if arch == "cpu":
        return "cpu"
    if arch == "gpu":
        if not wp.is_cuda_available():
            raise RuntimeError(
                "--arch gpu needs a CUDA-enabled Warp installation. Use --arch cpu, or --arch "
                "auto to pick whichever is available."
            )
        return "cuda:0"
    return "cuda:0" if wp.is_cuda_available() else "cpu"


def build_trainer(args, scene, targets, cameras, focal, scene_radius, device):
    """The renderer, with spherical harmonics loaded when they apply. Returns (trainer, using_sh)."""
    capacity = len(scene.means)
    background = (
        np.ones(3, np.float32) if args.background == "white" else np.zeros(3, np.float32)
    )
    training_options = {"projection": {
        "max_radius_fraction": args.max_projected_radius_fraction,
    }}
    if args.run_config.get("compact_box") is not None:
        training_options["compact_box"] = args.run_config["compact_box"]
    base = WarpImageTrainer(
        targets, cameras, focal, scene_radius, capacity, capacity, device, 0,
        init_scene=scene, background=background,
        tile_pair_capacity=args.tile_pair_capacity,
        kernel=args.kernel,
        training_options=training_options,
    )
    sh_path = args.sh_npz if args.sh_npz is not None else args.ply.with_suffix(".sh.npz")
    if args.sh_degree == 2 and not sh_path.exists():
        raise FileNotFoundError(
            f"--sh-degree 2 requested, but no spherical-harmonic sidecar at {sh_path}."
        )
    if args.sh_degree == 0 or not sh_path.exists():
        return base, False

    with np.load(sh_path) as sh_file:
        sh_values = np.asarray(sh_file["sh_rest"], dtype=np.float32)
    if sh_values.shape != (capacity, 8, 3):
        raise ValueError(
            f"Expected {capacity} rows of degree-2 coefficients in {sh_path}, got "
            f"{sh_values.shape}. The sidecar and the PLY must come from the same run."
        )
    # Imported here, not at module scope: the spherical-harmonic trainer is a later version of
    # the course material, and the assignment repository ships only the base trainer. Loading it
    # eagerly would make this script unimportable there even for the degree-0 runs that never
    # touch it.
    try:
        SHTrainer = importlib.import_module("3dgs_trainer_v2").SHTrainer
    except ModuleNotFoundError as error:
        raise RuntimeError(
            f"{sh_path} holds spherical-harmonic coefficients, but the trainer that renders them "
            "is not available here. Pass --sh-degree 0 to evaluate the PLY's own colour instead."
        ) from error
    # The trainer takes targets and a loss configuration because it is built for training.
    # Evaluation only calls its renderer, so the objective here is irrelevant.
    trainer = SHTrainer(base, cameras, {"feature_rest": 0.0}, targets, {"mode": "mse"})
    trainer.sh_rest.assign(wp.array(sh_values.reshape(-1, 3), dtype=wp.vec3, device=device))
    return trainer, True


def evaluate(args: argparse.Namespace) -> None:
    config_paths = (args.ply.parent / "run_config.json",
                    args.ply.parent.parent / "run_config.json")
    run_config = {}
    for config_path in config_paths:
        if config_path.exists():
            run_config = json.loads(config_path.read_text())
            break
    args.run_config = run_config
    if args.kernel is None:
        args.kernel = run_config.get("experiment", {}).get("kernel", "gaussian")
    if args.max_projected_radius_fraction is None:
        args.max_projected_radius_fraction = run_config.get(
            "projection", {}).get("max_radius_fraction", 0.0)
    wp.init()
    device = resolve_device(args.arch)
    data = args.data.expanduser().resolve()
    real_scene = (RealScene(data, run_config.get("reporting", {}).get("eval_views", 8))
                  if (data / "nb-info.json").exists() else None)
    args.background = args.background or ("black" if real_scene is not None else "white")
    if real_scene is None:
        width = args.width if args.width is not None else args.resolution
        height = args.height if args.height is not None else args.resolution
    else:
        original_width, original_height, _ = next(iter(real_scene.cameras.values()))
        if args.width is None and args.height is not None:
            height = args.height
            width = 2 * round(height * original_width / original_height / 2)
        else:
            width = args.width or (2 * round(args.resolution / 2))
            height = args.height or (2 * round(width * original_height / original_width / 2))
    manifest_path = data / args.manifest if real_scene is None else data / (
        "test_list.txt" if args.split == "test" else "train_list.txt"
    )
    if real_scene is None and not manifest_path.exists():
        raise FileNotFoundError(
            f"{manifest_path} does not exist. A NeRF-synthetic download that only fetched "
            "transforms_train.json has no held-out split; fetch the full dataset, or pass "
            "--manifest transforms_train.json to evaluate against the training views instead."
        )

    scene = GaussianSet.from_ply(args.ply)
    background = (
        np.ones(3, np.float32) if args.background == "white" else np.zeros(3, np.float32)
    )
    if real_scene is None:
        targets, cameras, focal, distance, frame_paths = load_views(
            data, width, height, background, manifest_name=args.manifest
        )
    else:
        targets, cameras, focal, distance, frame_paths = real_scene.load_views(
            args.split, width, height,
        )
    scene_radius = real_scene.scene_radius if real_scene is not None else distance * 0.35
    trainer, using_sh = build_trainer(args, scene, targets, cameras, focal, scene_radius, device)

    wanted = list(dict.fromkeys(args.metrics))
    lpips_metric = LPIPSMetric(args.lpips_net) if "lpips" in wanted else None

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    print(
        f"Evaluating {len(scene.means):,} splats from {args.ply} against {len(frame_paths)} "
        f"view(s) in {manifest_path} at {width}x{height} on Warp {device}; "
        f"appearance={'degree-2 SH' if using_sh else 'degree-0 RGB'}; "
        f"kernel={args.kernel}; max_radius_fraction={args.max_projected_radius_fraction}; "
        f"metrics={', '.join(wanted)}.",
        flush=True,
    )

    fields = ["view", "file", "mse"] + [
        {"psnr": "psnr_db", "ssim": "ssim", "lpips": "lpips"}[m] for m in wanted
    ]
    rows = []
    for view_id, frame_path in enumerate(frame_paths):
        try:
            image, mse = trainer.render_single_view_image(view_id)
        except RuntimeError as error:
            raise RuntimeError(
                f"{error} (pass a larger --tile-pair-capacity to this script)"
            ) from error
        image = np.clip(image, 0.0, 1.0)
        row = {"view": view_id, "file": frame_path, "mse": mse}
        if "psnr" in wanted:
            row["psnr_db"] = psnr_from_mse(mse)
        if "ssim" in wanted:
            row["ssim"] = ssim_rgb(image, targets[view_id])
        if "lpips" in wanted:
            row["lpips"] = lpips_metric(image, targets[view_id])
        Image.fromarray(np.uint8(image * 255.0)).save(
            args.output_dir / f"{Path(frame_path).stem}.png"
        )
        rows.append(row)
        reported = " ".join(
            f"{label}={row[key]:{fmt}}"
            for label, key, fmt in (
                ("psnr", "psnr_db", "6.2f"), ("ssim", "ssim", ".5f"), ("lpips", "lpips", ".5f"),
            )
            if key in row
        )
        print(f"  [{view_id:4d}] {frame_path:30s} mse={mse:.6f} {reported}", flush=True)

    with args.csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    means = {key: float(np.mean([row[key] for row in rows])) for key in fields[2:]}
    summary = ", ".join(
        f"mean {key}={value:.5f}" if key != "psnr_db" else f"mean psnr={value:.2f} dB"
        for key, value in means.items()
    )
    print(
        f"\n{len(rows)} view(s): {summary}. "
        f"Renders in {args.output_dir}, per-view results in {args.csv}.",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("ply", type=Path, help="Trained splats to evaluate.")
    parser.add_argument("data", type=Path, help="Dataset directory, e.g. data/lego.")
    parser.add_argument(
        "--manifest", default="transforms_test.json",
        help="NeRF-synthetic manifest. Default: the held-out test split.",
    )
    parser.add_argument("--split", choices=("train", "val", "test"), default="test",
                        help="Mip-NeRF 360 split; default: test.")
    parser.add_argument(
        "--metrics", nargs="+", choices=METRICS, default=list(METRICS),
        metavar="METRIC", help=f"Any of: {', '.join(METRICS)}. Default: all three.",
    )
    parser.add_argument(
        "--lpips-net", choices=("alex", "vgg", "squeeze"), default="alex",
        help="Backbone for LPIPS. Ignored unless lpips is among --metrics.",
    )
    parser.add_argument(
        "--arch", choices=("auto", "cpu", "gpu"), default="auto",
        help="auto uses CUDA when Warp reports it, and CPU otherwise.",
    )
    parser.add_argument("--resolution", type=int, default=DEFAULT_RESOLUTION)
    parser.add_argument("--width", type=int, default=None, help="Overrides --resolution.")
    parser.add_argument("--height", type=int, default=None, help="Overrides --resolution.")
    parser.add_argument("--background", choices=("white", "black"),
                        help="Default: black for COLMAP scenes, white for synthetic scenes.")
    parser.add_argument(
        "--sh-degree", type=int, choices=(0, 2), default=None,
        help="0 evaluates the PLY's own colour; 2 requires a spherical-harmonic sidecar. "
             "Default: use the sidecar when one exists.",
    )
    parser.add_argument(
        "--sh-npz", type=Path, default=None,
        help="Sidecar path. Default: the PLY's name with a .sh.npz suffix.",
    )
    parser.add_argument("--tile-pair-capacity", type=int, default=None)
    parser.add_argument("--kernel", choices=("gaussian", "linear", "polynomial"), default=None,
                        help="Defaults to the PLY run_config.json kernel, or gaussian.")
    parser.add_argument("--max-projected-radius-fraction", type=float, default=None,
                        help="Cull splats whose projected support exceeds this fraction of the "
                             "larger image dimension. Defaults to the run config, or 0 (off).")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/eval"))
    parser.add_argument("--csv", type=Path, default=Path("outputs/eval/metrics.csv"))
    evaluate(parser.parse_args())


if __name__ == "__main__":
    main()

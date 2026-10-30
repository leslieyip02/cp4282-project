"""Build a portable static kernel comparison site from copied experiment runs.

Example:
    python scripts/build_kernel_site.py --input-root outputs-20k/kernels
The generated site is outputs-20k/site/ by default. Serve its parent with
`python -m http.server` or serve the site directory directly.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import shutil

import numpy as np
from plyfile import PlyData

ROOT = Path(__file__).resolve().parents[1]
KERNELS = ("gaussian", "linear", "polynomial")
SCALE_EDGES = np.linspace(-6, 2, 33)
OPACITY_EDGES = np.linspace(0, 1, 33)


def copy_asset(source: Path, output: Path, relative: Path) -> str:
    destination = output / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    if (not destination.exists() or destination.stat().st_size != source.stat().st_size
            or destination.stat().st_mtime_ns != source.stat().st_mtime_ns):
        shutil.copy2(source, destination)
    return relative.as_posix()


def snapshot_stats(path: Path) -> dict:
    vertex = PlyData.read(str(path))["vertex"].data
    count = len(vertex)
    scales = np.stack([np.exp(np.asarray(vertex[f"scale_{axis}"], np.float64))
                       for axis in range(3)], axis=1)
    geometric_scale = np.cbrt(np.prod(scales, axis=1))
    log_scale = np.log10(np.maximum(geometric_scale, 1e-12))
    logits = np.asarray(vertex["opacity"], np.float64)
    opacity = 1 / (1 + np.exp(-np.clip(logits, -80, 80)))
    return {
        "count": count,
        "medianScale": float(np.median(geometric_scale)),
        "medianOpacity": float(np.median(opacity)),
        "opaqueCount": int(np.count_nonzero(opacity >= 0.5)),
        "scaleHistogram": np.histogram(log_scale, bins=SCALE_EDGES)[0].tolist(),
        "opacityHistogram": np.histogram(opacity, bins=OPACITY_EDGES)[0].tolist(),
    }


def validation_curve(path: Path) -> dict:
    groups: dict[int, list[dict]] = {}
    with path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            groups.setdefault(int(row["iteration"]), []).append(row)
    return {str(step): {
        "psnr": float(np.mean([float(row["psnr_db"]) for row in rows])),
        "ssim": float(np.mean([float(row["ssim"]) for row in rows])),
        "count": int(round(np.mean([int(row["gaussians"]) for row in rows]))),
    } for step, rows in sorted(groups.items())}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--view-root", type=Path,
                        help="Defaults to view_renders/ beside input-root")
    parser.add_argument("--output-dir", type=Path,
                        help="Defaults to site/ beside input-root")
    parser.add_argument("--scenes", nargs="+", help="Defaults to all scene directories")
    parser.add_argument("--seeds", nargs="+", type=int,
                        help="Defaults to all Gaussian seeds per scene")
    args = parser.parse_args()
    input_root = args.input_root.resolve()
    view_root = (args.view_root or input_root.parent / "view_renders").resolve()
    output = (args.output_dir or input_root.parent / "site").resolve()
    output.mkdir(parents=True, exist_ok=True)
    scenes = args.scenes or sorted(path.name for path in input_root.iterdir() if path.is_dir())
    data = {"version": 1, "scaleEdges": SCALE_EDGES.tolist(),
            "opacityEdges": OPACITY_EDGES.tolist(), "scenes": {}}
    for scene in scenes:
        seed_root = input_root / scene / "gaussian"
        seeds = args.seeds or sorted(int(path.name[5:]) for path in seed_root.glob("seed_*")
                                     if path.is_dir() and path.name[5:].isdigit())
        scene_data = {"seeds": seeds, "views": {}, "runs": {kernel: {} for kernel in KERNELS}}
        for seed in seeds:
            exemplar = input_root / scene / "gaussian" / f"seed_{seed}"
            split = json.loads((exemplar / "split.json").read_text())
            fixed_name = split["validation"][0]
            fixed_key = f"val:{fixed_name}"
            for split_name, names in (("train", split["train"]),
                                      ("val", split["validation"])):
                for name in names:
                    reference = view_root / scene / "reference" / split_name / f"{Path(name).stem}.png"
                    key = f"{split_name}:{name}"
                    if reference.is_file():
                        url = copy_asset(reference, output,
                                         Path("assets") / scene / "reference" / split_name / reference.name)
                    else:
                        url = None
                    if key == fixed_key or reference.is_file():
                        scene_data["views"][key] = {"name": name, "split": split_name,
                                                    "reference": url}
            for kernel in KERNELS:
                run = input_root / scene / kernel / f"seed_{seed}"
                if not run.is_dir():
                    raise FileNotFoundError(run)
                summary = json.loads((run / "test_summary.json").read_text())
                config = json.loads((run / "run_config.json").read_text())
                run_data = {"summary": summary, "steps": {}, "curve": validation_curve(
                    run / "validation_metrics.csv"),
                    "width": config["model"]["width"],
                    "height": config["model"]["height"]}
                for ply in sorted((run / "snapshots").glob("out_*.ply")):
                    step = str(int(ply.stem.split("_")[-1]))
                    entry = {"stats": snapshot_stats(ply), "images": {}}
                    fixed_image = ply.with_suffix(".png")
                    if fixed_image.is_file():
                        entry["images"][fixed_key] = copy_asset(
                            fixed_image, output,
                            Path("assets") / scene / kernel / f"seed_{seed}" /
                            "fixed" / fixed_image.name)
                    export_base = view_root / scene / kernel / f"seed_{seed}"
                    for split_name in ("train", "val"):
                        split_dir = export_base / split_name
                        if not split_dir.is_dir():
                            continue
                        for view_dir in split_dir.iterdir():
                            if not view_dir.is_dir():
                                continue
                            image = view_dir / f"out_{int(step):06d}.png"
                            if not image.is_file():
                                continue
                            matches = [key for key, view in scene_data["views"].items()
                                       if view["split"] == split_name
                                       and Path(view["name"]).stem == view_dir.name]
                            for key in matches:
                                entry["images"][key] = copy_asset(
                                    image, output, Path("assets") / scene / kernel /
                                    f"seed_{seed}" / split_name / view_dir.name / image.name)
                    run_data["steps"][step] = entry
                scene_data["runs"][kernel][str(seed)] = run_data
        data["scenes"][scene] = scene_data
        print(f"Prepared {scene}: {len(seeds)} seeds, {len(scene_data['views'])} available views")
    for name in ("index.html", "styles.css", "app.js"):
        shutil.copy2(ROOT / "site" / name, output / name)
    (output / "data.json").write_text(json.dumps(data, separators=(",", ":")))
    print(f"Built {output / 'index.html'}")


if __name__ == "__main__":
    main()

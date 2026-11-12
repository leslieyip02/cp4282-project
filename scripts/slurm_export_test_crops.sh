#!/bin/bash
#SBATCH --job-name=gs-test-crops
#SBATCH --gres=gpu:nv:1
#SBATCH --constraint=xgpd|xgpf
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${SLURM_SUBMIT_DIR:-$repo_root}"

if [[ $# -ne 5 ]]; then
  echo "Usage: sbatch scripts/slurm_export_test_crops.sh DATA_DIR OUTPUT_ROOT TEST_FRAME SEED LEFT,TOP,RIGHT,BOTTOM" >&2
  exit 2
fi

data_dir="$1"
output_root="$2"
frame="$3"
seed="$4"
crop="$5"
scene="${DATASET_NAME:-$(basename "$data_dir")}" 
frame_stem="$(basename "$frame")"
frame_stem="${frame_stem%.*}"
destination="$output_root/test_crops/$scene/seed_$seed/$frame_stem"

if [[ -z "${PYTHON_BIN:-}" && -x .venv/bin/python ]]; then
  python_bin="$PWD/.venv/bin/python"
else
  python_bin="${PYTHON_BIN:-python}"
fi

if [[ ! "$seed" =~ ^[0-9]+$ ]]; then
  echo "SEED must be an integer: $seed" >&2
  exit 2
fi

size=$("$python_bin" - "$data_dir" "$output_root" "$scene" "$frame" "$seed" "$crop" <<'PY'
import csv
import json
import sys
from pathlib import Path

data, root, scene, frame, seed, crop = sys.argv[1:]
data, root = Path(data), Path(root)
seed = int(seed)
if not data.is_dir():
    raise SystemExit(f"Dataset directory missing: {data}")
real = (data / "nb-info.json").is_file()
if real:
    split = json.loads((root / "kernels" / scene / "gaussian" /
                        f"seed_{seed}" / "split.json").read_text())
    if frame not in split["test"]:
        raise SystemExit(f"Frame is not in the recorded test split: {frame}")
else:
    manifest = json.loads((data / "transforms_test.json").read_text())
    if frame not in {item["file_path"] for item in manifest["frames"]}:
        raise SystemExit(f"Frame is not in transforms_test.json: {frame}")

sizes = set()
for kernel in ("gaussian", "linear", "polynomial"):
    run = root / "kernels" / scene / kernel / f"seed_{seed}"
    if not (run / "final.ply").is_file():
        raise SystemExit(f"Final checkpoint missing: {run / 'final.ply'}")
    config = json.loads((run / "run_config.json").read_text())
    if config["experiment"]["kernel"] != kernel or config["runtime"]["seed"] != seed:
        raise SystemExit(f"Run label differs from saved config: {run}")
    sizes.add((config["model"]["width"], config["model"]["height"]))
    with (run / "test_metrics.csv").open(newline="") as stream:
        if frame not in {row["frame"] for row in csv.DictReader(stream)}:
            raise SystemExit(f"Frame is missing from saved final-test metrics: {run}")
    if real:
        other_split = json.loads((run / "split.json").read_text())
        if other_split != split:
            raise SystemExit(f"Recorded camera split differs: {run}")
if len(sizes) != 1:
    raise SystemExit(f"Run render dimensions differ: {sizes}")
width, height = sizes.pop()
try:
    left, top, right, bottom = (int(part) for part in crop.split(","))
except ValueError as error:
    raise SystemExit("Crop must be LEFT,TOP,RIGHT,BOTTOM") from error
if not (0 <= left < right <= width and 0 <= top < bottom <= height):
    raise SystemExit(f"Crop {crop} lies outside the saved {width}x{height} image")
print(width, height)
PY
)
read -r width height <<< "$size"
echo "Validated $scene seed $seed test frame $frame at ${width}x${height}; crop $crop"

if [[ "${VALIDATE_ONLY:-0}" == 1 ]]; then
  exit 0
fi

"$python_bin" scripts/check_training_env.py --gpu
for kernel in gaussian linear polynomial; do
  run="$output_root/kernels/$scene/$kernel/seed_$seed"
  "$python_bin" 3dgs_evaluator.py "$run/final.ply" "$data_dir" \
    --split test --manifest transforms_test.json --frames "$frame" \
    --width "$width" --height "$height" --arch gpu --metrics psnr \
    --output-dir "$destination/full/$kernel" \
    --csv "$destination/full/$kernel/metrics.csv"
done

"$python_bin" - "$data_dir" "$frame" "$destination" "$width" "$height" "$crop" <<'PY'
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

data, frame, destination, width, height, crop = sys.argv[1:]
data, destination = Path(data), Path(destination)
width, height = int(width), int(height)
box = tuple(int(part) for part in crop.split(","))
stem = Path(frame).stem
real = (data / "nb-info.json").is_file()

if real:
    info = json.loads((data / "nb-info.json").read_text())
    source = data / info["loader_kwargs"]["images_path"] / frame
    with Image.open(source) as image:
        reference = image.convert("RGB").resize((width, height), Image.Resampling.LANCZOS)
else:
    base = data / frame
    candidates = ([base] if base.suffix else []) + [
        base.with_suffix(suffix) for suffix in (".png", ".jpg", ".jpeg")
    ]
    source = next((path for path in candidates if path.is_file()), None)
    if source is None:
        raise SystemExit(f"Synthetic reference image missing for {frame}")
    with Image.open(source) as image:
        rgba = np.asarray(image.convert("RGBA").resize(
            (width, height), Image.Resampling.LANCZOS), np.float32) / 255.0
    alpha = rgba[..., 3:4]
    rgb = rgba[..., :3] * alpha + (1.0 - alpha)
    reference = Image.fromarray(np.uint8(np.clip(rgb * 255.0, 0, 255)))

reference_path = destination / "full" / "reference" / f"{stem}.png"
reference_path.parent.mkdir(parents=True, exist_ok=True)
reference.save(reference_path)

labels = ("reference", "gaussian", "linear", "polynomial")
crop_dir = destination / "crops"
crop_dir.mkdir(parents=True, exist_ok=True)
crops = []
for label in labels:
    source_image = destination / "full" / label / f"{stem}.png"
    with Image.open(source_image) as image:
        cropped = image.convert("RGB").crop(box)
    cropped.save(crop_dir / f"{label}.png")
    crops.append(cropped)

crop_width, crop_height = crops[0].size
padding, header = 8, 24
comparison = Image.new("RGB", (5 * padding + 4 * crop_width,
                               2 * padding + header + crop_height), "white")
draw = ImageDraw.Draw(comparison)
font = ImageFont.load_default()
for index, (label, image) in enumerate(zip(labels, crops)):
    x = padding + index * (crop_width + padding)
    draw.text((x, padding), {"reference": "Reference", "gaussian": "Gaussian",
                             "linear": "Linear radial", "polynomial": "Quadratic"}[label],
              fill="black", font=font)
    comparison.paste(image, (x, padding + header))
comparison.save(destination / "comparison.png")

(destination / "metadata.json").write_text(json.dumps({
    "scene": destination.parent.parent.name,
    "seed": destination.parent.name,
    "frame": frame,
    "split": "test",
    "checkpoint": "final.ply",
    "render_size": [width, height],
    "crop_box": box,
    "reference_source": str(source),
}, indent=2) + "\n")
print(f"Saved four crops and comparison image in {destination}")
PY

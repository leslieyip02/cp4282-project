"""Show kernel differences and their change in squared error on test crops."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
VIEWS = (
    ("outputs", "lego", "r_0", "./test/r_0"),
    ("outputs", "lego", "r_1", "./test/r_1"),
    ("outputs-cap200k", "kitchen", "DSCF0744", "DSCF0744.JPG"),
    ("outputs-cap200k", "kitchen", "DSCF0656", "DSCF0656.JPG"),
)
DIFF_GAIN = 6


def font(size: int) -> ImageFont.FreeTypeFont:
    path = Path("/System/Library/Fonts/Supplemental/Arial.ttf")
    if not path.exists():
        path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    return ImageFont.truetype(str(path), size)


def full_psnr(root: str, scene: str, kernel: str, frame: str) -> float:
    path = ROOT / root / "kernels" / scene / kernel / "seed_17" / "test_metrics.csv"
    with path.open(newline="") as handle:
        return next(float(row["psnr_db"]) for row in csv.DictReader(handle) if row["frame"] == frame)


def crop_psnr(reference: np.ndarray, render: np.ndarray) -> float:
    mse = np.mean((reference - render) ** 2)
    return -10 * np.log10(mse) if mse else float("inf")


def signed_error_map(delta: np.ndarray, limit: float) -> Image.Image:
    # Blue = less squared error than Gaussian; red = more squared error.
    magnitude = np.clip(np.abs(delta) / limit, 0, 1)[..., None]
    white = np.full((*delta.shape, 3), 255.0)
    target = np.where(delta[..., None] >= 0, [205, 35, 42], [35, 95, 205])
    return Image.fromarray(np.uint8(np.rint(white * (1 - magnitude) + target * magnitude)))


def render_view(root: str, scene: str, view: str, frame: str) -> None:
    folder = ROOT / root / "test_crops" / scene / "seed_17" / view
    images = {
        key: np.asarray(Image.open(folder / "crops" / f"{key}.png").convert("RGB"), dtype=np.float64) / 255
        for key in ("reference", "gaussian", "linear", "polynomial")
    }
    ref, gauss = images["reference"], images["gaussian"]
    err_g = np.mean((gauss - ref) ** 2, axis=2)
    deltas = {key: np.mean((images[key] - ref) ** 2, axis=2) - err_g for key in ("linear", "polynomial")}
    limit = max(float(np.percentile(np.abs(np.stack(list(deltas.values()))), 95)), 1e-8)

    h, w = ref.shape[:2]
    scale = min(3, max(2, 420 // max(w, h)))
    cell_w, cell_h = w * scale, h * scale
    gap, margin, label_h, top_h, footer_h = 20, 24, 45, 86, 84
    width = margin * 2 + 3 * cell_w + 2 * gap
    height = top_h + 3 * (label_h + cell_h) + 2 * gap + footer_h
    canvas = Image.new("RGB", (width, height), "#f6f7f9")
    draw = ImageDraw.Draw(canvas)
    draw.text((margin, 14), f"{scene} / {view} — seed 17", font=font(25), fill="#1b2633")
    draw.text((margin, 48), "Pixel difference ×6  |  error change relative to Gaussian", font=font(17), fill="#42536a")

    def panel(row: int, col: int, title: str, pixels: Image.Image) -> None:
        x = margin + col * (cell_w + gap)
        y = top_h + row * (label_h + cell_h + gap)
        draw.text((x, y + 7), title, font=font(17), fill="#1b2633")
        canvas.paste(pixels.resize((cell_w, cell_h), Image.Resampling.NEAREST), (x, y + label_h))

    panel(0, 0, "Reference", Image.fromarray(np.uint8(images["reference"] * 255)))
    panel(0, 1, "Gaussian", Image.fromarray(np.uint8(images["gaussian"] * 255)))
    base_error = np.clip(np.sqrt(err_g) * DIFF_GAIN, 0, 1)
    panel(0, 2, "Gaussian error ×6", Image.fromarray(np.uint8(np.rint(base_error * 255))))

    g_psnr = full_psnr(root, scene, "gaussian", frame)
    for row, (key, name) in enumerate((("linear", "Linear"), ("polynomial", "Quadratic")), 1):
        alt = images[key]
        diff = np.uint8(np.rint(np.clip(np.abs(alt - gauss) * DIFF_GAIN, 0, 1) * 255))
        full_delta = full_psnr(root, scene, key, frame) - g_psnr
        crop_delta = crop_psnr(ref, alt) - crop_psnr(ref, gauss)
        panel(row, 0, f"{name}  |  full ΔPSNR {full_delta:+.3f} dB", Image.fromarray(np.uint8(alt * 255)))
        panel(row, 1, f"|{name} − Gaussian| ×6", Image.fromarray(diff))
        panel(row, 2, f"Δ error  |  crop {crop_delta:+.3f} dB", signed_error_map(deltas[key], limit))

    y = height - footer_h + 7
    draw.text((margin, y), "Blue: lower error (better)     Red: higher error (worse)     White: little change", font=font(17), fill="#243347")
    draw.text((margin, y + 29), "Squared-error color saturates at each view's 95th percentile; crop PSNR uses saved PNGs.", font=font(15), fill="#536176")
    out = ROOT / "docs" / "figures" / f"test_diff_{scene}_{view}.png"
    canvas.save(out, optimize=True)
    print(out.relative_to(ROOT))


if __name__ == "__main__":
    for args in VIEWS:
        render_view(*args)

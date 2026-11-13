"""Compare final validation renders and pixel differences for two real scenes.

Uses the saved seed-17, step-20,000 PNGs in outputs-cap200k/view_renders/.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "outputs-cap200k" / "view_renders"
OUTPUT = ROOT / "docs" / "figures"
VIEWS = (("garden", "DSC08061"), ("treehill", "_DSC8894"))
KERNELS = ("gaussian", "linear", "polynomial")
DIFF_GAIN = 6
ERROR_GAIN = 4


def font(size: int) -> ImageFont.FreeTypeFont:
    candidates = (Path("/System/Library/Fonts/Supplemental/Arial.ttf"),
                  Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"))
    return ImageFont.truetype(str(next(path for path in candidates if path.exists())), size)


def signed_error_map(delta: np.ndarray, limit: float) -> Image.Image:
    """Blue is less squared RGB error than Gaussian; red is more."""
    magnitude = np.clip(np.abs(delta) / limit, 0, 1)[..., None]
    white = np.full((*delta.shape, 3), 255.0)
    target = np.where(delta[..., None] >= 0, [205, 35, 42], [35, 95, 205])
    return Image.fromarray(np.uint8(np.rint(white * (1 - magnitude) + target * magnitude)))


def load_view(scene: str, frame: str) -> dict[str, np.ndarray]:
    paths = {"reference": INPUT / scene / "reference" / "val" / f"{frame}.png"}
    paths.update({kernel: INPUT / scene / kernel / "seed_17" / "val" /
                  frame / "out_020000.png" for kernel in KERNELS})
    images = {}
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        with Image.open(path) as image:
            images[name] = np.asarray(image.convert("RGB"), dtype=np.float64) / 255
    if len({image.shape for image in images.values()}) != 1:
        raise ValueError(f"Mismatched render sizes for {scene}/{frame}")
    return images


def render_view(scene: str, frame: str) -> Path:
    images = load_view(scene, frame)
    reference, gaussian = images["reference"], images["gaussian"]
    gaussian_error = np.mean((gaussian - reference) ** 2, axis=2)
    changes = {kernel: np.mean((images[kernel] - reference) ** 2, axis=2)
               - gaussian_error for kernel in ("linear", "polynomial")}
    # One color limit for both alternative kernels within a scene.
    limit = max(float(np.percentile(np.abs(np.stack(list(changes.values()))), 95)), 1e-8)

    height, width = reference.shape[:2]
    scale, margin, gap, label_height, top_height, footer_height = 2, 30, 24, 48, 110, 90
    cell_width, cell_height = width * scale, height * scale
    canvas = Image.new("RGB", (2 * margin + 3 * cell_width + 2 * gap,
                               top_height + 3 * (label_height + cell_height + gap)
                               - gap + footer_height), "#f6f7f9")
    draw = ImageDraw.Draw(canvas)
    draw.text((margin, 15), f"{scene.capitalize()} · {frame}.JPG · seed 17 · step 20,000",
              font=font(27), fill="#1b2633")
    draw.text((margin, 56), "Reference and final validation renders · pixel differences enlarged for inspection",
              font=font(19), fill="#42536a")

    def panel(row: int, column: int, title: str, pixels: Image.Image) -> None:
        x = margin + column * (cell_width + gap)
        y = top_height + row * (label_height + cell_height + gap)
        draw.text((x, y + 8), title, font=font(20), fill="#1b2633")
        canvas.paste(pixels.resize((cell_width, cell_height), Image.Resampling.NEAREST),
                     (x, y + label_height))

    def rgb_image(pixels: np.ndarray) -> Image.Image:
        return Image.fromarray(np.uint8(np.rint(np.clip(pixels, 0, 1) * 255)))

    panel(0, 0, "Reference", rgb_image(reference))
    panel(0, 1, "Gaussian", rgb_image(gaussian))
    panel(0, 2, f"Gaussian error ×{ERROR_GAIN}",
          rgb_image(np.repeat(np.clip(np.sqrt(gaussian_error)[..., None] * ERROR_GAIN,
                                      0, 1), 3, axis=2)))
    for row, (kernel, label) in enumerate((("linear", "Linear"),
                                           ("polynomial", "Quadratic")), start=1):
        panel(row, 0, label, rgb_image(images[kernel]))
        panel(row, 1, f"|{label} − Gaussian| ×{DIFF_GAIN}",
              rgb_image(np.abs(images[kernel] - gaussian) * DIFF_GAIN))
        panel(row, 2, "Squared-error change vs Gaussian",
              signed_error_map(changes[kernel], limit))

    footer_y = canvas.height - footer_height + 11
    draw.text((margin, footer_y),
              "Blue: lower pixel error     Red: higher pixel error     White: little change",
              font=font(18), fill="#243347")
    draw.text((margin, footer_y + 32),
              "Error-change color saturates at this scene's 95th percentile; visual quality needs direct inspection.",
              font=font(16), fill="#536176")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    output = OUTPUT / f"validation_diff_{scene}_{frame}.png"
    canvas.save(output, optimize=True)
    return output


if __name__ == "__main__":
    for scene, frame in VIEWS:
        print(render_view(scene, frame).relative_to(ROOT))

"""Generate a reproducible random 3DGS initialization for a NeRF-synthetic scene."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from shared.gaussian_set import C0, GaussianSet

DEFAULT_SEED = 7
DEFAULT_POINTS = 50_000
DEFAULT_BOUND = 1.3


def generate(data: Path, *, seed: int = DEFAULT_SEED, points: int = DEFAULT_POINTS,
             bound: float = DEFAULT_BOUND, overwrite: bool = False) -> Path:
    data = data.resolve()
    if not (data / "transforms_train.json").is_file():
        raise FileNotFoundError(f"NeRF-synthetic transforms_train.json missing in {data}")
    if points < 1:
        raise ValueError("points must be positive")
    if not np.isfinite(bound) or bound <= 0:
        raise ValueError("bound must be finite and positive")
    output = data / "init.ply"
    metadata = data / "init.json"
    if output.exists() and not overwrite:
        raise FileExistsError(f"{output} already exists; pass --overwrite to replace it")

    rng = np.random.default_rng(seed)
    means = rng.uniform(-bound, bound, (points, 3)).astype(np.float32)
    # Match the original 3DGS NeRF-synthetic random color initialization.
    colors = (0.5 + C0 * rng.random((points, 3)) / 255.0).astype(np.float32)
    scene = GaussianSet(
        means=means,
        scales=np.full((points, 3), bound / 80.0, np.float32),
        rotations=np.tile(np.array([1, 0, 0, 0], np.float32), (points, 1)),
        opacities=np.full(points, 1.0 / (1.0 + np.exp(4.0)), np.float32),
        colors=colors,
    )
    scene.to_ply(output)
    metadata.write_text(json.dumps({
        "method": "uniform_random_cube",
        "seed": seed,
        "points": points,
        "bound": bound,
        "coordinate_range": [-bound, bound],
        "scale": bound / 80.0,
        "opacity_logit": -4.0,
    }, indent=2) + "\n")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="NeRF-synthetic scene directory")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--points", type=int, default=DEFAULT_POINTS)
    parser.add_argument("--bound", type=float, default=DEFAULT_BOUND)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    print(generate(args.data, seed=args.seed, points=args.points,
                   bound=args.bound, overwrite=args.overwrite))


if __name__ == "__main__":
    main()

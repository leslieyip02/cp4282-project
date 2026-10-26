"""Prepare all six synthetic-scene warm-up target sets before training."""

from __future__ import annotations

import argparse
import importlib
from pathlib import Path

import numpy as np

from image_filters import METHODS
from prepared_targets import dataset_name, prepared_training_views

gs = importlib.import_module("3dgs_trainer")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", help="Cache label; defaults to the dataset directory name")
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=Path(__file__).resolve().parent / "config/synthetic.yaml")
    parser.add_argument("--width", type=int)
    parser.add_argument("--height", type=int)
    parser.add_argument("--background", choices=("white", "black"))
    parser.add_argument("--cache-dir", type=Path, default=Path(__file__).resolve().parent / "outputs/preprocessed")
    args = parser.parse_args()
    config = gs.TrainingConfig.load(args.config)
    model = config["model"]
    width = args.width or model["width"]
    height = args.height or model["height"]
    color = args.background or model["background"]
    scene = dataset_name(args.data, args.scene)
    background = np.ones(3, np.float32) if color == "white" else np.zeros(3, np.float32)
    targets, _, _, _, _ = gs.load_views(args.data, width, height, background)
    for method in METHODS[1:]:
        prepared = prepared_training_views(targets, method, scene, args.cache_dir)
        print(f"{scene}/{method}: {prepared.shape}", flush=True)


if __name__ == "__main__":
    main()

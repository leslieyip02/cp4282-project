"""Content-addressed copies of filtered synthetic training targets."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import re
import uuid

import numpy as np

from image_filters import filter_views


def dataset_name(data: Path, explicit: str | None = None) -> str:
    """Choose a safe output/cache label from a dataset path or explicit name."""
    name = explicit or Path(data).resolve().name
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", name) or name in (".", ".."):
        raise ValueError("Dataset name must contain only letters, digits, dots, underscores or hyphens")
    return name


def prepared_training_views(images: np.ndarray, method: str, scene: str,
                            cache_dir: Path) -> np.ndarray:
    """Return reproducible filtered targets, reusing a copy across seeds.

The cache key includes the canonical RGB targets after resizing and alpha
compositing, so a changed source image, resolution, or background invalidates
the old copy. Atomic rename lets concurrent cluster jobs share the cache.
    """
    if method == "none":
        return images
    images = np.ascontiguousarray(images, dtype=np.float32)
    digest = hashlib.sha256(images.tobytes()).hexdigest()[:16]
    cache_dir = Path(cache_dir) / scene
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"v1_{method}_{images.shape[2]}x{images.shape[1]}_{digest}.npy"
    if path.exists():
        result = np.load(path, allow_pickle=False)
    else:
        result = filter_views(images, method)
        temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
        try:
            with temporary.open("wb") as stream:
                np.save(stream, result, allow_pickle=False)
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
    expected = (
        (len(images), images.shape[1] // 2, images.shape[2] // 2, 3)
        if method == "downsample" else images.shape
    )
    if result.shape != expected or result.dtype != np.float32:
        raise ValueError(f"Invalid prepared target cache: {path}")
    return result

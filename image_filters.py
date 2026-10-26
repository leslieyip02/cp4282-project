"""Deterministic warm-up targets for the synthetic-scene experiment.

All filters operate on RGB images after alpha compositing and resizing to the
scene's configured full resolution. Intensities stay in the trainer's [0, 1]
colour space; no filter is applied to validation or test images.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter, uniform_filter

METHODS = (
    "none", "downsample", "mean", "gaussian", "bilateral", "kuwahara", "unsharp",
)


def _box_sum(image: np.ndarray, size: int, y: int, x: int, height: int, width: int):
    integral = np.pad(image, ((1, 0), (1, 0), (0, 0))).cumsum(0).cumsum(1)
    return (
        integral[y + size:y + size + height, x + size:x + size + width]
        - integral[y:y + height, x + size:x + size + width]
        - integral[y + size:y + size + height, x:x + width]
        + integral[y:y + height, x:x + width]
    )


def _kuwahara(image: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    padded = np.pad(image.astype(np.float64), ((3, 3), (3, 3), (0, 0)), mode="reflect")
    candidates = []
    variances = []
    for y, x in ((0, 0), (0, 3), (3, 0), (3, 3)):
        mean = _box_sum(padded, 4, y, x, height, width) / 16.0
        mean_square = _box_sum(padded * padded, 4, y, x, height, width) / 16.0
        candidates.append(mean)
        variances.append(np.maximum(mean_square - mean * mean, 0.0).mean(axis=2))
    choice = np.argmin(np.stack(variances), axis=0)
    return np.take_along_axis(np.stack(candidates), choice[None, ..., None], axis=0)[0]


def _bilateral(image: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    padded = np.pad(image, ((3, 3), (3, 3), (0, 0)), mode="reflect")
    numerator = np.zeros_like(image, dtype=np.float64)
    denominator = np.zeros((height, width, 1), dtype=np.float64)
    for dy in range(-3, 4):
        for dx in range(-3, 4):
            neighbour = padded[dy + 3:dy + 3 + height, dx + 3:dx + 3 + width]
            color_distance = np.sum((neighbour - image) ** 2, axis=2, keepdims=True)
            weight = np.exp(-(dx * dx + dy * dy) / (2 * 2.0**2)
                            - color_distance / (2 * 0.1**2))
            numerator += weight * neighbour
            denominator += weight
    return numerator / denominator


def filter_image(image: np.ndarray, method: str) -> np.ndarray:
    """Apply one proposal method to an H x W x 3 float image."""
    if method not in METHODS:
        raise ValueError(f"Unknown method {method!r}; choose from {METHODS}")
    image = np.asarray(image, dtype=np.float32)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image must have shape [height, width, 3]")
    if method == "none":
        result = image.copy()
    elif method == "downsample":
        height, width = image.shape[:2]
        if height % 2 or width % 2:
            raise ValueError("half-resolution area downsampling needs even width and height")
        result = image.reshape(height // 2, 2, width // 2, 2, 3).mean(axis=(1, 3))
    elif method == "mean":
        result = uniform_filter(image, size=(7, 7, 1), mode="reflect")
    elif method == "gaussian":
        result = gaussian_filter(image, sigma=(1.0, 1.0, 0.0), radius=(3, 3, 0), mode="reflect")
    elif method == "bilateral":
        result = _bilateral(image)
    elif method == "kuwahara":
        result = _kuwahara(image)
    else:
        blurred = gaussian_filter(image, sigma=(1.0, 1.0, 0.0), radius=(3, 3, 0), mode="reflect")
        result = image + (image - blurred)
    return np.ascontiguousarray(np.clip(result, 0.0, 1.0), dtype=np.float32)


def filter_views(images: np.ndarray, method: str) -> np.ndarray:
    """Filter training views only, preserving their manifest order."""
    return np.stack([filter_image(image, method) for image in images])

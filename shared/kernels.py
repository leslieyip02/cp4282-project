"""Screen-space splat kernels and their matching support radii.

All kernels reach half their central value at q = 2 log(2), where q is the
squared Mahalanobis distance in the projected ellipse.
"""

from __future__ import annotations

import math

import numpy as np

GAUSSIAN = 0
LINEAR = 1
POLYNOMIAL = 2
NAMES = ("gaussian", "linear", "polynomial")
LINEAR_RADIUS_SQUARED = 8.0 * math.log(2.0)
POLYNOMIAL_RADIUS_SQUARED = 2.0 * math.log(2.0) / (1.0 - 1.0 / math.sqrt(2.0))


def kernel_id(name: str) -> int:
    if name not in NAMES:
        raise ValueError(f"kernel must be one of {NAMES}")
    return NAMES.index(name)


def support_radius_squared(opacity, kind: int, compact: dict,
                           maximum: float, alpha_cutoff: float):
    """q support shared by CPU densification and GPU rasterizers.

    The compact-box beta scales the cutoff radius as in the original Gaussian
    implementation. With compact boxes disabled, retain the Gaussian's fixed
    q=maximum support; linear has its natural finite support.
    """
    opacity = np.asarray(opacity, dtype=np.float32)
    if kind == GAUSSIAN:
        natural = np.full_like(opacity, maximum)
    elif kind == LINEAR:
        natural = np.full_like(opacity, LINEAR_RADIUS_SQUARED)
    elif kind == POLYNOMIAL:
        natural = np.full_like(opacity, POLYNOMIAL_RADIUS_SQUARED)
    else:
        raise ValueError(f"unknown kernel id {kind}")
    if not compact["enabled"]:
        return natural
    threshold = max(float(compact["alpha_min"]), float(alpha_cutoff))
    positive = opacity > threshold
    ratio = np.zeros_like(opacity)
    np.divide(threshold, opacity, out=ratio, where=positive)
    if kind == GAUSSIAN:
        radius = 2.0 * float(compact["beta"]) * np.log(
            np.maximum(opacity / threshold, 1.0))
    elif kind == LINEAR:
        radius = LINEAR_RADIUS_SQUARED * (
            1.0 - np.power(ratio, float(compact["beta"]))) ** 2
    else:
        radius = POLYNOMIAL_RADIUS_SQUARED * (
            1.0 - np.power(ratio, float(compact["beta"]) / 2.0))
    return np.where(positive, np.minimum(natural, radius), 0.0).astype(np.float32)

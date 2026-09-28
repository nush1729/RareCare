"""Image quality checks that run before any model — cheap, explainable, numpy-only.

A learned quality head replaces the thresholds once trained; these heuristics stay
as a fallback and as the baseline in ablations.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

MIN_SIDE = 64
BLUR_MIN = 15.0  # variance of Laplacian on a 0-255 grayscale image
DARK_MAX_MEAN = 35.0
BRIGHT_MIN_MEAN = 235.0
GLARE_MAX_FRACTION = 0.25


@dataclass(frozen=True)
class QualityReport:
    ok: bool
    reasons: tuple[str, ...]
    metrics: dict[str, float]


def laplacian_variance(gray: NDArray[np.float64]) -> float:
    g = gray.astype(np.float64)
    lap = -4 * g[1:-1, 1:-1] + g[:-2, 1:-1] + g[2:, 1:-1] + g[1:-1, :-2] + g[1:-1, 2:]
    return float(lap.var())


def assess_quality(gray: NDArray[np.float64]) -> QualityReport:
    if gray.ndim != 2:
        raise ValueError("expected a 2-D grayscale array")
    h, w = gray.shape
    metrics = {
        "height": float(h),
        "width": float(w),
        "sharpness": laplacian_variance(gray) if min(h, w) >= 3 else 0.0,
        "brightness": float(gray.mean()),
        "glare_fraction": float((gray >= 250).mean()),
    }
    reasons: list[str] = []
    if min(h, w) < MIN_SIDE:
        reasons.append("The image is too small. Please use a higher-resolution picture.")
    if metrics["sharpness"] < BLUR_MIN:
        reasons.append("The image looks blurry. Hold the camera steady and tap to focus.")
    if metrics["brightness"] < DARK_MAX_MEAN:
        reasons.append("The image is too dark. Move to better light.")
    elif metrics["brightness"] > BRIGHT_MIN_MEAN:
        reasons.append("The image is overexposed. Avoid direct light or flash.")
    if metrics["glare_fraction"] > GLARE_MAX_FRACTION:
        reasons.append("There is strong glare. Tilt the camera or screen slightly.")
    return QualityReport(ok=not reasons, reasons=tuple(reasons), metrics=metrics)

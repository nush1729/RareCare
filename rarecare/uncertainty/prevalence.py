"""Label-shift correction when deployment prevalence differs from training prevalence.

Under label shift p(x|y) is fixed and only p(y) changes, so posteriors can be
re-weighted: p_t(y|x) ∝ p_s(y|x) · π_t(y) / π_s(y) (Saerens et al., 2002).
Used to move from dataset/UK-derived priors to regional (e.g. GLOBOCAN India)
incidence without retraining.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def adjust_posterior(
    probs: NDArray[np.float64],
    source_prior: NDArray[np.float64],
    target_prior: NDArray[np.float64],
) -> NDArray[np.float64]:
    probs = np.atleast_2d(np.asarray(probs, dtype=np.float64))
    src = np.asarray(source_prior, dtype=np.float64)
    tgt = np.asarray(target_prior, dtype=np.float64)
    if (src <= 0).any() or (tgt < 0).any():
        raise ValueError("priors must be positive (source) and non-negative (target)")
    weighted = probs * (tgt / src)
    out: NDArray[np.float64] = weighted / weighted.sum(axis=1, keepdims=True)
    return out

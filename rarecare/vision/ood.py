"""Out-of-distribution scoring on router logits (numpy)."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def energy_score(logits: NDArray[np.float64], temperature: float = 1.0) -> float:
    """E(x) = -T · logsumexp(f(x)/T). Lower energy = more in-distribution (Liu et al., 2020)."""
    z = np.asarray(logits, dtype=np.float64) / temperature
    m = z.max()
    return float(-temperature * (m + np.log(np.exp(z - m).sum())))


def is_ood(logits: NDArray[np.float64], threshold: float, temperature: float = 1.0) -> bool:
    """Threshold is chosen on validation data at 95% in-distribution TPR."""
    return energy_score(logits, temperature) > threshold

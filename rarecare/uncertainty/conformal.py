"""Class-conditional (Mondrian) split conformal prediction for screening.

The screening requirement is a bound on *misses of the positive class*, not
marginal coverage. We therefore calibrate a threshold using only positive
calibration examples (optionally per subgroup), which guarantees

    P(positive class in prediction set | y = positive, group = g) >= 1 - alpha

under exchangeability of calibration and test data within each group.

`or_rule` combines per-modality decisions: flag if *any* calibrated modality flags.
If modality m alone guarantees recall >= 1 - alpha on its positives, the OR of
flags keeps that recall (the flag set only grows). This is how the system keeps a
guarantee without paired text+image calibration data.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

GLOBAL = "__all__"


def conformal_quantile(scores: NDArray[np.float64], alpha: float) -> float:
    """Finite-sample-corrected (1 - alpha) quantile of nonconformity scores."""
    n = len(scores)
    if n == 0:
        raise ValueError("no calibration scores")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    rank = math.ceil((n + 1) * (1 - alpha))
    if rank > n:
        # Too few samples for this alpha: the only valid threshold is "always flag".
        return float("inf")
    return float(np.sort(scores)[rank - 1])


@dataclass
class PositiveClassConformal:
    """Flags a case as positive when 1 - p(positive) <= threshold."""

    alpha: float
    thresholds: dict[str, float] = field(default_factory=dict)
    n_calibration: dict[str, int] = field(default_factory=dict)

    def fit(
        self,
        p_positive: NDArray[np.float64],
        labels: NDArray[np.int64],
        groups: NDArray[np.str_] | None = None,
        min_group_size: int = 20,
    ) -> PositiveClassConformal:
        p_positive = np.asarray(p_positive, dtype=np.float64)
        labels = np.asarray(labels)
        pos = labels == 1
        scores = 1.0 - p_positive
        self.thresholds[GLOBAL] = conformal_quantile(scores[pos], self.alpha)
        self.n_calibration[GLOBAL] = int(pos.sum())
        if groups is not None:
            groups = np.asarray(groups)
            for g in np.unique(groups[pos]):
                mask = pos & (groups == g)
                if mask.sum() >= min_group_size:
                    self.thresholds[str(g)] = conformal_quantile(scores[mask], self.alpha)
                    self.n_calibration[str(g)] = int(mask.sum())
        return self

    def threshold_for(self, group: str | None) -> float:
        if group is not None and group in self.thresholds:
            return self.thresholds[group]
        return self.thresholds[GLOBAL]

    def flag(self, p_positive: float, group: str | None = None) -> bool:
        if not self.thresholds:
            raise RuntimeError("conformal predictor is not calibrated")
        return (1.0 - p_positive) <= self.threshold_for(group)

    def save(self, path: Path) -> None:
        path.write_text(
            json.dumps(
                {"alpha": self.alpha, "thresholds": self.thresholds, "n": self.n_calibration},
                indent=2,
            )
        )

    @classmethod
    def load(cls, path: Path) -> PositiveClassConformal:
        raw = json.loads(path.read_text())
        return cls(alpha=raw["alpha"], thresholds=raw["thresholds"], n_calibration=raw["n"])


def or_rule(flags: list[bool | None]) -> bool:
    """Flag if any available modality flags. None = modality absent."""
    return any(f for f in flags if f is not None)


def empirical_recall(flags: NDArray[np.bool_], labels: NDArray[np.int64]) -> float:
    pos = np.asarray(labels) == 1
    if not pos.any():
        return float("nan")
    return float(np.asarray(flags)[pos].mean())

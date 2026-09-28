"""Screening-oriented metrics. Every headline number is reported with a bootstrap CI."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
from numpy.typing import NDArray

TIER_ORDER = {"WATCH": 1, "SOON": 2, "URGENT": 3}


def under_triage_rate(pred: Sequence[str], true: Sequence[str]) -> float:
    """Fraction of cases whose predicted tier is *less* severe than the true tier."""
    pairs = [
        (TIER_ORDER[p], TIER_ORDER[t]) for p, t in zip(pred, true, strict=True) if t in TIER_ORDER and p in TIER_ORDER
    ]
    return float(np.mean([p < t for p, t in pairs])) if pairs else float("nan")


def over_triage_rate(pred: Sequence[str], true: Sequence[str]) -> float:
    pairs = [
        (TIER_ORDER[p], TIER_ORDER[t]) for p, t in zip(pred, true, strict=True) if t in TIER_ORDER and p in TIER_ORDER
    ]
    return float(np.mean([p > t for p, t in pairs])) if pairs else float("nan")


def urgent_recall(pred: Sequence[str], true: Sequence[str]) -> float:
    idx = [i for i, t in enumerate(true) if t == "URGENT"]
    return float(np.mean([pred[i] == "URGENT" for i in idx])) if idx else float("nan")


def micro_f1(pred_sets: Sequence[set[str]], true_sets: Sequence[set[str]]) -> float:
    tp = sum(len(p & t) for p, t in zip(pred_sets, true_sets, strict=True))
    fp = sum(len(p - t) for p, t in zip(pred_sets, true_sets, strict=True))
    fn = sum(len(t - p) for p, t in zip(pred_sets, true_sets, strict=True))
    return 2 * tp / max(1, 2 * tp + fp + fn)


def specificity_at_sensitivity(scores: NDArray[np.float64], labels: NDArray[np.int64], target: float) -> float:
    pos, neg = scores[labels == 1], scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    thr = np.quantile(pos, 1 - target)  # lowest threshold keeping >= target of positives
    return float((neg < thr).mean())


def aurc(confidence: NDArray[np.float64], correct: NDArray[np.bool_]) -> float:
    """Area under the risk-coverage curve (lower is better) for selective prediction."""
    order = np.argsort(-confidence)
    errors = 1.0 - correct[order].astype(np.float64)
    risks = np.cumsum(errors) / np.arange(1, len(errors) + 1)
    return float(risks.mean())


def net_benefit(pred_pos: NDArray[np.bool_], labels: NDArray[np.int64], threshold_prob: float) -> float:
    """Decision-curve net benefit at a threshold probability (Vickers & Elkin, 2006)."""
    n = len(labels)
    tp = float((pred_pos & (labels == 1)).sum())
    fp = float((pred_pos & (labels == 0)).sum())
    return tp / n - fp / n * threshold_prob / (1 - threshold_prob)


def bootstrap_ci(
    metric: Callable[[NDArray[np.int64]], float], n: int, n_boot: int = 1000, seed: int = 13, level: float = 0.95
) -> tuple[float, float, float]:
    """metric(idx) -> value, evaluated on resampled indices. Returns (point, lo, hi)."""
    rng = np.random.default_rng(seed)
    point = metric(np.arange(n))
    vals = [metric(rng.integers(0, n, n)) for _ in range(n_boot)]
    vals = [v for v in vals if not np.isnan(v)]
    lo, hi = np.quantile(vals, [(1 - level) / 2, 1 - (1 - level) / 2]) if vals else (np.nan, np.nan)
    return point, float(lo), float(hi)

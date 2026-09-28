"""Temperature scaling and calibration metrics (numpy only, no torch needed)."""

from __future__ import annotations

import itertools

import numpy as np
from numpy.typing import NDArray


def softmax(logits: NDArray[np.float64], temperature: float = 1.0) -> NDArray[np.float64]:
    z = np.asarray(logits, dtype=np.float64) / temperature
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    out: NDArray[np.float64] = e / e.sum(axis=-1, keepdims=True)
    return out


def nll(logits: NDArray[np.float64], labels: NDArray[np.int64], temperature: float) -> float:
    probs = softmax(logits, temperature)
    picked = probs[np.arange(len(labels)), labels]
    return float(-np.log(np.clip(picked, 1e-12, 1.0)).mean())


def fit_temperature(
    logits: NDArray[np.float64],
    labels: NDArray[np.int64],
    lo: float = 0.05,
    hi: float = 20.0,
    iters: int = 100,
) -> float:
    """Golden-section search on log-temperature; NLL is unimodal in T for fixed logits."""
    if len(labels) == 0:
        raise ValueError("need calibration samples")
    ratio = (np.sqrt(5) - 1) / 2
    a, b = np.log(lo), np.log(hi)
    c, d = b - ratio * (b - a), a + ratio * (b - a)
    fc, fd = nll(logits, labels, np.exp(c)), nll(logits, labels, np.exp(d))
    for _ in range(iters):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - ratio * (b - a)
            fc = nll(logits, labels, np.exp(c))
        else:
            a, c, fc = c, d, fd
            d = a + ratio * (b - a)
            fd = nll(logits, labels, np.exp(d))
    return float(np.exp((a + b) / 2))


def expected_calibration_error(probs: NDArray[np.float64], labels: NDArray[np.int64], n_bins: int = 15) -> float:
    """Top-label ECE with equal-width bins."""
    probs = np.asarray(probs, dtype=np.float64)
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == labels).astype(np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in itertools.pairwise(edges):
        mask = (conf > lo) & (conf <= hi)
        if mask.any():
            ece += mask.mean() * abs(correct[mask].mean() - conf[mask].mean())
    return float(ece)


def brier_score(probs: NDArray[np.float64], labels: NDArray[np.int64]) -> float:
    onehot = np.eye(probs.shape[1])[labels]
    return float(((probs - onehot) ** 2).sum(axis=1).mean())

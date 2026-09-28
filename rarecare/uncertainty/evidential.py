"""Subjective-logic opinions from Dirichlet evidence, and their fusion.

Follows Trusted Multi-view Classification (Han et al., ICLR 2021): each modality
yields Dirichlet parameters alpha = evidence + 1; its opinion is belief masses
b_k = e_k / S plus uncertainty u = K / S with S = sum(alpha). Two opinions combine
with the reduced Dempster rule. A missing modality is the *vacuous* opinion
(b = 0, u = 1), which is the identity of the combination rule — so the fused
result degrades gracefully to whichever modalities are present.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class Opinion:
    belief: FloatArray  # shape (K,)
    uncertainty: float
    conflict: float = 0.0  # conflict mass discarded when this opinion was produced

    def __post_init__(self) -> None:
        total = float(self.belief.sum()) + self.uncertainty
        if not np.isclose(total, 1.0, atol=1e-6):
            raise ValueError(f"belief + uncertainty must sum to 1, got {total}")
        if (self.belief < -1e-12).any() or not (0.0 <= self.uncertainty <= 1.0 + 1e-12):
            raise ValueError("belief and uncertainty must be non-negative")

    @property
    def k(self) -> int:
        return int(self.belief.shape[0])

    @classmethod
    def from_evidence(cls, evidence: FloatArray) -> Opinion:
        evidence = np.asarray(evidence, dtype=np.float64)
        if (evidence < 0).any():
            raise ValueError("evidence must be non-negative")
        k = evidence.shape[0]
        strength = evidence.sum() + k
        return cls(belief=evidence / strength, uncertainty=k / strength)

    @classmethod
    def vacuous(cls, k: int) -> Opinion:
        return cls(belief=np.zeros(k), uncertainty=1.0)

    def to_alpha(self) -> FloatArray:
        if self.uncertainty <= 0:
            raise ValueError("dogmatic opinion has no finite Dirichlet")
        strength = self.k / self.uncertainty
        return self.belief * strength + 1.0

    def expected_probability(self) -> FloatArray:
        alpha = self.to_alpha()
        return alpha / alpha.sum()


def combine(a: Opinion, b: Opinion) -> Opinion:
    """Reduced Dempster combination of two opinions over the same K classes."""
    if a.k != b.k:
        raise ValueError("opinions must share the class space")
    outer = np.outer(a.belief, b.belief)
    conflict = float(outer.sum() - np.trace(outer))
    norm = 1.0 - conflict
    if norm <= 1e-12:
        raise ValueError("total conflict: opinions are fully contradictory")
    belief = (a.belief * b.belief + a.belief * b.uncertainty + b.belief * a.uncertainty) / norm
    uncertainty = a.uncertainty * b.uncertainty / norm
    # Renormalise away floating-point drift.
    total = belief.sum() + uncertainty
    return Opinion(belief=belief / total, uncertainty=uncertainty / total, conflict=conflict)


def combine_all(opinions: list[Opinion]) -> Opinion:
    if not opinions:
        raise ValueError("need at least one opinion")
    fused = opinions[0]
    max_conflict = 0.0
    for op in opinions[1:]:
        fused = combine(fused, op)
        max_conflict = max(max_conflict, fused.conflict)
    return Opinion(fused.belief, fused.uncertainty, conflict=max_conflict)

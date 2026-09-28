"""Evidential deep learning loss (Sensoy et al., NeurIPS 2018)."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def kl_to_uniform_dirichlet(alpha: torch.Tensor) -> torch.Tensor:
    """KL(Dir(alpha) || Dir(1)) per sample. Penalises evidence on wrong classes."""
    k = alpha.shape[-1]
    ones = torch.ones_like(alpha)
    s = alpha.sum(-1, keepdim=True)
    term1 = torch.lgamma(s) - torch.lgamma(alpha).sum(-1, keepdim=True) - torch.lgamma(torch.tensor(float(k)))
    term2 = ((alpha - ones) * (torch.digamma(alpha) - torch.digamma(s))).sum(-1, keepdim=True)
    return (term1 + term2).squeeze(-1)


def evidential_loss(evidence: torch.Tensor, target: torch.Tensor, epoch: int, anneal_epochs: int = 10) -> torch.Tensor:
    """Expected-MSE evidential loss with annealed KL regulariser.

    evidence: (N, K) non-negative. target: (N,) int class ids.
    """
    k = evidence.shape[-1]
    y = F.one_hot(target, k).float()
    alpha = evidence + 1.0
    s = alpha.sum(-1, keepdim=True)
    p = alpha / s
    err = ((y - p) ** 2).sum(-1)
    var = (p * (1 - p) / (s + 1)).sum(-1)
    alpha_tilde = y + (1 - y) * alpha  # remove evidence of the true class before the KL
    anneal = min(1.0, epoch / max(1, anneal_epochs))
    return (err + var + anneal * kl_to_uniform_dirichlet(alpha_tilde)).mean()

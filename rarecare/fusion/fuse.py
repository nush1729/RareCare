"""Text + image fusion inside a safety envelope.

Two layers with different jobs:

1. **Safety layer (guarantee):** each calibrated modality produces a conformal flag
   for "suspicious"; the OR-rule escalates the cancer to URGENT if any flags.
   Images can only escalate, never downgrade, a text-derived tier.
2. **Evidential layer (ranking + conflict):** subjective-logic fusion of per-modality
   opinions over {not suspicious, suspicious}. Its expected probability ranks cases;
   its conflict mass tells the user and clinician when text and image disagree.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rarecare.schemas import Tier
from rarecare.uncertainty.conformal import or_rule
from rarecare.uncertainty.evidential import Opinion, combine_all

# Evidence weight given to the rule baseline's text distribution. A trained
# evidential criterion head replaces this heuristic with learned evidence.
RULE_TEXT_EVIDENCE = 10.0
CONFLICT_ALERT = 0.3


@dataclass(frozen=True)
class ModalityInput:
    opinion: Opinion
    conformal_flag: bool | None  # None when the modality is not calibrated


@dataclass(frozen=True)
class FusionResult:
    tier: Tier
    fused: Opinion
    p_suspicious: float
    conflict: float
    conflict_alert: bool
    escalated_by_image: bool


def text_opinion_from_distribution(p_urgent: float, strength: float = RULE_TEXT_EVIDENCE) -> Opinion:
    p = float(np.clip(p_urgent, 0.0, 1.0))
    return Opinion.from_evidence(np.array([strength * (1 - p), strength * p]))


def fuse_cancer(text_tier: Tier, text: ModalityInput | None, image: ModalityInput | None) -> FusionResult:
    opinions = [m.opinion for m in (text, image) if m is not None] or [Opinion.vacuous(2)]
    fused = combine_all(opinions)
    p_susp = float(fused.expected_probability()[1])

    image_flag = image.conformal_flag if image is not None else None
    text_flag = text.conformal_flag if text is not None else None
    flagged = or_rule([text_flag, image_flag])

    tier = text_tier
    escalated = False
    if flagged and tier.severity < Tier.URGENT.severity:
        tier = Tier.URGENT
        escalated = image_flag is True
    return FusionResult(
        tier=tier,
        fused=fused,
        p_suspicious=p_susp,
        conflict=fused.conflict,
        conflict_alert=fused.conflict >= CONFLICT_ALERT,
        escalated_by_image=escalated,
    )

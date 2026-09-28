"""Typed contracts shared by the pipeline, the API and the tests."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class Tier(StrEnum):
    URGENT = "URGENT"
    SOON = "SOON"
    WATCH = "WATCH"
    NEED_MORE_INFO = "NEED_MORE_INFO"

    @property
    def severity(self) -> int:
        return {"URGENT": 3, "SOON": 2, "WATCH": 1, "NEED_MORE_INFO": 0}[self.value]


class ImageStatus(StrEnum):
    ASSESSED = "ASSESSED"
    RETAKE = "RETAKE"
    UNSUPPORTED_MEDICAL = "UNSUPPORTED_MEDICAL"
    NOT_MEDICAL = "NOT_MEDICAL"
    MODEL_UNAVAILABLE = "MODEL_UNAVAILABLE"


class ComponentMode(StrEnum):
    """How a component produced its output — reported to the user for honesty."""

    NEURAL = "neural"
    RULE_BASELINE = "rule-baseline"
    UNAVAILABLE = "unavailable"


class Finding(BaseModel):
    """A symptom mention extracted from text."""

    model_config = ConfigDict(frozen=True)

    concept_id: str
    label: str
    span: str
    start: int
    end: int
    negated: bool = False
    language_register: str = "unknown"


class EvidenceLink(BaseModel):
    """One traceable chain: phrase -> concept -> criterion -> tier."""

    span: str
    concept_id: str
    concept_label: str
    criterion_id: str
    tier: Tier
    action: str
    source: str
    # < 1 when the criterion depends on something not yet known (e.g. age group).
    probability: float = 1.0
    from_questionnaire: bool = False


class QuestionOption(BaseModel):
    value: str
    label: str


class Question(BaseModel):
    variable: str
    text: str
    options: list[QuestionOption]
    disclosure_cost: int
    expected_information_gain: float


class CancerAssessment(BaseModel):
    cancer: str
    label: str
    tier: Tier
    tier_distribution: dict[str, float]
    matched_criteria: list[str]


class ImageAssessment(BaseModel):
    status: ImageStatus
    message: str
    modality: str | None = None
    quality: dict[str, float] = Field(default_factory=dict)
    malignancy_probability: float | None = None
    uncertainty: float | None = None
    conformal_set: list[str] | None = None
    heatmap_png_b64: str | None = None


class Citation(BaseModel):
    doc_id: str
    title: str
    url: str
    snippet: str
    score: float


class AssessmentResult(BaseModel):
    tier: Tier
    headline: str
    cancers: list[CancerAssessment]
    evidence: list[EvidenceLink]
    findings: list[Finding]
    question: Question | None = None
    image: ImageAssessment | None = None
    modality_conflict: float | None = None
    citations: list[Citation] = Field(default_factory=list)
    safety_net: list[str] = Field(default_factory=list)
    component_modes: dict[str, ComponentMode] = Field(default_factory=dict)
    conformal_calibrated: bool = False
    disclaimer: str = (
        "RareCare is a research prototype, not a diagnosis. If you are worried, please see a doctor or health worker."
    )
    debug: dict[str, Any] | None = None

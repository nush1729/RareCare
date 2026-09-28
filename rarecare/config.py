"""Runtime configuration loaded from YAML with environment overrides."""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parent
DEFAULT_CONFIG = REPO_ROOT / "configs" / "default.yaml"


@dataclass(frozen=True)
class TriageConfig:
    # Stop asking once the most likely tier has at least this probability.
    confidence_to_decide: float = 0.9
    # Screening-conservative: if we must decide under uncertainty, escalate to the
    # most severe tier whose probability is at least this value.
    escalation_floor: float = 0.1
    max_questions: int = 4
    # Planner score = EIG / cost**cost_exponent. 0 ignores disclosure cost.
    cost_exponent: float = 1.0
    persistence_days: int = 21


@dataclass(frozen=True)
class ModelConfig:
    text_extractor_checkpoint: str | None = None
    criteria_classifier_checkpoint: str | None = None
    concept_normalizer_checkpoint: str | None = None
    vision_checkpoint: str | None = None
    conformal_thresholds: str | None = None
    retriever: str = "bm25"  # "bm25" | "medcpt"
    device: str = "cpu"


@dataclass(frozen=True)
class ConformalConfig:
    alpha: float = 0.05


@dataclass(frozen=True)
class AppConfig:
    max_upload_mb: int = 8
    session_ttl_seconds: int = 900
    max_sessions: int = 10_000
    rate_limit_per_minute: int = 30
    max_text_chars: int = 2_000


@dataclass(frozen=True)
class Config:
    triage: TriageConfig = field(default_factory=TriageConfig)
    models: ModelConfig = field(default_factory=ModelConfig)
    conformal: ConformalConfig = field(default_factory=ConformalConfig)
    app: AppConfig = field(default_factory=AppConfig)


def _build(cls: type, raw: dict[str, Any] | None) -> Any:
    raw = raw or {}
    known = {f.name for f in fields(cls)}
    unknown = set(raw) - known
    if unknown:
        raise ValueError(f"Unknown config keys for {cls.__name__}: {sorted(unknown)}")
    return cls(**raw)


def load_config(path: str | os.PathLike[str] | None = None) -> Config:
    """Load config from YAML. `RARECARE_CONFIG` overrides the default path."""
    cfg_path = Path(path) if path else Path(os.environ.get("RARECARE_CONFIG", str(DEFAULT_CONFIG)))
    raw: dict[str, Any] = {}
    if cfg_path.exists():
        raw = yaml.safe_load(cfg_path.read_text()) or {}
    return Config(
        triage=_build(TriageConfig, raw.get("triage")),
        models=_build(ModelConfig, raw.get("models")),
        conformal=_build(ConformalConfig, raw.get("conformal")),
        app=_build(AppConfig, raw.get("app")),
    )

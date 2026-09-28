"""End-to-end orchestration: text + optional image -> tier, question or explanation.

Every neural component has a working fallback, and the response reports which
mode each component ran in, so an untrained build is honest about what it is.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

from rarecare.agent.reasoner import (
    QuestionPlanner,
    TierDistribution,
    TierReasoner,
    TriageState,
    decode_answer,
)
from rarecare.config import Config, load_config
from rarecare.explain.explanation import HEADLINES, evidence_chain, retrieval_query
from rarecare.fusion.fuse import ModalityInput, fuse_cancer, text_opinion_from_distribution
from rarecare.kg.graph import KnowledgeGraph, default_graph
from rarecare.rag.retriever import Retriever, build_retriever
from rarecare.schemas import (
    AssessmentResult,
    CancerAssessment,
    ComponentMode,
    Finding,
    Tier,
)
from rarecare.text.extractor import Extractor, LexiconExtractor
from rarecare.text.normalize import (
    age_to_band,
    mentions_frequency,
    normalize,
    parse_age,
    parse_duration_days,
)
from rarecare.text.pii import scrub
from rarecare.uncertainty.conformal import PositiveClassConformal
from rarecare.vision.assessor import ImageAssessor, ImageOutcome

log = logging.getLogger(__name__)
_CLAUSE = re.compile(r"[.;,!?]|\band\b|\bbut\b", re.IGNORECASE)


@dataclass
class Session:
    state: TriageState
    findings: list[Finding]
    image: ImageOutcome | None = None
    history: list[str] = field(default_factory=list)


class RareCarePipeline:
    def __init__(self, config: Config | None = None, kg: KnowledgeGraph | None = None):
        self.cfg = config or load_config()
        self.kg = kg or default_graph()
        self.reasoner = TierReasoner(self.kg)
        self.planner = QuestionPlanner(self.reasoner, self.cfg.triage)
        self.modes: dict[str, ComponentMode] = {}

        m = self.cfg.models
        # Hybrid extraction: the lexicon is always on (high precision); a trained BioBERT
        # extractor only *adds* concepts the lexicon missed. Replacing the lexicon with
        # the neural model lowered F1 on every register in evaluation.
        self.extractor: Extractor = LexiconExtractor(self.kg)
        self.neural_extractor: Extractor | None = None
        self.modes["symptom_extractor"] = ComponentMode.RULE_BASELINE
        if m.text_extractor_checkpoint:
            from rarecare.text.extractor import BioBERTExtractor

            self.neural_extractor = BioBERTExtractor(m.text_extractor_checkpoint, self.kg, m.device)
            self.modes["symptom_extractor"] = ComponentMode.NEURAL

        self.criteria_clf = None
        self.modes["criteria_classifier"] = ComponentMode.UNAVAILABLE
        if m.criteria_classifier_checkpoint:
            from rarecare.text.neural import CriteriaClassifier

            self.criteria_clf = CriteriaClassifier(m.criteria_classifier_checkpoint, m.device)
            self.modes["criteria_classifier"] = ComponentMode.NEURAL

        self.normalizer = None
        self.modes["concept_normalizer"] = ComponentMode.UNAVAILABLE
        if m.concept_normalizer_checkpoint:
            from rarecare.text.neural import SapBERTNormalizer

            self.normalizer = SapBERTNormalizer(m.concept_normalizer_checkpoint, self.kg, m.device)
            self.modes["concept_normalizer"] = ComponentMode.NEURAL

        self.conformal: dict[str, PositiveClassConformal] = {}
        if m.conformal_thresholds and Path(m.conformal_thresholds).exists():
            self.conformal = _load_conformal(Path(m.conformal_thresholds))
        self.image = ImageAssessor(m.vision_checkpoint, self.conformal, m.device)
        self.modes["image_models"] = ComponentMode.NEURAL if self.image.available else ComponentMode.UNAVAILABLE

        self.retriever: Retriever = build_retriever(m.retriever, m.device)
        self.modes["retriever"] = ComponentMode.NEURAL if m.retriever == "medcpt" else ComponentMode.RULE_BASELINE
        self.modes["reasoner"] = ComponentMode.RULE_BASELINE

    # ------------------------------------------------------------------ text

    def _clean(self, text: str) -> str:
        normalized = normalize(text)
        protected = [(f.start, f.end) for f in LexiconExtractor(self.kg).extract(normalized)]
        return normalize(scrub(normalized, use_presidio=self.cfg.models.presidio, protected=protected))

    def extract(self, text: str) -> list[Finding]:
        clean = self._clean(text)
        findings = self.extractor.extract(clean)
        covered = {f.concept_id for f in findings}
        if self.neural_extractor is not None:
            for f in self.neural_extractor.extract(clean):
                if f.concept_id not in covered:
                    findings.append(f)
                    covered.add(f.concept_id)
        if self.normalizer is not None:
            for clause in filter(None, (c.strip() for c in _CLAUSE.split(clean))):
                if any(f.span.lower() in clause.lower() for f in findings):
                    continue
                hit = self.normalizer.link(clause)
                if hit and hit[0] not in covered:
                    start = clean.find(clause)
                    concept = self.kg.concepts[hit[0]]
                    findings.append(
                        Finding(
                            concept_id=concept.id,
                            label=concept.label,
                            span=clause,
                            start=start,
                            end=start + len(clause),
                            language_register="model",
                        )
                    )
                    covered.add(concept.id)
        if self.criteria_clf is not None:
            for cid in self.criteria_clf.present(clean) - covered:
                if cid in self.kg.concepts:
                    findings.append(
                        Finding(
                            concept_id=cid,
                            label=self.kg.concepts[cid].label,
                            span="(whole message)",
                            start=0,
                            end=0,
                            language_register="model",
                        )
                    )
        return findings

    def _initial_state(self, text: str, findings: list[Finding], age_band: str | None) -> TriageState:
        clean = self._clean(text)
        state = TriageState(
            present={f.concept_id for f in findings if not f.negated},
            negated={f.concept_id for f in findings if f.negated},
        )
        if age_band is None:
            age = parse_age(clean)
            age_band = age_to_band(age) if age is not None else None
        if age_band is not None:
            if age_band not in self.kg.variables["age_band"].values:
                raise ValueError(f"invalid age band {age_band!r}")
            state.variables["age_band"] = age_band
        days = parse_duration_days(clean)
        if mentions_frequency(clean) or (days is not None and days >= self.cfg.triage.persistence_days):
            state.variables["persistent"] = True
        elif days is not None:
            state.variables["persistent"] = False
        return state

    # --------------------------------------------------------------- public

    def start(
        self,
        text: str,
        age_band: str | None = None,
        image_bytes: bytes | None = None,
        selected_concepts: list[str] | None = None,
        duration: str | None = None,
    ) -> tuple[Session, AssessmentResult]:
        """Start an assessment from free text and/or questionnaire answers.

        `selected_concepts` are concept ids ticked in the questionnaire; `duration`
        is one of lt_3w | gte_3w | unsure. Explicit answers override text parsing.
        """
        findings = self.extract(text) if text.strip() else []
        known = {f.concept_id for f in findings}
        for cid in selected_concepts or []:
            if cid not in self.kg.concepts:
                raise ValueError(f"unknown symptom {cid!r}")
            if cid in known:
                continue
            findings.append(
                Finding(
                    concept_id=cid,
                    label=self.kg.concepts[cid].label,
                    span=self.kg.concepts[cid].label,
                    start=0,
                    end=0,
                    language_register="questionnaire",
                )
            )
            known.add(cid)
        state = self._initial_state(text, findings, age_band)
        if duration == "gte_3w":
            state.variables["persistent"] = True
        elif duration == "lt_3w":
            state.variables["persistent"] = False
        elif duration not in (None, "", "unsure"):
            raise ValueError(f"invalid duration {duration!r}")
        session = Session(state=state, findings=findings)
        if image_bytes:
            session.image = self.image.assess(image_bytes)
        return session, self.evaluate(session)

    def answer(self, session: Session, variable: str, raw_value: str) -> AssessmentResult:
        if variable not in self.kg.variables:
            raise ValueError(f"unknown question {variable!r}")
        if variable in session.state.variables or variable in session.state.declined:
            raise ValueError(f"question {variable!r} already answered")
        value = decode_answer(self.kg, variable, raw_value)
        session.state.questions_asked += 1
        if value is None:
            session.state.declined.add(variable)
        else:
            session.state.variables[variable] = value
        session.history.append(variable)
        return self.evaluate(session)

    def evaluate(self, session: Session) -> AssessmentResult:
        decision = self.planner.decide(session.state)
        dist = decision.distribution

        cancers: list[CancerAssessment] = []
        tier = decision.tier
        conflict = None
        img = session.image
        cancer_names = set(dist.per_cancer) | ({img.cancer} if img and img.cancer else set())
        for cancer in sorted(cancer_names):
            cdist = dist.per_cancer.get(cancer, {Tier.WATCH: 1.0})
            ctier = self.planner.conservative_tier(_as_dist(cdist))
            if img is not None and img.cancer == cancer and img.opinion is not None:
                fused = fuse_cancer(
                    ctier,
                    ModalityInput(text_opinion_from_distribution(cdist.get(Tier.URGENT, 0.0)), None),
                    ModalityInput(img.opinion, img.conformal_flag),
                )
                ctier, conflict = fused.tier, round(fused.conflict, 4)
                if tier != Tier.NEED_MORE_INFO and ctier.severity > tier.severity:
                    tier = ctier
            cancers.append(
                CancerAssessment(
                    cancer=cancer,
                    label=self.kg.cancers[cancer],
                    tier=ctier,
                    tier_distribution={t.value: round(p, 4) for t, p in cdist.items()},
                    matched_criteria=sorted(
                        c.id for c in self.kg.criteria if c.cancer == cancer and dist.criteria_prob.get(c.id, 0.0) > 0.0
                    ),
                )
            )

        resolved: dict[str, str] = {}
        ctx = session.state.variables.get("bleeding_context")
        resolves = self.kg.variables["bleeding_context"].resolves or {}
        if ctx is not None and resolves.get(str(ctx)):
            resolved["vaginal_bleeding_unspecified"] = str(resolves[str(ctx)])
        evidence = evidence_chain(self.kg, session.findings, dist.criteria_prob, resolved)

        involved = {c.cancer for c in cancers}
        query = retrieval_query(session.findings)
        citations = self.retriever.search(query, involved, k=3) if query else []
        safety = [self.kg.safety_net[c] for c in sorted(involved) if c in self.kg.safety_net]
        if not safety:
            safety = list(self.kg.safety_net.values())

        return AssessmentResult(
            tier=tier,
            headline=HEADLINES[tier],
            cancers=cancers,
            evidence=evidence,
            findings=session.findings,
            question=decision.question,
            image=img.assessment if img else None,
            modality_conflict=conflict,
            citations=citations,
            safety_net=safety,
            component_modes=dict(self.modes),
            conformal_calibrated=bool(self.conformal),
        )


def _as_dist(d: dict[Tier, float]) -> TierDistribution:
    return TierDistribution(overall=d, per_cancer={}, criteria_prob={})


def _load_conformal(path: Path) -> dict[str, PositiveClassConformal]:
    import json

    raw = json.loads(path.read_text())
    return {
        cancer: PositiveClassConformal(alpha=v["alpha"], thresholds=v["thresholds"], n_calibration=v["n"])
        for cancer, v in raw.items()
    }

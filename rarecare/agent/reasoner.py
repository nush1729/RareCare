"""Tier reasoning under unknown variables, and disclosure-cost-aware questioning.

The reasoner enumerates every assignment of the unknown variables that could
change a decision, weighting each by its prior, to get a distribution over
referral tiers. The planner asks the question with the best
expected-information-gain / disclosure-cost ratio. The LLM never makes decisions;
it can only rephrase the chosen question (see `phrasing.py`).
"""

from __future__ import annotations

import itertools
import math
from collections import defaultdict
from dataclasses import dataclass, field

from rarecare.config import TriageConfig
from rarecare.kg.graph import KnowledgeGraph, VariableValue
from rarecare.schemas import Question, QuestionOption, Tier

DECIDED_TIERS = (Tier.URGENT, Tier.SOON, Tier.WATCH)
MAX_ENUMERATION = 4096  # guards against combinatorial blow-up if the KG grows


@dataclass
class TriageState:
    present: set[str] = field(default_factory=set)
    negated: set[str] = field(default_factory=set)
    variables: dict[str, VariableValue] = field(default_factory=dict)
    declined: set[str] = field(default_factory=set)
    questions_asked: int = 0


@dataclass(frozen=True)
class TierDistribution:
    overall: dict[Tier, float]
    per_cancer: dict[str, dict[Tier, float]]
    criteria_prob: dict[str, float]

    def entropy(self) -> float:
        return _entropy(self.overall.values())

    def top(self) -> tuple[Tier, float]:
        tier = max(self.overall, key=lambda t: (self.overall[t], t.severity))
        return tier, self.overall[tier]


def _entropy(probs: object) -> float:
    return -sum(p * math.log2(p) for p in probs if p > 0)  # type: ignore[attr-defined]


class TierReasoner:
    def __init__(self, kg: KnowledgeGraph):
        self.kg = kg

    def unknown_variables(self, state: TriageState) -> list[str]:
        relevant = self.kg.relevant_variables(state.present)
        return sorted(v for v in relevant if v not in state.variables)

    def evaluate(
        self, present: set[str], variables: dict[str, VariableValue]
    ) -> tuple[Tier, dict[str, Tier], list[str]]:
        """Deterministic tiers for a fully specified assignment."""
        concepts = set(present)
        for var_name, value in variables.items():
            var = self.kg.variables[var_name]
            if var.resolves is None:
                continue
            ambiguous = {c.id for c in self.kg.concepts.values() if c.ambiguous_variable == var_name}
            if concepts & ambiguous:
                resolved = var.resolves.get(str(value))
                concepts -= ambiguous
                if resolved:
                    concepts.add(resolved)

        per_cancer: dict[str, Tier] = {}
        for cid in concepts:
            for cancer in self.kg.concepts[cid].cancers:
                per_cancer.setdefault(cancer, Tier.WATCH)
        matched: list[str] = []
        for crit in self.kg.criteria:
            if crit.satisfied(concepts, variables):
                matched.append(crit.id)
                if crit.tier.severity > per_cancer.get(crit.cancer, Tier.WATCH).severity:
                    per_cancer[crit.cancer] = crit.tier
        overall = max(per_cancer.values(), key=lambda t: t.severity, default=Tier.WATCH)
        return overall, per_cancer, matched

    def distribution(self, state: TriageState) -> TierDistribution:
        unknown = self.unknown_variables(state)
        spaces = [list(zip(self.kg.variables[v].values, self.kg.variables[v].prior, strict=True)) for v in unknown]
        n_assignments = math.prod(len(s) for s in spaces) if spaces else 1
        if n_assignments > MAX_ENUMERATION:
            raise RuntimeError(f"Too many unknown combinations ({n_assignments})")

        overall: dict[Tier, float] = defaultdict(float)
        per_cancer: dict[str, dict[Tier, float]] = defaultdict(lambda: defaultdict(float))
        crit_prob: dict[str, float] = defaultdict(float)
        for combo in itertools.product(*spaces) if spaces else [()]:
            weight = math.prod(p for _, p in combo) if combo else 1.0
            assignment = dict(state.variables)
            assignment.update({v: val for v, (val, _) in zip(unknown, combo, strict=True)})
            tier, cancers, matched = self.evaluate(state.present, assignment)
            overall[tier] += weight
            for cancer, ctier in cancers.items():
                per_cancer[cancer][ctier] += weight
            for cid in matched:
                crit_prob[cid] += weight
        return TierDistribution(
            overall=dict(overall),
            per_cancer={k: dict(v) for k, v in per_cancer.items()},
            criteria_prob=dict(crit_prob),
        )


@dataclass(frozen=True)
class PlannerDecision:
    tier: Tier
    question: Question | None
    distribution: TierDistribution


class QuestionPlanner:
    def __init__(self, reasoner: TierReasoner, cfg: TriageConfig):
        self.reasoner = reasoner
        self.cfg = cfg

    def expected_information_gain(self, state: TriageState, variable: str) -> float:
        prior_h = self.reasoner.distribution(state).entropy()
        var = self.reasoner.kg.variables[variable]
        expected_h = 0.0
        for value, p in zip(var.values, var.prior, strict=True):
            hypothetical = TriageState(
                present=state.present,
                negated=state.negated,
                variables={**state.variables, variable: value},
                declined=state.declined,
            )
            expected_h += p * self.reasoner.distribution(hypothetical).entropy()
        return max(0.0, prior_h - expected_h)

    def decide(self, state: TriageState) -> PlannerDecision:
        dist = self.reasoner.distribution(state)
        top_tier, top_p = dist.top()
        if top_p >= self.cfg.confidence_to_decide:
            return PlannerDecision(top_tier, None, dist)

        if state.questions_asked < self.cfg.max_questions:
            best: tuple[float, str, float] | None = None
            for var in self.reasoner.unknown_variables(state):
                if var in state.declined:
                    continue
                eig = self.expected_information_gain(state, var)
                if eig <= 1e-9:
                    continue
                cost = self.reasoner.kg.variables[var].cost
                score = eig / (cost**self.cfg.cost_exponent)
                if best is None or score > best[0]:
                    best = (score, var, eig)
            if best is not None:
                _, var, eig = best
                return PlannerDecision(Tier.NEED_MORE_INFO, self._question(var, eig), dist)

        return PlannerDecision(self.conservative_tier(dist), None, dist)

    def conservative_tier(self, dist: TierDistribution) -> Tier:
        """Most severe tier with at least `escalation_floor` probability.

        Under-triage is the costly error in screening, so unresolved uncertainty
        escalates rather than averaging away.
        """
        for tier in DECIDED_TIERS:
            if dist.overall.get(tier, 0.0) >= self.cfg.escalation_floor:
                return tier
        return Tier.WATCH

    def _question(self, variable: str, eig: float) -> Question:
        var = self.reasoner.kg.variables[variable]
        options = [QuestionOption(value=_encode(v), label=_option_label(variable, v)) for v in var.values]
        options.append(QuestionOption(value="__decline__", label="Prefer not to say"))
        return Question(
            variable=variable,
            text=var.question,
            options=options,
            disclosure_cost=var.cost,
            expected_information_gain=round(eig, 4),
        )


_LABELS: dict[str, dict[str, str]] = {
    "persistent": {"true": "Yes", "false": "No"},
    "unilateral": {"true": "Unilateral (one side)", "false": "Bilateral (both sides)"},
    "bleeding_context": {
        "after_sex": "Post-coital (after intercourse)",
        "between_periods": "Intermenstrual (between periods)",
        "after_menopause": "Post-menopausal",
        "during_period": "During menstruation only",
    },
}


def _encode(value: VariableValue) -> str:
    return str(value).lower() if isinstance(value, bool) else str(value)


def _option_label(variable: str, value: VariableValue) -> str:
    return _LABELS.get(variable, {}).get(_encode(value), str(value))


def decode_answer(kg: KnowledgeGraph, variable: str, raw: str) -> VariableValue | None:
    """Map an option value back to the typed variable value. None = declined."""
    if raw == "__decline__":
        return None
    for value in kg.variables[variable].values:
        if _encode(value) == raw:
            return value
    raise ValueError(f"Invalid answer {raw!r} for {variable}")

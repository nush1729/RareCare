"""Guideline knowledge graph: surface form -> concept -> criterion -> cancer.

One structure serves three jobs: mapping extracted concepts to referral criteria,
enumerating which unknown variables could change the decision (for the question
planner), and producing traceable evidence chains for explanations.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import networkx as nx

from rarecare.schemas import Tier

DATA_DIR = Path(__file__).resolve().parent / "data"

VariableValue = str | bool


@dataclass(frozen=True)
class Pattern:
    regex: re.Pattern[str]
    register: str


@dataclass(frozen=True)
class Concept:
    id: str
    label: str
    cancers: tuple[str, ...]
    patterns: tuple[Pattern, ...]
    ambiguous_variable: str | None = None


@dataclass(frozen=True)
class Criterion:
    id: str
    cancer: str
    tier: Tier
    any_of: frozenset[str]
    conditions: dict[str, frozenset[VariableValue]]
    action: str
    source: str

    def triggered_by(self, present: set[str]) -> set[str]:
        return set(self.any_of & present)

    def satisfied(self, present: set[str], variables: dict[str, VariableValue]) -> bool:
        """True iff a trigger concept is present and every condition is met.

        A condition on an unknown variable is *not* met; callers that need to reason
        about unknowns enumerate them explicitly (see `TierReasoner`).
        """
        if not self.triggered_by(present):
            return False
        for var, allowed in self.conditions.items():
            if var not in variables or variables[var] not in allowed:
                return False
        return True


@dataclass(frozen=True)
class Variable:
    name: str
    values: tuple[VariableValue, ...]
    prior: tuple[float, ...]
    cost: int
    question: str
    resolves: dict[str, str | None] | None = None

    def __post_init__(self) -> None:
        if len(self.values) != len(self.prior):
            raise ValueError(f"Variable {self.name}: values/prior length mismatch")
        if abs(sum(self.prior) - 1.0) > 1e-6:
            raise ValueError(f"Variable {self.name}: prior must sum to 1")
        if self.cost < 1:
            raise ValueError(f"Variable {self.name}: cost must be >= 1")


class KnowledgeGraph:
    def __init__(self, criteria_path: Path | None = None, lexicon_path: Path | None = None):
        criteria_raw = _read_json(criteria_path or DATA_DIR / "criteria.json")
        lexicon_raw = _read_json(lexicon_path or DATA_DIR / "lexicon.json")

        self.meta: dict[str, Any] = criteria_raw["_meta"]
        self.cancers: dict[str, str] = {k: v["label"] for k, v in criteria_raw["cancers"].items()}
        self.safety_net: dict[str, str] = criteria_raw["safety_net"]
        self.concepts: dict[str, Concept] = {
            cid: Concept(
                id=cid,
                label=c["label"],
                cancers=tuple(c["cancers"]),
                patterns=tuple(Pattern(re.compile(p["p"], re.IGNORECASE), p["register"]) for p in c["patterns"]),
                ambiguous_variable=c.get("ambiguous"),
            )
            for cid, c in lexicon_raw["concepts"].items()
        }
        self.variables: dict[str, Variable] = {
            name: Variable(
                name=name,
                values=tuple(v["values"]),
                prior=tuple(v["prior"]),
                cost=int(v["cost"]),
                question=v["question"],
                resolves=v.get("resolves"),
            )
            for name, v in criteria_raw["variables"].items()
        }
        self.criteria: list[Criterion] = [
            Criterion(
                id=c["id"],
                cancer=c["cancer"],
                tier=Tier(c["tier"]),
                any_of=frozenset(c["any_of"]),
                conditions={k: frozenset(v) for k, v in c["conditions"].items()},
                action=c["action"],
                source=c["source"],
            )
            for c in criteria_raw["criteria"]
        ]
        self._validate()
        self.graph = self._build_graph()

    def _validate(self) -> None:
        for crit in self.criteria:
            if crit.cancer not in self.cancers:
                raise ValueError(f"{crit.id}: unknown cancer {crit.cancer}")
            missing = crit.any_of - self.concepts.keys()
            if missing:
                raise ValueError(f"{crit.id}: unknown concepts {sorted(missing)}")
            for var, allowed in crit.conditions.items():
                if var not in self.variables:
                    raise ValueError(f"{crit.id}: unknown variable {var}")
                bad = allowed - set(self.variables[var].values)
                if bad:
                    raise ValueError(f"{crit.id}: invalid values {bad} for {var}")
        for concept in self.concepts.values():
            if concept.ambiguous_variable and concept.ambiguous_variable not in self.variables:
                raise ValueError(f"{concept.id}: unknown ambiguous variable")

    def _build_graph(self) -> nx.DiGraph:
        g = nx.DiGraph()
        for cancer, label in self.cancers.items():
            g.add_node(("cancer", cancer), label=label)
        for concept in self.concepts.values():
            g.add_node(("concept", concept.id), label=concept.label)
            for cancer in concept.cancers:
                g.add_edge(("concept", concept.id), ("cancer", cancer), rel="associated_with")
        for crit in self.criteria:
            g.add_node(("criterion", crit.id), tier=crit.tier.value, action=crit.action)
            g.add_edge(("criterion", crit.id), ("cancer", crit.cancer), rel="refers_for")
            for cid in crit.any_of:
                g.add_edge(("concept", cid), ("criterion", crit.id), rel="triggers")
            for var in crit.conditions:
                g.add_node(("variable", var))
                g.add_edge(("variable", var), ("criterion", crit.id), rel="conditions")
        return g

    def criteria_for_concept(self, concept_id: str) -> list[Criterion]:
        node = ("concept", concept_id)
        if node not in self.graph:
            return []
        ids = {n[1] for n in self.graph.successors(node) if n[0] == "criterion"}
        return [c for c in self.criteria if c.id in ids]

    def relevant_variables(self, present: set[str]) -> set[str]:
        """Variables that condition any criterion triggered by a present concept."""
        out: set[str] = set()
        for cid in present:
            for crit in self.criteria_for_concept(cid):
                out.update(crit.conditions)
            ambiguous = self.concepts[cid].ambiguous_variable if cid in self.concepts else None
            if ambiguous:
                out.add(ambiguous)
                # Concepts the ambiguity can resolve to bring their own conditions.
                for resolved in (self.variables[ambiguous].resolves or {}).values():
                    if resolved:
                        for crit in self.criteria_for_concept(resolved):
                            out.update(crit.conditions)
        return out


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as fh:
        data: dict[str, Any] = json.load(fh)
    return data


@lru_cache(maxsize=1)
def default_graph() -> KnowledgeGraph:
    return KnowledgeGraph()

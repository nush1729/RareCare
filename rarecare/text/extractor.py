"""Symptom extraction.

Two interchangeable implementations behind one interface:

* `LexiconExtractor` — deterministic, lexicon + NegEx-style negation. Always
  available; it is the rule baseline the neural model must beat.
* `BioBERTExtractor` — BioBERT token classifier (BIO tags per concept) with an
  assertion head, loaded from a fine-tuned checkpoint (see training/).
"""

from __future__ import annotations

import re
from typing import Any, Protocol

from rarecare.kg.graph import KnowledgeGraph
from rarecare.schemas import Finding

_NEGATION_CUES = re.compile(
    r"\b(no|not|never|without|denies|deny|don't|dont|do not|doesn't|haven't|hasn't|"
    r"didn't|nahi|nahin|na)\b",
    re.IGNORECASE,
)
# Clause boundaries stop negation scope ("no pain but a lump" -> lump not negated).
_SCOPE_BREAK = re.compile(r"[.;!?]|\b(but|however|although|though|lekin|par)\b", re.IGNORECASE)
_NEGATION_WINDOW_TOKENS = 5


class Extractor(Protocol):
    def extract(self, text: str) -> list[Finding]: ...


def is_negated(text: str, start: int) -> bool:
    prefix = text[:start]
    breaks = list(_SCOPE_BREAK.finditer(prefix))
    if breaks:
        prefix = prefix[breaks[-1].end() :]
    window = " ".join(prefix.split()[-_NEGATION_WINDOW_TOKENS:])
    return bool(_NEGATION_CUES.search(window))


class LexiconExtractor:
    def __init__(self, kg: KnowledgeGraph):
        self.kg = kg

    def extract(self, text: str) -> list[Finding]:
        candidates: list[Finding] = []
        for concept in self.kg.concepts.values():
            for pattern in concept.patterns:
                for m in pattern.regex.finditer(text):
                    candidates.append(
                        Finding(
                            concept_id=concept.id,
                            label=concept.label,
                            span=m.group(0),
                            start=m.start(),
                            end=m.end(),
                            negated=is_negated(text, m.start()),
                            language_register=pattern.register,
                        )
                    )
        return _resolve(candidates, self.kg)


def _resolve(candidates: list[Finding], kg: KnowledgeGraph) -> list[Finding]:
    """Deduplicate per concept and drop ambiguous concepts that a specific one covers."""
    by_concept: dict[str, Finding] = {}
    for f in sorted(candidates, key=lambda f: (f.start, -(f.end - f.start))):
        prev = by_concept.get(f.concept_id)
        # Keep the first mention, but an affirmed mention beats a negated one.
        if prev is None or (prev.negated and not f.negated):
            by_concept[f.concept_id] = f

    findings = list(by_concept.values())
    for f in list(findings):
        concept = kg.concepts[f.concept_id]
        if not concept.ambiguous_variable:
            continue
        variable = kg.variables[concept.ambiguous_variable]
        specific = {c for c in (variable.resolves or {}).values() if c}
        if any(g.concept_id in specific and _overlaps(f, g) for g in findings):
            findings.remove(f)
    return sorted(findings, key=lambda f: f.start)


def _overlaps(a: Finding, b: Finding) -> bool:
    return a.start < b.end and b.start < a.end


class BioBERTExtractor:
    """Token-classification extractor. Label scheme: B-/I-<concept_id>, plus O.

    Negation comes from the same NegEx scope rule until the assertion head is
    trained; this keeps the two extractors comparable in ablations.
    """

    def __init__(self, checkpoint: str, kg: KnowledgeGraph, device: str = "cpu"):
        from transformers import pipeline  # heavy import kept lazy

        self.kg = kg
        self._ner: Any = pipeline(
            "token-classification",
            model=checkpoint,
            aggregation_strategy="simple",
            device=device if device != "cpu" else -1,
        )

    def extract(self, text: str) -> list[Finding]:
        out: list[Finding] = []
        for ent in self._ner(text):
            concept_id = str(ent["entity_group"])
            if concept_id not in self.kg.concepts:
                continue
            start, end = int(ent["start"]), int(ent["end"])
            out.append(
                Finding(
                    concept_id=concept_id,
                    label=self.kg.concepts[concept_id].label,
                    span=text[start:end],
                    start=start,
                    end=end,
                    negated=is_negated(text, start),
                    language_register="model",
                )
            )
        return _resolve(out, self.kg)

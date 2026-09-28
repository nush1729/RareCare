"""Evidence chains and user-facing wording. Templates only: no free-form generation
touches the decision or its justification."""

from __future__ import annotations

from rarecare.kg.graph import KnowledgeGraph
from rarecare.schemas import EvidenceLink, Finding, Tier

HEADLINES = {
    Tier.URGENT: "Please see a doctor within 2 weeks.",
    Tier.SOON: "Please book a check-up with a doctor or health worker soon.",
    Tier.WATCH: "Nothing here needs urgent action right now. Watch for the signs below.",
    Tier.NEED_MORE_INFO: "One quick question will help us give you a clearer answer.",
}


def evidence_chain(
    kg: KnowledgeGraph,
    findings: list[Finding],
    criteria_prob: dict[str, float],
    resolved: dict[str, str] | None = None,
) -> list[EvidenceLink]:
    """`resolved` maps an ambiguous concept to the specific one an answer resolved it to."""
    links: list[EvidenceLink] = []
    by_id = {c.id: c for c in kg.criteria}
    resolved = resolved or {}
    for f in findings:
        if f.negated:
            continue
        concept_id = resolved.get(f.concept_id, f.concept_id)
        for crit in kg.criteria_for_concept(concept_id):
            prob = criteria_prob.get(crit.id, 0.0)
            if prob <= 0.0:
                continue
            links.append(
                EvidenceLink(
                    span=f.span,
                    concept_id=concept_id,
                    concept_label=kg.concepts[concept_id].label,
                    criterion_id=crit.id,
                    tier=by_id[crit.id].tier,
                    action=crit.action,
                    source=crit.source,
                    probability=round(prob, 4),
                    from_questionnaire=f.language_register == "questionnaire",
                )
            )
    return links


def retrieval_query(findings: list[Finding]) -> str:
    return " ".join(f.label for f in findings if not f.negated)

import pytest

from rarecare.agent.reasoner import QuestionPlanner, TierReasoner, TriageState, decode_answer
from rarecare.config import TriageConfig
from rarecare.schemas import Tier


@pytest.fixture(scope="module")
def planner(kg):
    return QuestionPlanner(TierReasoner(kg), TriageConfig())


def test_no_findings_is_watch(planner):
    d = planner.decide(TriageState())
    assert d.tier is Tier.WATCH and d.question is None


def test_distribution_sums_to_one(planner):
    dist = planner.reasoner.distribution(TriageState(present={"breast_lump", "pelvic_pain"}))
    assert sum(dist.overall.values()) == pytest.approx(1.0)


def test_breast_lump_without_age_asks_age(planner):
    d = planner.decide(TriageState(present={"breast_lump"}))
    assert d.tier is Tier.NEED_MORE_INFO
    assert d.question.variable == "age_band"


def test_breast_lump_with_age_decides(planner):
    d = planner.decide(TriageState(present={"breast_lump"}, variables={"age_band": "50-54"}))
    assert d.tier is Tier.URGENT and d.question is None


def test_postmenopausal_bleeding_is_urgent_at_any_age(planner):
    d = planner.decide(TriageState(present={"postmenopausal_bleeding"}))
    assert d.tier is Tier.URGENT


def test_ambiguous_bleeding_resolves_via_question(planner):
    st = TriageState(present={"vaginal_bleeding_unspecified"})
    d = planner.decide(st)
    assert d.question.variable == "bleeding_context"
    st.variables["bleeding_context"] = "after_menopause"
    assert planner.decide(st).tier is Tier.URGENT


def test_declining_every_question_escalates_conservatively(planner):
    st = TriageState(present={"breast_lump"}, declined={"age_band"})
    d = planner.decide(st)
    assert d.question is None
    assert d.tier is Tier.URGENT  # P(URGENT) = 0.7 >= escalation floor: never a silent downgrade


def test_question_budget_is_respected(kg):
    p = QuestionPlanner(TierReasoner(kg), TriageConfig(max_questions=0))
    assert p.decide(TriageState(present={"breast_lump"})).question is None


def test_cost_aware_prefers_cheaper_question(kg):
    # Nipple change at unknown age and laterality: age (cost 1) should beat laterality (cost 2)
    p = QuestionPlanner(TierReasoner(kg), TriageConfig(cost_exponent=1.0))
    d = p.decide(TriageState(present={"nipple_discharge"}))
    assert d.question.variable == "age_band"
    assert d.question.options[-1].value == "__decline__"


def test_decode_answer(kg):
    assert decode_answer(kg, "persistent", "true") is True
    assert decode_answer(kg, "age_band", "55+") == "55+"
    assert decode_answer(kg, "age_band", "__decline__") is None
    with pytest.raises(ValueError):
        decode_answer(kg, "age_band", "99")

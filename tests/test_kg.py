import json

import pytest

from rarecare.kg.graph import DATA_DIR, KnowledgeGraph
from rarecare.schemas import Tier


def test_graph_loads_and_links(kg):
    assert set(kg.cancers) == {"breast", "cervical", "ovarian", "endometrial"}
    assert {c.id for c in kg.criteria_for_concept("breast_lump")} == {"BR-LUMP-30", "BR-LUMP-U30"}


def test_criterion_requires_known_conditions(kg):
    crit = next(c for c in kg.criteria if c.id == "BR-LUMP-30")
    assert not crit.satisfied({"breast_lump"}, {})
    assert crit.satisfied({"breast_lump"}, {"age_band": "30-49"})
    assert not crit.satisfied({"breast_lump"}, {"age_band": "<30"})
    assert crit.tier is Tier.URGENT


def test_relevant_variables_include_resolution_targets(kg):
    # Ambiguous bleeding may resolve to post-menopausal bleeding, whose criteria depend on age.
    assert {"bleeding_context", "age_band"} <= kg.relevant_variables({"vaginal_bleeding_unspecified"})


def test_invalid_graph_is_rejected(tmp_path):
    raw = json.loads((DATA_DIR / "criteria.json").read_text())
    raw["criteria"][0]["any_of"] = ["no_such_concept"]
    bad = tmp_path / "criteria.json"
    bad.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="unknown concepts"):
        KnowledgeGraph(criteria_path=bad)

import json

from training.stigmasymp import SURFACE, Generator


def test_surface_forms_cover_known_concepts(kg):
    raw = json.loads(SURFACE.read_text())
    assert set(k for k in raw if not k.startswith("_")) <= set(kg.concepts)


def test_generator_spans_and_phrase_split(kg):
    rows = Generator(kg, seed=3).generate(400)
    assert rows
    for v in rows:
        for s in v.spans:
            assert 0 <= s.start < s.end <= len(v.text)
    phrases = {sp: set() for sp in ("train", "val")}
    for v in rows:
        for s in v.spans:
            phrases[v.split].add(v.text[s.start : s.end].lower())
    assert not phrases["train"] & phrases["val"]


def test_generator_is_deterministic(kg):
    a = [v.text for v in Generator(kg, seed=5).generate(50)]
    b = [v.text for v in Generator(kg, seed=5).generate(50)]
    assert a == b

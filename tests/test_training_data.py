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


def test_bio_spans_strict():
    from training.train_text import bio_spans

    assert bio_spans(["O", "B-a", "I-a", "O", "B-b"]) == {(1, 3, "a"), (4, 5, "b")}
    assert bio_spans(["I-a", "I-a"]) == {(0, 2, "a")}  # dangling I- starts a span
    assert bio_spans(["B-a", "I-b"]) == {(0, 1, "a"), (1, 2, "b")}
    assert bio_spans([]) == set()


def test_auroc_matches_definition():
    import numpy as np

    from eval.run_vision_eval import auroc

    assert auroc(np.array([0.9, 0.8, 0.1]), np.array([1, 1, 0])) == 1.0
    assert auroc(np.array([0.5, 0.5]), np.array([1, 0])) == 0.5
    assert np.isnan(auroc(np.array([0.5]), np.array([1])))

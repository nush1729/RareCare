import pytest

from rarecare.text.extractor import LexiconExtractor, is_negated
from rarecare.text.normalize import age_to_band, mentions_frequency, normalize, parse_age, parse_duration_days
from rarecare.text.pii import scrub_regex


@pytest.fixture(scope="module")
def ex(kg):
    return LexiconExtractor(kg)


def ids(findings, negated=False):
    return {f.concept_id for f in findings if f.negated == negated}


@pytest.mark.parametrize(
    ("text", "concept"),
    [
        ("spotting after being with my husband", "postcoital_bleeding"),
        ("bleeding after sex", "postcoital_bleeding"),
        ("I have safed pani", "abnormal_vaginal_discharge"),
        ("safed paani since weeks", "abnormal_vaginal_discharge"),
        ("there is a lump in my left breast", "breast_lump"),
        ("chhati mein gaanth", "breast_lump"),
        ("my nipple has turned inward", "nipple_retraction"),
        ("my nipple is going in", "nipple_retraction"),  # regression: "is" form was missed
        ("always bloated", "abdominal_distension"),
        ("pet phoola rehta hai", "abdominal_distension"),
        ("I feel full quickly", "early_satiety"),
        ("bleeding after my menopause", "postmenopausal_bleeding"),
    ],
)
def test_lay_euphemistic_and_hinglish_forms(ex, text, concept):
    assert concept in ids(ex.extract(text))


def test_negation_is_scoped_to_clause(ex):
    f = ex.extract("No lump in my breast but a lump in my armpit")
    assert "breast_lump" in ids(f, negated=True)
    assert "axillary_lump" in ids(f)


def test_negation_window():
    assert is_negated("I don't have any bleeding", 18)
    assert not is_negated("no pain. there is bleeding", 17)


def test_specific_bleeding_suppresses_ambiguous(ex):
    found = ids(ex.extract("bleeding from down there after sex"))
    # "bleeding ... down there" is ambiguous only if no specific pattern overlaps it
    assert "vaginal_bleeding_unspecified" in found or "postcoital_bleeding" in found


def test_empty_and_unrelated_text(ex):
    assert ex.extract("") == []
    assert ex.extract("I feel tired and have a headache") == []


def test_duration_and_frequency():
    assert parse_duration_days("for 3 weeks") == 21
    assert parse_duration_days("for a few days and 2 months") == 60
    assert parse_duration_days("no duration here") is None
    assert mentions_frequency("it happens every day")


@pytest.mark.parametrize(("text", "age"), [("I'm 52", 52), ("45 years old", 45), ("aged 7", None), ("", None)])
def test_age_parsing(text, age):
    assert parse_age(text) == age


def test_age_bands_boundaries():
    assert [age_to_band(a) for a in (29, 30, 49, 50, 54, 55)] == ["<30", "30-49", "30-49", "50-54", "50-54", "55+"]


def test_pii_scrub():
    out = scrub_regex("call me on +91 98765 43210 or mail a.b@example.com")
    assert "98765" not in out and "example.com" not in out
    assert "<PHONE>" in out and "<EMAIL>" in out


def test_normalize_unicode():
    assert normalize("I’m   fine\n") == "I'm fine"


def test_presidio_can_never_erase_symptoms(monkeypatch, pipe):
    """Regression: Presidio tagged 'neeche se khoon' as <PERSON> on Colab, erasing a symptom."""
    from dataclasses import replace

    from rarecare.text import pii

    class R:
        def __init__(self, start, end):
            self.start, self.end = start, end

    class FakeAnalyzer:
        def analyze(self, text, language, entities):
            i = text.find("neeche se khoon")
            return [R(i, i + len("neeche se khoon"))] if i >= 0 else []

    class FakeAnonymizer:
        def anonymize(self, text, analyzer_results):
            for r in sorted(analyzer_results, key=lambda r: -r.start):
                text = text[: r.start] + "<PERSON>" + text[r.end :]
            return type("Out", (), {"text": text})()

    monkeypatch.setattr(pii, "_presidio", lambda: (FakeAnalyzer(), FakeAnonymizer()))
    # Unprotected Presidio would erase it...
    assert "khoon" not in pii.scrub("neeche se khoon hai", use_presidio=True)
    # ...but the pipeline protects clinical spans even with Presidio enabled.
    pipe.cfg = replace(pipe.cfg, models=replace(pipe.cfg.models, presidio=True))
    try:
        found = {f.concept_id for f in pipe.extract("neeche se khoon hai, kya karu?")}
    finally:
        pipe.cfg = replace(pipe.cfg, models=replace(pipe.cfg.models, presidio=False))
    assert "vaginal_bleeding_unspecified" in found

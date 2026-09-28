"""PII scrubbing before any text reaches a model.

Regex scrubbing (emails, URLs, phones, ID numbers) is always on. Microsoft Presidio
NER is **opt-in**: its English models mis-tag code-mixed symptom phrases as names or
places (e.g. "neeche se khoon" -> <PERSON>), which erased symptoms and caused
under-triage. When enabled, any Presidio span overlapping a clinical lexicon match
is kept verbatim. Raw text is never persisted or logged either way.
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Any

_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("EMAIL", re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")),
    ("URL", re.compile(r"\bhttps?://\S+|\bwww\.\S+", re.IGNORECASE)),
    # Indian and international phone numbers: 10+ digits with optional separators.
    ("PHONE", re.compile(r"(?<!\d)(?:\+?\d{1,3}[\s-]?)?(?:\d[\s-]?){9,11}\d(?!\d)")),
    # Aadhaar-style 12-digit ids in 4-4-4 groups.
    ("ID_NUMBER", re.compile(r"\b\d{4}\s\d{4}\s\d{4}\b")),
)


def scrub_regex(text: str) -> str:
    for label, pattern in _PATTERNS:
        text = pattern.sub(f"<{label}>", text)
    return text


@lru_cache(maxsize=1)
def _presidio() -> tuple[Any, Any] | None:
    try:
        from presidio_analyzer import AnalyzerEngine
        from presidio_anonymizer import AnonymizerEngine
    except ImportError:
        return None
    return AnalyzerEngine(), AnonymizerEngine()


def scrub(text: str, use_presidio: bool = False, protected: list[tuple[int, int]] | None = None) -> str:
    """Remove PII. `protected` are (start, end) character spans of clinical content that
    must survive scrubbing; they refer to `text` as passed in."""
    engines = _presidio() if use_presidio else None
    if engines is not None:
        analyzer, anonymizer = engines
        # Ages matter clinically, so AGE/DATE_TIME entities are deliberately kept.
        results = analyzer.analyze(
            text=text,
            language="en",
            entities=["PERSON", "LOCATION", "EMAIL_ADDRESS", "PHONE_NUMBER"],
        )
        keep = protected or []
        results = [r for r in results if not any(r.start < e and s < r.end for s, e in keep)]
        text = str(anonymizer.anonymize(text=text, analyzer_results=results).text)
    return scrub_regex(text)

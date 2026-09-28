"""PII scrubbing before any text reaches a model.

Regex scrubbing is always on. Microsoft Presidio is used on top when installed
(`pip install rarecare[ml]`), because it catches names and addresses that regexes
cannot. Neither is perfect, so raw text is also never persisted or logged.
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


def scrub(text: str, use_presidio: bool = True) -> str:
    text = scrub_regex(text)
    engines = _presidio() if use_presidio else None
    if engines is None:
        return text
    analyzer, anonymizer = engines
    # Ages matter clinically, so AGE/DATE_TIME entities are deliberately kept.
    results = analyzer.analyze(
        text=text,
        language="en",
        entities=["PERSON", "LOCATION", "EMAIL_ADDRESS", "PHONE_NUMBER"],
    )
    return str(anonymizer.anonymize(text=text, analyzer_results=results).text)

"""Light, offset-preserving text normalisation plus structured-fact parsing."""

from __future__ import annotations

import re
import unicodedata

_WS = re.compile(r"\s+")

_NUMBER_WORDS = {
    "a": 1,
    "an": 1,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "few": 3,
    "couple of": 2,
    "several": 4,
}
_UNIT_DAYS = {"day": 1, "week": 7, "month": 30, "year": 365}

_DURATION = re.compile(
    r"\b(?:for|since|past|last)?\s*(?:the\s+)?"
    r"(?P<n>\d{1,3}|a|an|one|two|three|four|five|six|seven|eight|nine|ten|few|couple of|several)\s+"
    r"(?P<unit>day|week|month|year)s?\b",
    re.IGNORECASE,
)
_FREQUENCY = re.compile(
    r"\b(every ?day|daily|all the time|always|most days|constantly|for months|for weeks|"
    r"roz|har din)\b",
    re.IGNORECASE,
)
_AGE = re.compile(
    r"\b(?:i am|i'm|im|age[d]?|aged)\s*(?P<a>\d{2})\b|\b(?P<b>\d{2})\s*(?:years? old|yrs?|yo|y/o)\b",
    re.IGNORECASE,
)


def normalize(text: str) -> str:
    """NFKC, unify quotes/dashes, collapse whitespace. Case is preserved."""
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("’", "'").replace("‘", "'").replace("–", "-")
    return _WS.sub(" ", text).strip()


def parse_duration_days(text: str) -> int | None:
    """Longest duration mentioned, in days (the most informative for persistence)."""
    best: int | None = None
    for m in _DURATION.finditer(text):
        n_raw = m.group("n").lower()
        n = int(n_raw) if n_raw.isdigit() else _NUMBER_WORDS.get(n_raw, 1)
        days = n * _UNIT_DAYS[m.group("unit").lower()]
        best = days if best is None else max(best, days)
    return best


def mentions_frequency(text: str) -> bool:
    return bool(_FREQUENCY.search(text))


def parse_age(text: str) -> int | None:
    for m in _AGE.finditer(text):
        raw = m.group("a") or m.group("b")
        age = int(raw)
        if 12 <= age <= 110:
            return age
    return None


def age_to_band(age: int) -> str:
    if age < 30:
        return "<30"
    if age < 50:
        return "30-49"
    if age < 55:
        return "50-54"
    return "55+"

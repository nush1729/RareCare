"""StigmaSymp-W: templated vignette generator for training the text models.

Each vignette combines 1-2 target symptoms (with character-level span labels),
optional negated distractors, and optional age / duration context, rendered in one
language register. Labels come from the same reasoner the app uses, so the
supervision target is exactly the guideline encoding.

Leakage control: surface forms are split *by phrase* — every phrase is assigned to
exactly one of train/val, so validation measures generalisation to unseen
phrasings. The **test set is human-written** and is never produced by this script.

Usage:
    python -m training.stigmasymp --n 6000 --out data/stigmasymp_w
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

from rarecare.agent.reasoner import QuestionPlanner, TierReasoner, TriageState
from rarecare.config import TriageConfig
from rarecare.kg.graph import KnowledgeGraph, default_graph
from rarecare.text.normalize import age_to_band

SURFACE = Path(__file__).resolve().parent / "data" / "surface_forms.json"
REGISTERS = ("clinical", "lay", "euphemistic", "hinglish")

OPENERS = {
    "clinical": ["Patient reports {s}.", "Presenting with {s}."],
    "lay": ["I have {s}.", "I've noticed {s}.", "There is {s} and I'm worried."],
    "euphemistic": [
        "I don't know how to say this but there is {s}.",
        "Something is off, {s}.",
        "Feeling embarrassed to ask, I have {s}.",
    ],
    "hinglish": ["Mujhe {s} hai.", "Kuch dino se {s} ho raha hai.", "{s} hai, kya karu?"],
}
DURATIONS = [
    ("for 2 days", 2),
    ("for a week", 7),
    ("for 3 weeks", 21),
    ("for 2 months", 60),
    ("for 6 months", 180),
    ("every day", 999),
]
NEGATION_TEMPLATES = ["No {s}.", "I don't have {s}.", "There is no {s}."]


@dataclass
class Span:
    start: int
    end: int
    concept_id: str
    negated: bool


@dataclass
class Vignette:
    id: str
    text: str
    register: str
    spans: list[Span]
    present: list[str]
    variables: dict[str, object]
    tier: str
    split: str


def _phrase_split(phrase: str, val_fraction: float) -> str:
    h = int(hashlib.sha256(phrase.encode()).hexdigest(), 16) % 1000
    return "val" if h < val_fraction * 1000 else "train"


class Generator:
    def __init__(self, kg: KnowledgeGraph, seed: int = 13, val_fraction: float = 0.2):
        self.kg = kg
        self.reasoner = TierReasoner(kg)
        self.planner = QuestionPlanner(self.reasoner, TriageConfig())
        self.rng = random.Random(seed)
        self.val_fraction = val_fraction
        raw = json.loads(SURFACE.read_text())
        self.surface: dict[str, dict[str, list[str]]] = {k: v for k, v in raw.items() if not k.startswith("_")}
        unknown = set(self.surface) - set(kg.concepts)
        if unknown:
            raise ValueError(f"surface forms for unknown concepts: {unknown}")

    def _pick(self, concept: str, register: str, split: str) -> str | None:
        forms = [p for p in self.surface[concept].get(register, []) if _phrase_split(p, self.val_fraction) == split]
        return self.rng.choice(forms) if forms else None

    def one(self, split: str) -> Vignette | None:
        rng = self.rng
        register = rng.choice(REGISTERS)
        concepts = [c for c in self.surface if self._pick(c, register, split)]
        if not concepts:
            return None
        k = 1 if rng.random() < 0.6 else 2
        targets = rng.sample(concepts, min(k, len(concepts)))

        parts: list[str] = []
        spans: list[Span] = []
        cursor = 0

        def add(template: str, phrase: str, concept: str, negated: bool) -> None:
            nonlocal cursor
            before, after = template.split("{s}")
            start = cursor + len(before)
            spans.append(Span(start, start + len(phrase), concept, negated))
            piece = before + phrase + after
            parts.append(piece)
            cursor += len(piece) + 1

        for concept in targets:
            phrase = self._pick(concept, register, split)
            assert phrase is not None
            duration = rng.choice(DURATIONS) if rng.random() < 0.6 else None
            template = rng.choice(OPENERS[register])
            if duration:
                template = template.replace("{s}", "{s} " + duration[0])
            add(template, phrase, concept, negated=False)

        if rng.random() < 0.3:
            distractors = [c for c in concepts if c not in targets]
            if distractors:
                d = rng.choice(distractors)
                phrase = self._pick(d, register, split)
                if phrase:
                    add(rng.choice(NEGATION_TEMPLATES), phrase, d, negated=True)

        variables: dict[str, object] = {}
        if rng.random() < 0.5:
            age = rng.randint(18, 80)
            parts.append(f"I am {age} years old.")
            variables["age_band"] = age_to_band(age)

        text = " ".join(parts)
        present = {s.concept_id for s in spans if not s.negated}
        days = [d for phrase, d in DURATIONS if phrase in text]
        if days:
            variables["persistent"] = max(days) >= 21
        state = TriageState(present=present, variables=dict(variables))  # type: ignore[arg-type]
        tier = self.planner.conservative_tier(self.reasoner.distribution(state))
        vid = hashlib.sha1(text.encode()).hexdigest()[:12]
        return Vignette(vid, text, register, spans, sorted(present), variables, tier.value, split)

    def generate(self, n: int) -> list[Vignette]:
        out: dict[str, Vignette] = {}
        n_val = int(n * self.val_fraction)
        for split, target in (("train", n - n_val), ("val", n_val)):
            attempts = 0
            count = 0
            while count < target and attempts < target * 20:
                attempts += 1
                v = self.one(split)
                if v is not None and v.id not in out:
                    out[v.id] = v
                    count += 1
        return list(out.values())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--out", type=Path, default=Path("data/stigmasymp_w"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rows = Generator(default_graph(), seed=args.seed).generate(args.n)
    for split in ("train", "val"):
        with (args.out / f"{split}.jsonl").open("w", encoding="utf-8") as fh:
            for v in rows:
                if v.split == split:
                    fh.write(json.dumps(asdict(v), ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} vignettes to {args.out}")


if __name__ == "__main__":
    main()

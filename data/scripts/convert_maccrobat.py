"""Convert MACCROBAT brat annotations (.txt + .ann) to the StigmaSymp jsonl schema with
generic SYMPTOM spans (entity type Sign_symptom), for stage-1 extractor training.

    python data/scripts/convert_maccrobat.py --src data/raw/maccrobat --out data/maccrobat
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from pathlib import Path

_SENT = re.compile(r"(?<=[.!?])\s+")
_ENTITY = re.compile(r"^T\d+\t(\S+) (\d+) (\d+)(?:;\d+ \d+)*\t")


def doc_rows(txt: Path) -> list[dict[str, object]]:
    text = txt.read_text(encoding="utf-8")
    ann = txt.with_suffix(".ann")
    spans = []
    for line in ann.read_text(encoding="utf-8").splitlines():
        m = _ENTITY.match(line)
        if m and m.group(1) == "Sign_symptom":
            spans.append((int(m.group(2)), int(m.group(3))))
    rows: list[dict[str, object]] = []
    offset = 0
    for sent in _SENT.split(text):
        start = text.find(sent, offset)
        end = start + len(sent)
        offset = end
        local = [
            {"start": s - start, "end": e - start, "concept_id": "SYMPTOM", "negated": False}
            for s, e in spans
            if s >= start and e <= end
        ]
        if sent.strip():
            rows.append(
                {
                    "id": hashlib.sha1(sent.encode()).hexdigest()[:12],
                    "text": sent,
                    "register": "clinical",
                    "spans": local,
                    "present": [],
                    "variables": {},
                    "tier": "",
                    "split": "",
                }
            )
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--val", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    docs = sorted(args.src.rglob("*.txt"))
    random.Random(args.seed).shuffle(docs)
    n_val = int(len(docs) * args.val)
    args.out.mkdir(parents=True, exist_ok=True)
    # Split by document, never by sentence, to avoid leakage.
    for split, subset in (("val", docs[:n_val]), ("train", docs[n_val:])):
        with (args.out / f"{split}.jsonl").open("w", encoding="utf-8") as fh:
            for d in subset:
                for r in doc_rows(d):
                    r["split"] = split
                    fh.write(json.dumps(r) + "\n")
    print(f"converted {len(docs)} documents to {args.out}")


if __name__ == "__main__":
    main()

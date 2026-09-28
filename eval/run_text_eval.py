"""Evaluate the text path on a labelled jsonl set, broken down by language register.

The per-register breakdown *is* the euphemism-gap measurement: the same criteria,
phrased clinically vs. euphemistically vs. in Hinglish.

    python -m eval.run_text_eval --data data/stigmasymp_w/val.jsonl --out reports/text_eval.json
    python -m eval.run_text_eval --data data/human_test.jsonl --config configs/trained.yaml
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from eval.metrics import bootstrap_ci, micro_f1, over_triage_rate, under_triage_rate, urgent_recall
from rarecare.agent.reasoner import TriageState
from rarecare.config import load_config
from rarecare.pipeline import RareCarePipeline


def predict(pipe: RareCarePipeline, row: dict[str, Any]) -> tuple[set[str], str]:
    findings = pipe.extract(row["text"])
    present = {f.concept_id for f in findings if not f.negated}
    state = TriageState(present=present, variables=dict(row.get("variables", {})))
    tier = pipe.planner.conservative_tier(pipe.reasoner.distribution(state))
    return present, tier.value


def evaluate(rows: list[dict[str, Any]], pipe: RareCarePipeline) -> dict[str, Any]:
    preds = [predict(pipe, r) for r in rows]
    by_register: dict[str, list[int]] = defaultdict(list)
    for i, r in enumerate(rows):
        by_register[r.get("register", "unknown")].append(i)

    def block(idx: list[int]) -> dict[str, Any]:
        arr = np.array(idx)

        def f1(sub: np.ndarray) -> float:
            return micro_f1([preds[arr[i]][0] for i in sub], [set(rows[arr[i]]["present"]) for i in sub])

        def ut(sub: np.ndarray) -> float:
            return under_triage_rate([preds[arr[i]][1] for i in sub], [rows[arr[i]]["tier"] for i in sub])

        return {
            "n": len(idx),
            "concept_micro_f1": bootstrap_ci(f1, len(idx)),
            "under_triage": bootstrap_ci(ut, len(idx)),
            "over_triage": over_triage_rate([preds[i][1] for i in idx], [rows[i]["tier"] for i in idx]),
            "urgent_recall": urgent_recall([preds[i][1] for i in idx], [rows[i]["tier"] for i in idx]),
        }

    return {
        "overall": block(list(range(len(rows)))),
        "by_register": {reg: block(idx) for reg, idx in sorted(by_register.items())},
        "component_modes": {k: v.value for k, v in pipe.modes.items()},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--config", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    rows = [json.loads(line) for line in args.data.open(encoding="utf-8") if line.strip()]
    report = evaluate(rows, RareCarePipeline(load_config(args.config)))
    text = json.dumps(report, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)
    print(text)


if __name__ == "__main__":
    main()

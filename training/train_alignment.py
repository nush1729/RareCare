"""Contrastive lay/euphemism -> clinical alignment of SapBERT (InfoNCE, in-batch negatives).

Positive pairs: (surface form in any register, clinical concept name). This targets the
"euphemism gap": BioBERT-family encoders place "safed pani" far from "abnormal
vaginal discharge". Train/val split is by phrase, as in StigmaSymp-W.

    python -m training.train_alignment --out checkpoints/sapbert-aligned
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

BASE_MODEL = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
SURFACE = Path(__file__).resolve().parent / "data" / "surface_forms.json"


def build_pairs() -> tuple[list[tuple[str, str]], dict[str, list[str]]]:
    from rarecare.kg.graph import default_graph

    kg = default_graph()
    raw = json.loads(SURFACE.read_text())
    pairs: list[tuple[str, str]] = []
    names: dict[str, list[str]] = {}
    for cid, registers in raw.items():
        if cid.startswith("_"):
            continue
        clinical = registers.get("clinical") or [kg.concepts[cid].label]
        names[cid] = [kg.concepts[cid].label, *clinical]
        for reg, phrases in registers.items():
            if reg == "clinical":
                continue
            for ph in phrases:
                pairs.append((ph, kg.concepts[cid].label))
    return pairs, names


def main() -> None:
    import torch
    import torch.nn.functional as F
    from transformers import AutoModel, AutoTokenizer

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--temperature", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModel.from_pretrained(BASE_MODEL).to(device)
    pairs, names = build_pairs()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    def embed(texts: list[str]) -> torch.Tensor:
        enc = tok(texts, padding=True, truncation=True, max_length=32, return_tensors="pt").to(device)
        return F.normalize(model(**enc).last_hidden_state[:, 0], dim=-1)

    for epoch in range(args.epochs):
        random.shuffle(pairs)
        model.train()
        total = 0.0
        for i in range(0, len(pairs), args.batch):
            batch = pairs[i : i + args.batch]
            # Duplicate targets in one batch would be false negatives; dedupe by target.
            seen: dict[str, tuple[str, str]] = {}
            for a, b in batch:
                seen.setdefault(b, (a, b))
            if len(seen) < 2:
                continue
            anchors, targets = zip(*seen.values(), strict=True)
            logits = embed(list(anchors)) @ embed(list(targets)).T / args.temperature
            labels = torch.arange(len(anchors), device=device)
            loss = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2
            loss.backward()
            opt.step()
            opt.zero_grad()
            total += float(loss)
        print(f"epoch {epoch}: loss {total:.3f}")

    args.out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(args.out))
    tok.save_pretrained(str(args.out))
    (args.out / "concept_names.json").write_text(json.dumps(names, indent=2))


if __name__ == "__main__":
    main()

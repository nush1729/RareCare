"""Neural text components loaded from fine-tuned checkpoints (see notebooks/ and training/).

* `CriteriaClassifier` — BioBERT multi-label head over concept ids with evidential
  (Dirichlet) outputs per concept. Adds recall on phrasings the lexicon misses.
* `SapBERTNormalizer` — nearest-neighbour concept linking with a SapBERT encoder
  contrastively aligned on (euphemism, clinical concept) pairs.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from rarecare.kg.graph import KnowledgeGraph


class CriteriaClassifier:
    def __init__(self, checkpoint: str, device: str = "cpu", threshold: float = 0.5):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._torch = torch
        self.tok = AutoTokenizer.from_pretrained(checkpoint)
        self.model = AutoModelForSequenceClassification.from_pretrained(checkpoint).to(device).eval()
        self.labels: list[str] = [self.model.config.id2label[i] for i in range(self.model.config.num_labels)]
        self.device = device
        self.threshold = threshold

    def predict(self, text: str) -> dict[str, float]:
        """P(concept present) per label, from per-label binary Dirichlet evidence."""
        torch = self._torch
        enc = self.tok(text, truncation=True, max_length=256, return_tensors="pt").to(self.device)
        with torch.no_grad():
            evidence = torch.nn.functional.softplus(self.model(**enc).logits[0])
        # Logits laid out as [neg_0..neg_{L-1}, pos_0..pos_{L-1}] (see training/train_text.py).
        n = len(self.labels) // 2
        neg, pos = evidence[:n] + 1, evidence[n:] + 1
        probs = (pos / (pos + neg)).cpu().tolist()
        return {self.labels[n + i].removeprefix("pos_"): p for i, p in enumerate(probs)}

    def present(self, text: str) -> set[str]:
        return {c for c, p in self.predict(text).items() if p >= self.threshold}


class SapBERTNormalizer:
    def __init__(self, checkpoint: str, kg: KnowledgeGraph, device: str = "cpu", min_sim: float = 0.75):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self._torch = torch
        self.kg = kg
        self.device = device
        self.min_sim = min_sim
        self.tok = AutoTokenizer.from_pretrained(checkpoint)
        self.model = AutoModel.from_pretrained(checkpoint).to(device).eval()
        names_path = Path(checkpoint) / "concept_names.json"
        names: dict[str, list[str]] = (
            json.loads(names_path.read_text())
            if names_path.exists()
            else {cid: [c.label] for cid, c in kg.concepts.items()}
        )
        self.index_ids = [cid for cid, ns in names.items() for _ in ns]
        self.index_emb = self._embed([n for ns in names.values() for n in ns])

    def _embed(self, texts: list[str]) -> Any:
        torch = self._torch
        with torch.no_grad():
            enc = self.tok(texts, padding=True, truncation=True, max_length=32, return_tensors="pt").to(self.device)
            emb = self.model(**enc).last_hidden_state[:, 0, :]
        return torch.nn.functional.normalize(emb, dim=-1)

    def link(self, span: str) -> tuple[str, float] | None:
        sims = (self._embed([span]) @ self.index_emb.T)[0]
        best = int(sims.argmax())
        score = float(sims[best])
        return (self.index_ids[best], score) if score >= self.min_sim else None

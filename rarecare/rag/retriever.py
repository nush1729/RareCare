"""Guideline retrieval for grounded, cited explanations.

`BM25Retriever` is pure Python and always available. `MedCPTRetriever` uses NCBI's
MedCPT query/article encoders with a FAISS inner-product index when installed.
Retrieval is restricted to passages for the cancers under discussion, so an
explanation can never cite guidance about an unrelated cancer.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from rarecare.schemas import Citation

CORPUS = Path(__file__).resolve().parent / "corpus" / "guidelines.jsonl"
_TOKEN = re.compile(r"[a-z0-9]+")
_STOP = frozenset(
    [
        "a",
        "an",
        "the",
        "and",
        "or",
        "of",
        "to",
        "in",
        "on",
        "for",
        "with",
        "is",
        "are",
        "be",
        "by",
        "as",
        "if",
        "it",
        "at",
        "from",
        "that",
        "this",
    ]
)


@dataclass(frozen=True)
class Passage:
    id: str
    cancers: tuple[str, ...]
    title: str
    url: str
    text: str


def load_corpus(path: Path = CORPUS) -> list[Passage]:
    out: list[Passage] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                d = json.loads(line)
                out.append(Passage(d["id"], tuple(d["cancers"]), d["title"], d["url"], d["text"]))
    return out


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOP]


class Retriever(Protocol):
    def search(self, query: str, cancers: set[str], k: int = 3) -> list[Citation]: ...


class BM25Retriever:
    def __init__(self, passages: list[Passage] | None = None, k1: float = 1.5, b: float = 0.75):
        self.passages = passages if passages is not None else load_corpus()
        self.k1, self.b = k1, b
        self.docs = [tokenize(p.title + " " + p.text) for p in self.passages]
        self.avgdl = sum(map(len, self.docs)) / max(1, len(self.docs))
        df: Counter[str] = Counter()
        for d in self.docs:
            df.update(set(d))
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}
        self.tf = [Counter(d) for d in self.docs]

    def _score(self, q: list[str], i: int) -> float:
        tf, dl = self.tf[i], len(self.docs[i])
        s = 0.0
        for t in q:
            if t in tf:
                f = tf[t]
                s += self.idf[t] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
        return s

    def search(self, query: str, cancers: set[str], k: int = 3) -> list[Citation]:
        q = tokenize(query)
        scored = [
            (self._score(q, i), p) for i, p in enumerate(self.passages) if not cancers or set(p.cancers) & cancers
        ]
        scored.sort(key=lambda x: x[0], reverse=True)
        return [
            Citation(doc_id=p.id, title=p.title, url=p.url, snippet=p.text, score=round(s, 4))
            for s, p in scored[:k]
            if s > 0
        ]


class MedCPTRetriever:
    def __init__(self, passages: list[Passage] | None = None, device: str = "cpu"):
        import faiss
        import numpy as np
        import torch
        from transformers import AutoModel, AutoTokenizer

        self._torch, self._np = torch, np
        self.passages = passages if passages is not None else load_corpus()
        self.device = device
        self.q_tok = AutoTokenizer.from_pretrained("ncbi/MedCPT-Query-Encoder")
        self.q_enc = AutoModel.from_pretrained("ncbi/MedCPT-Query-Encoder").to(device).eval()
        a_tok = AutoTokenizer.from_pretrained("ncbi/MedCPT-Article-Encoder")
        a_enc = AutoModel.from_pretrained("ncbi/MedCPT-Article-Encoder").to(device).eval()
        with torch.no_grad():
            enc = a_tok(
                [[p.title, p.text] for p in self.passages],
                truncation=True,
                padding=True,
                max_length=512,
                return_tensors="pt",
            ).to(device)
            emb = a_enc(**enc).last_hidden_state[:, 0, :].cpu().numpy().astype("float32")
        self.index: Any = faiss.IndexFlatIP(emb.shape[1])
        self.index.add(emb)

    def search(self, query: str, cancers: set[str], k: int = 3) -> list[Citation]:
        with self._torch.no_grad():
            enc = self.q_tok([query], truncation=True, padding=True, max_length=64, return_tensors="pt").to(self.device)
            q = self.q_enc(**enc).last_hidden_state[:, 0, :].cpu().numpy().astype("float32")
        scores, idx = self.index.search(q, len(self.passages))
        out: list[Citation] = []
        for s, i in zip(scores[0], idx[0], strict=True):
            p = self.passages[int(i)]
            if cancers and not set(p.cancers) & cancers:
                continue
            out.append(Citation(doc_id=p.id, title=p.title, url=p.url, snippet=p.text, score=round(float(s), 4)))
            if len(out) == k:
                break
        return out


def build_retriever(kind: str, device: str = "cpu") -> Retriever:
    if kind == "medcpt":
        return MedCPTRetriever(device=device)
    if kind == "bm25":
        return BM25Retriever()
    raise ValueError(f"unknown retriever {kind!r}")

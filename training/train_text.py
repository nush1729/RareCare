"""Fine-tune BioBERT for (1) symptom span extraction and (2) evidential concept detection.

Two-stage extractor training:
  stage 1: generic SYMPTOM spans on MACCROBAT (data/scripts/convert_maccrobat.py)
  stage 2: concept-specific spans on StigmaSymp-W (training/stigmasymp.py)

    python -m training.train_text extractor --generic --data data/maccrobat --out checkpoints/stage1
    python -m training.train_text extractor --init checkpoints/stage1 --data data/stigmasymp_w \
        --out checkpoints/biobert-extractor
    python -m training.train_text criteria  --data data/stigmasymp_w --out checkpoints/biobert-criteria

Designed for a single Colab T4/A100; see notebooks/03_train_text_biobert.ipynb.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np

BASE_MODEL = "dmis-lab/biobert-base-cased-v1.2"


def set_seed(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def concept_ids() -> list[str]:
    from rarecare.kg.graph import default_graph

    return sorted(default_graph().concepts)


# --------------------------------------------------------------------------- extractor


def bio_labels(generic: bool = False) -> list[str]:
    """Concept-specific BIO tags, or generic SYMPTOM tags for MACCROBAT pre-finetuning."""
    if generic:
        return ["O", "B-SYMPTOM", "I-SYMPTOM"]
    labels = ["O"]
    for c in concept_ids():
        labels += [f"B-{c}", f"I-{c}"]
    return labels


def bio_spans(tags: list[str]) -> set[tuple[int, int, str]]:
    """Exact-match entity spans (start, end, type) from a BIO sequence (seqeval-style strict)."""
    spans: set[tuple[int, int, str]] = set()
    start, kind = -1, ""
    for i, tag in enumerate([*tags, "O"]):
        inside = tag.startswith("I-") and tag[2:] == kind and start >= 0
        if inside:
            continue
        if start >= 0:
            spans.add((start, i, kind))
            start, kind = -1, ""
        if tag.startswith(("B-", "I-")):
            start, kind = i, tag[2:]
    return spans


def encode_bio(rows: list[dict[str, Any]], tokenizer: Any, label2id: dict[str, int], max_len: int = 128) -> Any:
    """Char spans -> token BIO labels via offset mapping. Negated spans are still
    entities (negation is handled by the assertion step), matching the app."""
    from datasets import Dataset

    enc = tokenizer(
        [r["text"] for r in rows], truncation=True, max_length=max_len, return_offsets_mapping=True, padding=False
    )
    all_labels = []
    for i, r in enumerate(rows):
        offsets = enc["offset_mapping"][i]
        labels = []
        for start, end in offsets:
            if start == end:
                labels.append(-100)
                continue
            tag = "O"
            for sp in r["spans"]:
                if start >= sp["start"] and end <= sp["end"]:
                    tag = ("B-" if start == sp["start"] else "I-") + sp["concept_id"]
                    break
            labels.append(label2id[tag])
        all_labels.append(labels)
    enc.pop("offset_mapping")
    enc["labels"] = all_labels
    return Dataset.from_dict(dict(enc))


def train_extractor(args: argparse.Namespace) -> None:
    from transformers import (
        AutoModelForTokenClassification,
        AutoTokenizer,
        DataCollatorForTokenClassification,
        Trainer,
        TrainingArguments,
    )

    set_seed(args.seed)
    labels = bio_labels(generic=args.generic)
    label2id = {lab: i for i, lab in enumerate(labels)}
    base = str(args.init) if args.init else BASE_MODEL
    tok = AutoTokenizer.from_pretrained(base)
    train = encode_bio(read_jsonl(args.data / "train.jsonl"), tok, label2id)
    val = encode_bio(read_jsonl(args.data / "val.jsonl"), tok, label2id)
    model = AutoModelForTokenClassification.from_pretrained(
        base,
        num_labels=len(labels),
        id2label=dict(enumerate(labels)),
        label2id=label2id,
        ignore_mismatched_sizes=True,  # stage 2 swaps the generic head for concept tags
    )

    def metrics(p: Any) -> dict[str, float]:
        preds = p.predictions.argmax(-1)
        tp = fp = fn = 0
        for pr, la in zip(preds, p.label_ids, strict=True):
            keep = la != -100
            gold = bio_spans([labels[x] for x in la[keep]])
            pred = bio_spans([labels[x] for x in pr[keep]])
            tp += len(gold & pred)
            fp += len(pred - gold)
            fn += len(gold - pred)
        prec = tp / max(1, tp + fp)
        rec = tp / max(1, tp + fn)
        return {"precision": prec, "recall": rec, "f1": 2 * prec * rec / max(1e-9, prec + rec)}

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(args.out),
            learning_rate=args.lr,
            per_device_train_batch_size=args.batch,
            per_device_eval_batch_size=64,
            num_train_epochs=args.epochs,
            weight_decay=0.01,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            seed=args.seed,
            report_to=args.report_to,
            fp16=args.fp16,
        ),
        train_dataset=train,
        eval_dataset=val,
        processing_class=tok,
        data_collator=DataCollatorForTokenClassification(tok),
        compute_metrics=metrics,
    )
    trainer.train()
    trainer.save_model(str(args.out))
    tok.save_pretrained(str(args.out))
    (args.out / "eval.json").write_text(json.dumps(trainer.evaluate(), indent=2))


# ---------------------------------------------------------------------- criteria (evidential)


def train_criteria(args: argparse.Namespace) -> None:
    """Multi-label concept presence with a per-label binary evidential head.

    Logits are laid out as [neg_0..neg_{L-1}, pos_0..pos_{L-1}]; softplus gives
    evidence; each label is an independent 2-class Dirichlet. Negated mentions are
    negatives, which is exactly what the lexicon baseline gets wrong most often.
    """
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

    from training.losses import evidential_loss

    set_seed(args.seed)
    concepts = concept_ids()
    n = len(concepts)
    id2label = {i: f"neg_{c}" for i, c in enumerate(concepts)} | {n + i: f"pos_{c}" for i, c in enumerate(concepts)}
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL,
        num_labels=2 * n,
        id2label=id2label,
        label2id={v: k for k, v in id2label.items()},
        problem_type="multi_label_classification",
    )
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)

    def batches(rows: list[dict[str, Any]], shuffle: bool) -> DataLoader[Any]:
        def collate(b: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
            enc = tok([r["text"] for r in b], truncation=True, max_length=128, padding=True, return_tensors="pt")
            y = torch.zeros(len(b), n, dtype=torch.long)
            for i, r in enumerate(b):
                for c in r["present"]:
                    y[i, concepts.index(c)] = 1
            enc["y"] = y
            return dict(enc)

        return DataLoader(rows, batch_size=args.batch, shuffle=shuffle, collate_fn=collate)  # type: ignore[arg-type]

    train_dl = batches(read_jsonl(args.data / "train.jsonl"), True)
    val_dl = batches(read_jsonl(args.data / "val.jsonl"), False)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * len(train_dl) * args.epochs), len(train_dl) * args.epochs)

    best = -1.0
    for epoch in range(args.epochs):
        model.train()
        for b in train_dl:
            y = b.pop("y").to(device)
            logits = model(**{k: v.to(device) for k, v in b.items()}).logits
            ev = torch.nn.functional.softplus(logits)
            ev2 = torch.stack([ev[:, :n], ev[:, n:]], dim=-1).reshape(-1, 2)
            loss = evidential_loss(ev2, y.reshape(-1), epoch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad()

        model.eval()
        tp = fp = fn = 0
        with torch.no_grad():
            for b in val_dl:
                y = b.pop("y").to(device)
                ev = torch.nn.functional.softplus(model(**{k: v.to(device) for k, v in b.items()}).logits)
                p_pos = (ev[:, n:] + 1) / (ev[:, n:] + ev[:, :n] + 2)
                pred = (p_pos >= 0.5).long()
                tp += int(((pred == 1) & (y == 1)).sum())
                fp += int(((pred == 1) & (y == 0)).sum())
                fn += int(((pred == 0) & (y == 1)).sum())
        f1 = 2 * tp / max(1, 2 * tp + fp + fn)
        print(f"epoch {epoch}: val micro-F1 = {f1:.4f}")
        if f1 > best:
            best = f1
            model.save_pretrained(str(args.out))
            tok.save_pretrained(str(args.out))
    (args.out / "eval.json").write_text(json.dumps({"val_micro_f1": best, "seed": args.seed}, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("task", choices=["extractor", "criteria"])
    ap.add_argument("--data", type=Path, default=Path("data/stigmasymp_w"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--report-to", default="none")
    ap.add_argument("--generic", action="store_true", help="MACCROBAT stage: generic SYMPTOM tags")
    ap.add_argument("--init", type=Path, default=None, help="start from a stage-1 checkpoint")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    {"extractor": train_extractor, "criteria": train_criteria}[args.task](args)


if __name__ == "__main__":
    main()

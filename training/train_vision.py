"""Train the multi-head ConvNeXt-Tiny (router + quality + organ evidential heads).

Input is a manifest CSV built by `data/scripts/build_vision_manifest.py`:

    path,modality,label,source,split
    data/raw/busi/malignant/x.png,breast_us,1,busi,train
    data/raw/medmnist/chest/0001.png,unsupported_medical,-1,chestmnist,train
    data/raw/outliers/coco/..jpg,non_medical,-1,coco,train

`label` is 1 = suspicious, 0 = not suspicious, -1 = no organ label.

Losses (summed):
* router CE over 5 classes (outlier exposure: non_medical images are real samples)
* quality CE on synthetic degradations (blur / dark / overexposed) of in-scope images
* evidential loss on the organ head matching the image's modality

    python -m training.train_vision --manifest data/manifests/vision.csv --out checkpoints/vision
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Any

import numpy as np

from rarecare.vision.model import ORGAN_HEADS, ROUTER_CLASSES


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


def degrade(img: Any, rng: random.Random) -> Any:
    from PIL import ImageEnhance, ImageFilter

    kind = rng.choice(["blur", "dark", "bright"])
    if kind == "blur":
        return img.filter(ImageFilter.GaussianBlur(radius=rng.uniform(4, 9)))
    if kind == "dark":
        return ImageEnhance.Brightness(img).enhance(rng.uniform(0.08, 0.2))
    return ImageEnhance.Brightness(img).enhance(rng.uniform(2.8, 4.0))


class VisionDataset:
    def __init__(self, rows: list[dict[str, str]], train: bool, degrade_p: float = 0.15, seed: int = 13):
        from torchvision import transforms as T

        self.rows = rows
        self.train = train
        self.degrade_p = degrade_p
        self.rng = random.Random(seed)
        norm = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        if train:
            # Heavy, acquisition-style augmentation: near-OOD robustness (new machines, phones).
            self.tf = T.Compose(
                [
                    T.RandomResizedCrop(224, scale=(0.6, 1.0)),
                    T.RandomHorizontalFlip(),
                    T.RandomApply([T.ColorJitter(0.3, 0.3, 0.2, 0.02)], p=0.8),
                    T.RandomApply([T.GaussianBlur(5, sigma=(0.1, 1.5))], p=0.3),
                    T.RandomRotation(12),
                    T.ToTensor(),
                    norm,
                ]
            )
        else:
            self.tf = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(), norm])

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> dict[str, Any]:
        import torch
        from PIL import Image

        r = self.rows[i]
        img = Image.open(r["path"]).convert("RGB")
        quality = 0
        in_scope = r["modality"] in ORGAN_HEADS
        if self.train and in_scope and self.rng.random() < self.degrade_p:
            img, quality = degrade(img, self.rng), 1
        return {
            "x": self.tf(img),
            "router": torch.tensor(ROUTER_CLASSES.index(r["modality"])),
            "quality": torch.tensor(quality),
            "label": torch.tensor(int(r["label"]) if quality == 0 else -1),
        }


def run(args: argparse.Namespace) -> None:
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, WeightedRandomSampler

    from rarecare.vision.model import build_model
    from training.losses import evidential_loss

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

    rows = read_manifest(args.manifest)
    train_rows = [r for r in rows if r["split"] == "train"]
    val_rows = [r for r in rows if r["split"] == "val"]
    # Balance modalities so small in-scope sets aren't drowned by MedMNIST/outliers.
    counts: dict[str, int] = {}
    for r in train_rows:
        counts[r["modality"]] = counts.get(r["modality"], 0) + 1
    weights = [1.0 / counts[r["modality"]] for r in train_rows]
    train_dl = DataLoader(
        VisionDataset(train_rows, train=True, seed=args.seed),
        batch_size=args.batch,
        sampler=WeightedRandomSampler(weights, num_samples=len(train_rows), replacement=True),
        num_workers=args.workers,
        pin_memory=device == "cuda",
    )
    val_dl = DataLoader(VisionDataset(val_rows, train=False), batch_size=64, num_workers=args.workers)

    model = build_model(pretrained=True).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=args.epochs * len(train_dl), pct_start=0.1
    )
    organ_names = list(ORGAN_HEADS)

    best_auc, history = -1.0, []
    for epoch in range(args.epochs):
        model.train()
        for b in train_dl:
            x = b["x"].to(device)
            out = model(x)
            loss = F.cross_entropy(out["router_logits"], b["router"].to(device))
            loss = loss + 0.5 * F.cross_entropy(out["quality_logits"], b["quality"].to(device))
            router = b["router"].to(device)
            label = b["label"].to(device)
            for organ in organ_names:
                mask = (router == ROUTER_CLASSES.index(organ)) & (label >= 0)
                if mask.any():
                    loss = loss + evidential_loss(out[f"evidence_{organ}"][mask], label[mask], epoch)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()

        metrics = evaluate(model, val_dl, device)
        history.append(metrics)
        print(f"epoch {epoch}: {json.dumps(metrics)}")
        score = float(np.nanmean([metrics.get(f"auroc_{o}", np.nan) for o in organ_names]))
        if score > best_auc:
            best_auc = score
            # OOD threshold: 95% of in-scope validation images must pass.
            energies = in_scope_energies(model, val_dl, device)
            meta = {
                "ood_threshold": float(np.percentile(energies, 95)),
                "energy_temperature": 1.0,
                "seed": args.seed,
                "epoch": epoch,
                "val": metrics,
            }
            args.out.mkdir(parents=True, exist_ok=True)
            torch.save({"model": model.state_dict(), "meta": meta}, args.out / "convnext_tiny.pt")
    (args.out / "history.json").write_text(json.dumps(history, indent=2))


def in_scope_energies(model: Any, dl: Any, device: str) -> list[float]:
    import torch

    from rarecare.vision.ood import energy_score

    model.eval()
    out: list[float] = []
    with torch.no_grad():
        for b in dl:
            logits = model(b["x"].to(device))["router_logits"].cpu().numpy()
            for lg, r in zip(logits, b["router"].numpy(), strict=True):
                if ROUTER_CLASSES[int(r)] in ORGAN_HEADS:
                    out.append(energy_score(lg.astype(np.float64)))
    return out


def evaluate(model: Any, dl: Any, device: str) -> dict[str, float]:
    import torch
    from sklearn.metrics import roc_auc_score

    model.eval()
    router_ok = n = 0
    per_organ: dict[str, tuple[list[float], list[int]]] = {o: ([], []) for o in ORGAN_HEADS}
    with torch.no_grad():
        for b in dl:
            out = model(b["x"].to(device))
            router = b["router"].to(device)
            router_ok += int((out["router_logits"].argmax(-1) == router).sum())
            n += len(router)
            for organ in ORGAN_HEADS:
                mask = (router == ROUTER_CLASSES.index(organ)) & (b["label"].to(device) >= 0)
                if mask.any():
                    alpha = out[f"evidence_{organ}"][mask] + 1
                    p = (alpha[:, 1] / alpha.sum(-1)).cpu().tolist()
                    per_organ[organ][0].extend(p)
                    per_organ[organ][1].extend(b["label"][mask.cpu()].tolist())
    metrics: dict[str, float] = {"router_acc": router_ok / max(1, n)}
    for organ, (p, y) in per_organ.items():
        if len(set(y)) == 2:
            metrics[f"auroc_{organ}"] = float(roc_auc_score(y, p))
    return metrics


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=13)
    run(ap.parse_args())


if __name__ == "__main__":
    main()

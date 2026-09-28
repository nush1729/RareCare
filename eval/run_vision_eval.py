"""Held-out vision evaluation on splits never used for training or model selection.

* Breast suspicion on the BreastMNIST **test** split: AUROC with bootstrap 95% CI,
  specificity at 90% / 95% sensitivity, ECE, Brier.
* Router accuracy on the MedMNIST **test** splits of the router classes.
* Conformal recall: val + test pooled, then R random calibration/test halves
  (standard practice for small data); reports mean empirical recall of the
  suspicious class vs the 1 - alpha target, and the mean flag rate.

    python -m eval.run_vision_eval --checkpoint checkpoints/vision/convnext_tiny.pt --out reports/vision_eval.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from eval.metrics import bootstrap_ci, specificity_at_sensitivity
from rarecare.uncertainty.calibration import brier_score, expected_calibration_error
from rarecare.uncertainty.conformal import PositiveClassConformal, empirical_recall
from rarecare.vision.model import ROUTER_CLASSES

IMG = {".png", ".jpg", ".jpeg"}


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    pos, neg = scores[labels == 1], scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    # Mann-Whitney U with ties counted as half.
    greater = (pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()
    return float(greater / (len(pos) * len(neg)))


def main() -> None:
    import torch
    from PIL import Image

    from rarecare.vision.model import load_checkpoint
    from rarecare.vision.preprocess import to_model_input

    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/vision/convnext_tiny.pt")
    ap.add_argument("--root", type=Path, default=Path("data/raw/medmnist"))
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--repeats", type=int, default=200)
    ap.add_argument("--out", type=Path, default=Path("reports/vision_eval.json"))
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_checkpoint(args.checkpoint, dev)

    def run(paths: list[Path]) -> tuple[np.ndarray, np.ndarray]:
        p_susp, router = [], []
        for i in range(0, len(paths), 64):
            x = torch.stack([torch.from_numpy(to_model_input(Image.open(p).convert("RGB"))) for p in paths[i : i + 64]])
            with torch.no_grad():
                out = model(x.to(dev))
            a = out["evidence_breast_us"] + 1
            p_susp.extend((a[:, 1] / a.sum(-1)).cpu().tolist())
            router.extend(out["router_logits"].argmax(-1).cpu().tolist())
        return np.array(p_susp), np.array(router)

    def split_files(split: str) -> tuple[list[Path], np.ndarray]:
        files, labels = [], []
        for cls, lab in (("benign", 0), ("malignant", 1)):
            for p in sorted((args.root / "breastmnist" / split / cls).glob("*")):
                if p.suffix.lower() in IMG:
                    files.append(p)
                    labels.append(lab)
        return files, np.array(labels)

    report: dict[str, object] = {}

    # --- breast suspicion on the untouched test split
    files, y = split_files("test")
    p, router = run(files)
    n = len(y)
    report["breast_test"] = {
        "n": n,
        "n_malignant": int(y.sum()),
        "auroc": bootstrap_ci(lambda idx: auroc(p[idx], y[idx]), n),
        "specificity_at_90_sensitivity": specificity_at_sensitivity(p, y, 0.90),
        "specificity_at_95_sensitivity": specificity_at_sensitivity(p, y, 0.95),
        "ece": expected_calibration_error(np.stack([1 - p, p], 1), y),
        "brier": brier_score(np.stack([1 - p, p], 1), y),
        "router_says_breast_us": float((router == ROUTER_CLASSES.index("breast_us")).mean()),
    }

    # --- router on unseen scans of other types
    router_acc = {}
    for ds in sorted(d for d in args.root.iterdir() if d.is_dir() and d.name != "breastmnist"):
        paths = sorted(q for q in (ds / "test" / "all").glob("*") if q.suffix.lower() in IMG)[:500]
        if paths:
            _, r = run(paths)
            router_acc[ds.name] = float((r == ROUTER_CLASSES.index("unsupported_medical")).mean())
    report["router_unsupported_medical_test_accuracy"] = router_acc

    # --- conformal: pooled val + test, repeated random halves
    vfiles, vy = split_files("val")
    vp, _ = run(vfiles)
    pool_p, pool_y = np.concatenate([vp, p]), np.concatenate([vy, y])
    rng = np.random.default_rng(13)
    recalls, flag_rates = [], []
    for _ in range(args.repeats):
        perm = rng.permutation(len(pool_y))
        cal, test = perm[: len(perm) // 2], perm[len(perm) // 2 :]
        cp = PositiveClassConformal(alpha=args.alpha).fit(pool_p[cal], pool_y[cal])
        flags = np.array([cp.flag(float(v)) for v in pool_p[test]])
        recalls.append(empirical_recall(flags, pool_y[test]))
        flag_rates.append(float(flags.mean()))
    report["conformal_breast"] = {
        "alpha": args.alpha,
        "target_recall": 1 - args.alpha,
        "pooled_n": len(pool_y),
        "pooled_malignant": int(pool_y.sum()),
        "repeats": args.repeats,
        "mean_recall": float(np.nanmean(recalls)),
        "recall_p5": float(np.nanpercentile(recalls, 5)),
        "mean_flag_rate": float(np.mean(flag_rates)),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

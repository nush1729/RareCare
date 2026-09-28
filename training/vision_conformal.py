"""Split held-out vision data into calibration / test halves, fit conformal thresholds on
the calibration half, and verify recall on the untouched test half.

    python -m training.vision_conformal --manifest data/manifests/vision.csv \
        --checkpoint checkpoints/vision/convnext_tiny.pt --alpha 0.05
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np

from rarecare.uncertainty.calibration import brier_score, expected_calibration_error
from rarecare.uncertainty.conformal import PositiveClassConformal, empirical_recall
from rarecare.vision.model import ORGAN_HEADS


def main() -> None:
    import torch
    from PIL import Image

    from rarecare.vision.model import load_checkpoint
    from rarecare.vision.preprocess import to_model_input

    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, default=Path("data/manifests/vision.csv"))
    ap.add_argument("--checkpoint", default="checkpoints/vision/convnext_tiny.pt")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--out", type=Path, default=Path("checkpoints/conformal.json"))
    ap.add_argument("--report", type=Path, default=Path("reports/vision_conformal_test.json"))
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_checkpoint(args.checkpoint, dev)
    rows = [r for r in csv.DictReader(args.manifest.open()) if r["split"] == "val" and int(r["label"]) >= 0]
    random.Random(args.seed).shuffle(rows)
    half = len(rows) // 2

    def predict(subset: list[dict[str, str]]) -> list[dict[str, object]]:
        out = []
        for r in subset:
            x = torch.from_numpy(to_model_input(Image.open(r["path"]).convert("RGB")))[None].to(dev)
            with torch.no_grad():
                a = model(x)[f"evidence_{r['modality']}"][0] + 1
            out.append(
                {
                    "cancer": ORGAN_HEADS[r["modality"]],
                    "p": float(a[1] / a.sum()),
                    "y": int(r["label"]),
                    "group": r["source"],
                }
            )
        return out

    cal, test = predict(rows[:half]), predict(rows[half:])
    conformal: dict[str, dict[str, object]] = {}
    report: dict[str, dict[str, float]] = {}
    for cancer in sorted({r["cancer"] for r in cal}):
        c = [r for r in cal if r["cancer"] == cancer]
        t = [r for r in test if r["cancer"] == cancer]
        cp = PositiveClassConformal(alpha=args.alpha).fit(
            np.array([r["p"] for r in c]), np.array([r["y"] for r in c]), np.array([r["group"] for r in c])
        )
        conformal[cancer] = {"alpha": cp.alpha, "thresholds": cp.thresholds, "n": cp.n_calibration}
        p_t = np.array([r["p"] for r in t], dtype=np.float64)
        y_t = np.array([r["y"] for r in t])
        flags = np.array([cp.flag(float(p), str(r["group"])) for p, r in zip(p_t, t, strict=True)])
        probs = np.stack([1 - p_t, p_t], 1)
        report[cancer] = {
            "alpha": args.alpha,
            "n_calibration_pos": cp.n_calibration.get("__all__", 0),
            "n_test": len(t),
            "n_test_pos": int(y_t.sum()),
            "test_recall_suspicious": empirical_recall(flags, y_t),
            "flag_rate": float(flags.mean()),
            "ece": expected_calibration_error(probs, y_t),
            "brier": brier_score(probs, y_t),
        }
    args.out.write_text(json.dumps(conformal, indent=2))
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

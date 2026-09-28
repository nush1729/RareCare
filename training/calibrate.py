"""Post-hoc calibration: fit conformal thresholds per cancer (and per subgroup)
on a held-out calibration split, then write checkpoints/conformal.json for serving.

Input: a CSV of predictions on the held-out calibration split (written by
notebooks/06_calibration_conformal.ipynb):

    cancer,p_suspicious,label,group
    breast,0.83,1,busi
    ...

    python -m training.calibrate --preds checkpoints/calib_preds.csv --alpha 0.05 --out checkpoints/conformal.json
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from rarecare.uncertainty.conformal import PositiveClassConformal, empirical_recall


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", type=Path, required=True)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--min-group", type=int, default=20)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    rows = list(csv.DictReader(args.preds.open()))
    out: dict[str, dict[str, object]] = {}
    for cancer in sorted({r["cancer"] for r in rows}):
        sub = [r for r in rows if r["cancer"] == cancer]
        p = np.array([float(r["p_suspicious"]) for r in sub])
        y = np.array([int(r["label"]) for r in sub])
        g = np.array([r.get("group", "") for r in sub])
        cp = PositiveClassConformal(alpha=args.alpha).fit(p, y, g, min_group_size=args.min_group)
        flags = np.array([cp.flag(pi, gi) for pi, gi in zip(p, g, strict=True)])
        print(
            f"{cancer}: n_pos={cp.n_calibration}, in-sample recall={empirical_recall(flags, y):.3f} "
            f"(target >= {1 - args.alpha:.2f}; verify on a separate test split)"
        )
        out[cancer] = {"alpha": cp.alpha, "thresholds": cp.thresholds, "n": cp.n_calibration}
    args.out.write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

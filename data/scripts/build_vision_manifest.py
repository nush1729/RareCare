"""Build the vision training manifest from the on-disk dataset layout.

Expected layout (after download_data.py and manual downloads):
    data/raw/busi/{benign,malignant,normal}/*.png          -> breast_us
    data/raw/sipakmed/<class folders>/*.bmp               -> cervical_cytology
    data/raw/mmotu/{benign,malignant}/*.jpg                -> ovarian_us
    data/raw/medmnist/breastmnist/<split>/{benign,malignant}/*.png
    data/raw/medmnist/<other>/<split>/all/*.png            -> unsupported_medical
    data/raw/outliers/**/*.jpg                             -> non_medical

External test sets (BUS-BRA, Herlev) are *not* included; they are evaluated
separately so they stay truly external.

    python data/scripts/build_vision_manifest.py --root data/raw --out data/manifests/vision.csv
"""

from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path

IMG = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
SIPAKMED_SUSPICIOUS = {"im_dyskeratotic", "im_koilocytotic"}
SIPAKMED_NOT = {"im_superficial-intermediate", "im_parabasal", "im_metaplastic"}


def images(folder: Path) -> list[Path]:
    return sorted(p for p in folder.rglob("*") if p.suffix.lower() in IMG and "_mask" not in p.stem)


def split_of(path: Path, val: float, group_key: str | None = None) -> str:
    """Deterministic hash split; `group_key` keeps related images (same patient/slide) together."""
    key = group_key or path.name
    return "val" if int(hashlib.sha256(key.encode()).hexdigest(), 16) % 1000 < val * 1000 else "train"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("data/raw"))
    ap.add_argument("--out", type=Path, default=Path("data/manifests/vision.csv"))
    ap.add_argument("--val", type=float, default=0.2)
    args = ap.parse_args()
    r = args.root
    rows: list[dict[str, str]] = []

    def add(path: Path, modality: str, label: int, source: str, split: str | None = None) -> None:
        rows.append(
            {
                "path": str(path),
                "modality": modality,
                "label": str(label),
                "source": source,
                "split": split or split_of(path, args.val),
            }
        )

    for cls, label in (("benign", 0), ("normal", 0), ("malignant", 1)):
        for p in images(r / "busi" / cls):
            add(p, "breast_us", label, "busi")
    if (r / "sipakmed").exists():
        for folder in (r / "sipakmed").iterdir():
            name = folder.name.lower()
            if name in SIPAKMED_SUSPICIOUS or name in SIPAKMED_NOT:
                for p in images(folder):
                    add(p, "cervical_cytology", int(name in SIPAKMED_SUSPICIOUS), "sipakmed")
    for cls, label in (("benign", 0), ("malignant", 1)):
        for p in images(r / "mmotu" / cls):
            add(p, "ovarian_us", label, "mmotu")
    for split in ("train", "val"):
        for cls, label in (("benign", 0), ("malignant", 1)):
            for p in images(r / "medmnist" / "breastmnist" / split / cls):
                add(p, "breast_us", label, "breastmnist", split)
    for ds in sorted((r / "medmnist").glob("*")) if (r / "medmnist").exists() else []:
        if ds.name == "breastmnist":
            continue
        for split in ("train", "val"):
            for p in images(ds / split / "all"):
                add(p, "unsupported_medical", -1, ds.name, split)
    for p in images(r / "outliers"):
        add(p, "non_medical", -1, "outliers")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["path", "modality", "label", "source", "split"])
        w.writeheader()
        w.writerows(rows)
    by: dict[str, int] = {}
    for row in rows:
        by[row["modality"]] = by.get(row["modality"], 0) + 1
    print(f"wrote {len(rows)} rows to {args.out}: {by}")


if __name__ == "__main__":
    main()

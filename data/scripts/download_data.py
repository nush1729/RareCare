"""Fetch the datasets that can be scripted; print instructions for the rest.

    python data/scripts/download_data.py --root data/raw            # everything automatable
    python data/scripts/download_data.py --root data/raw --list     # show registry status

MedMNIST+ is exported to PNG folders so every image dataset shares one on-disk layout:
    data/raw/<dataset>/<class>/<file>.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

REGISTRY = Path(__file__).resolve().parents[1] / "registry.yaml"

# MedMNIST flag -> (rarecare modality, class mapping or None for "no organ label")
MEDMNIST_EXPORTS = {
    "breastmnist": ("breast_us", {0: "malignant", 1: "benign"}),  # BreastMNIST: 0 = malignant, 1 = normal/benign
    "chestmnist": ("unsupported_medical", None),
    "octmnist": ("unsupported_medical", None),
    "retinamnist": ("unsupported_medical", None),
    "organamnist": ("unsupported_medical", None),
    "bloodmnist": ("unsupported_medical", None),
    "pneumoniamnist": ("unsupported_medical", None),
    "dermamnist": ("unsupported_medical", None),
}


def export_medmnist(root: Path, size: int, per_split_cap: int, flags: list[str] | None = None) -> None:
    import medmnist
    from medmnist import INFO

    for flag, (_modality, mapping) in MEDMNIST_EXPORTS.items():
        if flags and flag not in flags:
            continue
        cls = getattr(medmnist, INFO[flag]["python_class"])
        cache = root / "_medmnist_cache"
        cache.mkdir(parents=True, exist_ok=True)
        for split in ("train", "val", "test"):
            ds = cls(split=split, download=True, size=size, root=str(cache))
            out = root / "medmnist" / flag / split
            n = min(len(ds), per_split_cap)
            for i in range(n):
                img, label = ds[i]
                lab = int(label[0]) if hasattr(label, "__len__") else int(label)
                sub = mapping[lab] if mapping else "all"
                (out / sub).mkdir(parents=True, exist_ok=True)
                img.save(out / sub / f"{i:06d}.png")
            print(f"exported {n} {flag}/{split} images")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("data/raw"))
    ap.add_argument("--size", type=int, default=224, choices=[28, 64, 128, 224])
    ap.add_argument("--cap", type=int, default=3000, help="max images per MedMNIST split (router classes)")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--flags", nargs="*", help="subset of MedMNIST flags (default: all)")
    ap.add_argument("--router-size", type=int, default=None, help="smaller size for router-only sets")
    args = ap.parse_args()

    registry = yaml.safe_load(REGISTRY.read_text())
    manual = []
    for group, items in registry.items():
        for name, meta in items.items():
            if args.list:
                print(f"[{group}] {name:22s} access={meta['access']:18s} verified={meta['verified']}  {meta['url']}")
            elif str(meta["access"]).startswith("manual"):
                manual.append((name, meta))
    if args.list:
        return

    args.root.mkdir(parents=True, exist_ok=True)
    flags = args.flags or list(MEDMNIST_EXPORTS)
    in_scope = [f for f in flags if MEDMNIST_EXPORTS[f][1] is not None]
    router = [f for f in flags if MEDMNIST_EXPORTS[f][1] is None]
    export_medmnist(args.root, args.size, args.cap, in_scope)
    if router:
        export_medmnist(args.root, args.router_size or args.size, args.cap, router)
    print("\nManual datasets (licence/registration required). Download and extract to data/raw/<name>/:")
    for name, meta in manual:
        print(f"  - {name}: {meta['url']}  ({meta['role']})")


if __name__ == "__main__":
    main()

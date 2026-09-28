#!/usr/bin/env bash
# Full training pipeline for a Colab GPU runtime. Launch from the Colab Terminal:
#
#   cd /content/RareCare && git pull -q && \
#   nohup bash scripts/colab_train_all.sh > /content/drive/MyDrive/rarecare/train.log 2>&1 &
#
# Each stage is skipped if its output already exists, so re-running resumes after a disconnect.
# Checkpoints and reports are written straight to Google Drive.
set -euo pipefail

DRIVE=/content/drive/MyDrive/rarecare
cd "$(dirname "$0")/.."
mkdir -p "$DRIVE/checkpoints" "$DRIVE/reports" data/raw
for d in checkpoints reports; do
  [ -L "$d" ] || { rm -rf "$d"; ln -s "$DRIVE/$d" "$d"; }
done
if [ -d "$DRIVE/raw" ]; then
  for p in "$DRIVE"/raw/*; do [ -e "data/raw/$(basename "$p")" ] || ln -s "$p" "data/raw/$(basename "$p")"; done
fi

stage() { echo; echo "=== [$(date +%H:%M:%S)] $* ==="; }

stage "0/7 dependencies"
python -m pip install -q -e . transformers timm datasets accelerate
python -m pip install -q --no-deps medmnist   # its CLI dep `fire` is sdist-only
python -c "import torch; print('GPU:', torch.cuda.get_device_name(0))"

stage "1/7 StigmaSymp-W vignettes"
[ -f data/stigmasymp_w/train.jsonl ] || python -m training.stigmasymp --n 6000 --seed 13 --out data/stigmasymp_w

stage "2/7 BioBERT evidential concept classifier (3 seeds)"
for seed in 13 17 23; do
  [ -f "checkpoints/biobert-criteria-s$seed/eval.json" ] || \
    python -m training.train_text criteria --data data/stigmasymp_w --out "checkpoints/biobert-criteria-s$seed" \
      --epochs 4 --seed "$seed"
  cat "checkpoints/biobert-criteria-s$seed/eval.json"
done
[ -d checkpoints/biobert-criteria ] || cp -r checkpoints/biobert-criteria-s13 checkpoints/biobert-criteria

stage "3/7 BioBERT symptom extractor"
[ -f checkpoints/biobert-extractor/eval.json ] || \
  python -m training.train_text extractor --data data/stigmasymp_w --out checkpoints/biobert-extractor --epochs 4 --fp16
cat checkpoints/biobert-extractor/eval.json

stage "4/7 SapBERT lay -> clinical alignment"
[ -f checkpoints/sapbert-aligned/concept_names.json ] || \
  python -m training.train_alignment --out checkpoints/sapbert-aligned --epochs 10

stage "5/7 vision data + ConvNeXt-Tiny"
if [ ! -f data/manifests/vision.csv ]; then
  python data/scripts/download_data.py --root data/raw --size 224 --router-size 64 --cap 1500 \
    --flags breastmnist retinamnist bloodmnist pneumoniamnist dermamnist
  python - <<'PY'
import pathlib
from datasets import load_dataset  # HF mirror: much faster than the original CIFAR host
cifar = load_dataset("uoft-cs/cifar10", split="train[:3000]")
out = pathlib.Path("data/raw/outliers/cifar"); out.mkdir(parents=True, exist_ok=True)
for i, row in enumerate(cifar):
    row["img"].convert("RGB").resize((128, 128)).save(out / f"{i:05d}.png")
print("cifar outliers:", len(cifar))
PY
  python data/scripts/build_vision_manifest.py --root data/raw --out data/manifests/vision.csv
fi
[ -f checkpoints/vision/convnext_tiny.pt ] || \
  python -m training.train_vision --manifest data/manifests/vision.csv --out checkpoints/vision \
    --epochs 15 --batch 64 --workers 4

stage "6/7 conformal calibration + held-out test check"
python -m training.vision_conformal --manifest data/manifests/vision.csv \
  --checkpoint checkpoints/vision/convnext_tiny.pt --alpha 0.05

stage "7/7 evaluation reports"
python -m eval.run_text_eval --data data/stigmasymp_w/val.jsonl --out reports/text_eval_rule_val.json > /dev/null
python -m eval.run_text_eval --data data/stigmasymp_w/val.jsonl --config configs/trained_local.yaml \
  --out reports/text_eval_trained_val.json > /dev/null
python - <<'PY'
import json
for name in ("rule", "trained"):
    r = json.load(open(f"reports/text_eval_{name}_val.json"))
    print(name, {k: (round(v["concept_micro_f1"][0], 4), round(v["under_triage"][0], 4))
                 for k, v in r["by_register"].items()})
PY
stage "DONE"

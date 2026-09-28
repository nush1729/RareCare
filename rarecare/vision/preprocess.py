"""Safe image decoding and model preprocessing."""

from __future__ import annotations

import io

import numpy as np
from numpy.typing import NDArray
from PIL import Image, UnidentifiedImageError

ALLOWED_FORMATS = {"PNG", "JPEG", "WEBP", "BMP", "TIFF"}
MAX_PIXELS = 40_000_000  # decompression-bomb guard
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])


class InvalidImageError(ValueError):
    pass


def load_image(data: bytes) -> Image.Image:
    if not data:
        raise InvalidImageError("Empty file.")
    Image.MAX_IMAGE_PIXELS = MAX_PIXELS
    try:
        with Image.open(io.BytesIO(data)) as probe:
            fmt = probe.format
            probe.verify()
        if fmt not in ALLOWED_FORMATS:
            raise InvalidImageError(f"Unsupported format {fmt}. Use PNG or JPEG.")
        img = Image.open(io.BytesIO(data))
        img.load()
    except (UnidentifiedImageError, Image.DecompressionBombError, OSError) as exc:
        raise InvalidImageError("This file is not a readable image.") from exc
    return img.convert("RGB")


def to_gray_array(img: Image.Image) -> NDArray[np.float64]:
    return np.asarray(img.convert("L"), dtype=np.float64)


def to_model_input(img: Image.Image, size: int = 224, resize: int = 256) -> NDArray[np.float32]:
    """RGB -> normalised CHW float32, identical to the training/validation transform
    (torchvision ``Resize(resize)`` on the shorter side, then ``CenterCrop(size)``).

    Serving must match training exactly: an earlier aspect-preserving thumbnail that
    never upscaled put small scans on a black canvas the model had never seen.
    """
    w, h = img.size
    scale = resize / min(w, h)
    img = img.convert("RGB").resize(
        (max(size, round(w * scale)), max(size, round(h * scale))), Image.Resampling.BILINEAR
    )
    left = (img.width - size) // 2
    top = (img.height - size) // 2
    img = img.crop((left, top, left + size, top + size))
    arr = np.asarray(img, dtype=np.float64) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    out: NDArray[np.float32] = arr.transpose(2, 0, 1).astype(np.float32)
    return out

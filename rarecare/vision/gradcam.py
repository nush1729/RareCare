"""Grad-CAM++ heatmaps for the organ heads (torch, lazy)."""

from __future__ import annotations

import base64
import io
from typing import Any

import numpy as np
from numpy.typing import NDArray
from PIL import Image


def gradcam_pp(model: Any, x: Any, organ: str, target_class: int = 1) -> NDArray[np.float64]:
    """Returns a [0,1] heatmap (H, W) for `target_class` of the organ head."""
    import torch

    acts: dict[str, torch.Tensor] = {}
    grads: dict[str, torch.Tensor] = {}
    layer = model.backbone.stages[-1]
    h1 = layer.register_forward_hook(lambda _m, _i, o: acts.__setitem__("a", o))
    h2 = layer.register_full_backward_hook(lambda _m, _gi, go: grads.__setitem__("g", go[0]))
    try:
        model.zero_grad()
        out = model(x)
        out[f"evidence_{organ}"][0, target_class].backward()
        a, g = acts["a"][0], grads["g"][0]
        g2, g3 = g.pow(2), g.pow(3)
        denom = 2 * g2 + (a * g3).sum(dim=(1, 2), keepdim=True)
        alpha = g2 / torch.where(denom != 0, denom, torch.ones_like(denom))
        weights = (alpha * torch.relu(g)).sum(dim=(1, 2))
        cam = torch.relu((weights[:, None, None] * a).sum(0))
        cam = torch.nn.functional.interpolate(cam[None, None], size=x.shape[-2:], mode="bilinear", align_corners=False)[
            0, 0
        ]
        cam = cam - cam.min()
        cam = cam / cam.max().clamp(min=1e-8)
        return np.asarray(cam.detach().cpu().numpy(), dtype=np.float64)
    finally:
        h1.remove()
        h2.remove()


def overlay_png_b64(img: Image.Image, cam: NDArray[np.float64], size: int = 224) -> str:
    base = img.convert("RGB").resize((size, size))
    heat = np.zeros((size, size, 3), dtype=np.float64)
    heat[..., 0] = cam  # red channel = attention
    blended = (np.asarray(base, dtype=np.float64) / 255.0) * 0.6 + heat * 0.4
    out = Image.fromarray((np.clip(blended, 0, 1) * 255).astype(np.uint8))
    buf = io.BytesIO()
    out.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")

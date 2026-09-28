"""Multi-head ConvNeXt-Tiny: one shared backbone, several small heads.

Heads
-----
* router  : breast_us | cervical_cytology | ovarian_us | unsupported_medical | non_medical
* quality : ok | retake
* organ   : one evidential head per organ, outputs non-negative evidence over
            {not suspicious, suspicious}; alpha = evidence + 1 (Sensoy et al., 2018)

Torch and timm are imported lazily so the web app and tests run without them.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

ROUTER_CLASSES = (
    "breast_us",
    "cervical_cytology",
    "ovarian_us",
    "unsupported_medical",
    "non_medical",
)
ORGAN_HEADS = {"breast_us": "breast", "cervical_cytology": "cervical", "ovarian_us": "ovarian"}

if TYPE_CHECKING:
    import torch


def build_model(backbone: str = "convnext_tiny", pretrained: bool = True, dropout: float = 0.2) -> Any:
    import timm
    import torch
    from torch import nn

    class RareCareVision(nn.Module):  # type: ignore[misc]
        def __init__(self) -> None:
            super().__init__()
            self.backbone = timm.create_model(backbone, pretrained=pretrained, num_classes=0)
            dim = self.backbone.num_features
            self.dropout = nn.Dropout(dropout)
            self.router = nn.Linear(dim, len(ROUTER_CLASSES))
            self.quality = nn.Linear(dim, 2)
            self.organ = nn.ModuleDict({k: nn.Linear(dim, 2) for k in ORGAN_HEADS})

        def features(self, x: torch.Tensor) -> torch.Tensor:
            return self.backbone(x)

        def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
            f = self.dropout(self.features(x))
            out = {"router_logits": self.router(f), "quality_logits": self.quality(f)}
            for name, head in self.organ.items():
                out[f"evidence_{name}"] = torch.nn.functional.softplus(head(f))
            return out

    return RareCareVision()


def load_checkpoint(path: str, device: str = "cpu") -> Any:
    import torch

    model = build_model(pretrained=False)
    state = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(state.get("model", state))
    # OOD threshold / energy temperature are fitted on validation data at train time.
    model.meta = state.get("meta", {})
    model.eval().to(device)
    return model

"""Image path: quality -> router/OOD gate -> organ evidential head -> heatmap.

Unknown images are *recognised*, not diagnosed:

* poor quality        -> RETAKE with specific guidance
* other medical image -> UNSUPPORTED_MEDICAL (router class trained on other MedMNIST sets)
* non-medical image   -> NOT_MEDICAL (router class + outlier exposure + energy threshold)
* in-scope image      -> ASSESSED (near-OOD in-scope images get wider uncertainty, not rejection)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

from rarecare.schemas import ImageAssessment, ImageStatus
from rarecare.uncertainty.conformal import PositiveClassConformal
from rarecare.uncertainty.evidential import Opinion
from rarecare.vision.model import ORGAN_HEADS, ROUTER_CLASSES
from rarecare.vision.ood import is_ood
from rarecare.vision.preprocess import load_image, to_gray_array, to_model_input
from rarecare.vision.quality import assess_quality

log = logging.getLogger(__name__)

_MESSAGES = {
    ImageStatus.UNSUPPORTED_MEDICAL: (
        "This looks like a medical image type RareCare does not assess yet "
        "(supported: breast ultrasound, cervical cytology, ovarian ultrasound). "
        "Your symptom assessment below still applies."
    ),
    ImageStatus.NOT_MEDICAL: (
        "This does not look like a medical image, so it was not used. Your symptom assessment below still applies."
    ),
    ImageStatus.MODEL_UNAVAILABLE: (
        "Image analysis models are under training. Your image passed quality checks and will be "
        "assessable once they are deployed. Your symptom assessment below still applies."
    ),
}


@dataclass(frozen=True)
class ImageOutcome:
    assessment: ImageAssessment
    cancer: str | None = None
    opinion: Opinion | None = None
    conformal_flag: bool | None = None


class ImageAssessor:
    def __init__(
        self,
        checkpoint: str | None,
        conformal: dict[str, PositiveClassConformal] | None = None,
        device: str = "cpu",
    ):
        self.device = device
        self.conformal = conformal or {}
        self.model: Any = None
        self.ood_threshold = float("inf")
        self.energy_temperature = 1.0
        if checkpoint:
            from rarecare.vision.model import load_checkpoint

            self.model = load_checkpoint(checkpoint, device)
            meta = getattr(self.model, "meta", {}) or {}
            self.ood_threshold = float(meta.get("ood_threshold", float("inf")))
            self.energy_temperature = float(meta.get("energy_temperature", 1.0))

    @property
    def available(self) -> bool:
        return self.model is not None

    def assess(self, data: bytes, with_heatmap: bool = True) -> ImageOutcome:
        img = load_image(data)  # raises InvalidImageError
        report = assess_quality(to_gray_array(img))
        if not report.ok:
            return ImageOutcome(
                ImageAssessment(
                    status=ImageStatus.RETAKE,
                    message=" ".join(report.reasons),
                    quality=report.metrics,
                )
            )
        if not self.available:
            return ImageOutcome(
                ImageAssessment(
                    status=ImageStatus.MODEL_UNAVAILABLE,
                    message=_MESSAGES[ImageStatus.MODEL_UNAVAILABLE],
                    quality=report.metrics,
                )
            )
        return self._assess_with_model(img, report.metrics, with_heatmap)

    def _assess_with_model(self, img: Any, quality: dict[str, float], with_heatmap: bool) -> ImageOutcome:
        import torch

        x = torch.from_numpy(to_model_input(img))[None].to(self.device)
        with torch.no_grad():
            out = self.model(x)
        router_logits = out["router_logits"][0].cpu().numpy().astype(np.float64)
        modality = ROUTER_CLASSES[int(router_logits.argmax())]

        if modality == "non_medical" or is_ood(router_logits, self.ood_threshold, self.energy_temperature):
            status = ImageStatus.NOT_MEDICAL
            return ImageOutcome(ImageAssessment(status=status, message=_MESSAGES[status], quality=quality))
        if modality == "unsupported_medical":
            status = ImageStatus.UNSUPPORTED_MEDICAL
            return ImageOutcome(
                ImageAssessment(status=status, message=_MESSAGES[status], modality=modality, quality=quality)
            )

        cancer = ORGAN_HEADS[modality]
        evidence = out[f"evidence_{modality}"][0].cpu().numpy().astype(np.float64)
        opinion = Opinion.from_evidence(evidence)
        p_susp = float(opinion.expected_probability()[1])
        conf = self.conformal.get(cancer)
        flag = conf.flag(p_susp) if conf else None
        # Only the positive class is calibrated (miss-rate guarantee), so the set
        # reports whether "suspicious" survives the conformal threshold.
        conformal_set = None if conf is None else (["suspicious"] if flag else ["not suspicious"])

        heatmap = None
        if with_heatmap:
            try:
                from rarecare.vision.gradcam import gradcam_pp, overlay_png_b64

                cam = gradcam_pp(self.model, x.requires_grad_(True), modality)
                heatmap = overlay_png_b64(img, cam)
            except Exception:  # explanation must never break the assessment
                log.exception("grad-cam failed")

        return ImageOutcome(
            ImageAssessment(
                status=ImageStatus.ASSESSED,
                message=f"Assessed as {modality.replace('_', ' ')}.",
                modality=modality,
                quality=quality,
                malignancy_probability=round(p_susp, 4),
                uncertainty=round(opinion.uncertainty, 4),
                conformal_set=conformal_set,
                heatmap_png_b64=heatmap,
            ),
            cancer=cancer,
            opinion=opinion,
            conformal_flag=flag,
        )

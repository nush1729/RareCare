import io

import numpy as np
import pytest
from PIL import Image

from rarecare.schemas import ImageStatus
from rarecare.vision.assessor import ImageAssessor
from rarecare.vision.ood import energy_score
from rarecare.vision.preprocess import InvalidImageError, load_image, to_model_input
from rarecare.vision.quality import assess_quality


def png(arr):
    buf = io.BytesIO()
    Image.fromarray(arr.astype(np.uint8)).save(buf, format="PNG")
    return buf.getvalue()


def textured(h=256, w=256, seed=0):
    return np.random.default_rng(seed).integers(40, 200, (h, w, 3))


def test_quality_flags_blur_dark_small():
    assert assess_quality(np.full((256, 256), 120.0)).ok is False  # flat = blurry
    assert not assess_quality(np.full((256, 256), 5.0)).ok
    assert "too small" in " ".join(assess_quality(textured(32, 32)[..., 0].astype(float)).reasons)
    assert assess_quality(textured()[..., 0].astype(float)).ok


def test_load_image_rejects_garbage_and_empty():
    with pytest.raises(InvalidImageError):
        load_image(b"")
    with pytest.raises(InvalidImageError):
        load_image(b"not an image at all")


def test_model_input_shape():
    x = to_model_input(load_image(png(textured(300, 180))))
    assert x.shape == (3, 224, 224) and x.dtype == np.float32


def test_untrained_build_reports_model_unavailable():
    out = ImageAssessor(checkpoint=None).assess(png(textured()))
    assert out.assessment.status is ImageStatus.MODEL_UNAVAILABLE
    assert out.opinion is None


def test_poor_quality_asks_for_retake():
    out = ImageAssessor(checkpoint=None).assess(png(np.full((256, 256, 3), 3)))
    assert out.assessment.status is ImageStatus.RETAKE


def test_energy_lower_for_confident_logits():
    assert energy_score(np.array([10.0, 0.0, 0.0])) < energy_score(np.array([0.1, 0.0, 0.0]))


def test_model_input_matches_training_transform():
    """Regression: small scans must be upscaled like torchvision Resize+CenterCrop, not padded."""
    torch = pytest.importorskip("torch")
    tv = pytest.importorskip("torchvision.transforms")
    img = Image.fromarray(textured(64, 64).astype(np.uint8))
    ref = tv.Compose(
        [tv.Resize(256), tv.CenterCrop(224), tv.ToTensor(), tv.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
    )(img)
    ours = torch.from_numpy(to_model_input(img))
    assert ours.shape == ref.shape
    assert (ours - ref).abs().mean() < 0.05  # PIL vs tensor bilinear differ slightly; padding would differ hugely
    assert (ours[:, :10, :10].abs().sum()) > 0  # no black padding border

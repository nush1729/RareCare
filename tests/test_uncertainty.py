import numpy as np
import pytest

from rarecare.fusion.fuse import ModalityInput, fuse_cancer, text_opinion_from_distribution
from rarecare.schemas import Tier
from rarecare.uncertainty.calibration import expected_calibration_error, fit_temperature, softmax
from rarecare.uncertainty.conformal import PositiveClassConformal, conformal_quantile, empirical_recall, or_rule
from rarecare.uncertainty.evidential import Opinion, combine, combine_all
from rarecare.uncertainty.prevalence import adjust_posterior


def test_opinion_from_evidence():
    op = Opinion.from_evidence(np.array([8.0, 0.0]))
    assert op.uncertainty == pytest.approx(0.2)
    assert op.expected_probability()[0] == pytest.approx(0.9)
    with pytest.raises(ValueError):
        Opinion.from_evidence(np.array([-1.0, 1.0]))


def test_vacuous_is_identity():
    op = Opinion.from_evidence(np.array([3.0, 7.0]))
    fused = combine(op, Opinion.vacuous(2))
    np.testing.assert_allclose(fused.belief, op.belief)
    assert fused.uncertainty == pytest.approx(op.uncertainty)


def test_agreement_reduces_uncertainty_and_conflict_is_measured():
    a = Opinion.from_evidence(np.array([0.0, 10.0]))
    agree = combine(a, Opinion.from_evidence(np.array([0.0, 10.0])))
    assert agree.uncertainty < a.uncertainty
    disagree = combine_all([a, Opinion.from_evidence(np.array([10.0, 0.0]))])
    assert disagree.conflict > 0.5


def test_total_conflict_raises():
    with pytest.raises(ValueError):
        combine(Opinion(np.array([1.0, 0.0]), 0.0), Opinion(np.array([0.0, 1.0]), 0.0))


def test_temperature_scaling_recovers_temperature():
    rng = np.random.default_rng(0)
    true_logits = rng.normal(size=(4000, 3)) * 2
    labels = np.array([rng.choice(3, p=p) for p in softmax(true_logits)])
    t = fit_temperature(true_logits * 3.0, labels)  # overconfident by 3x
    assert t == pytest.approx(3.0, rel=0.15)


def test_ece_perfect_calibration_is_small():
    rng = np.random.default_rng(1)
    p = rng.uniform(0.5, 1.0, 20000)
    y = (rng.uniform(size=p.size) < p).astype(int)
    probs = np.stack([1 - p, p], 1)
    assert expected_calibration_error(probs, np.where(y == 1, 1, 0)) < 0.02


def test_conformal_quantile_edge_cases():
    with pytest.raises(ValueError):
        conformal_quantile(np.array([]), 0.1)
    assert conformal_quantile(np.array([0.5]), 0.05) == float("inf")  # too few samples: always flag


def test_conformal_recall_guarantee_holds_on_fresh_data():
    rng = np.random.default_rng(7)

    def sample(n):
        y = rng.integers(0, 2, n)
        p = np.clip(rng.normal(0.35 + 0.35 * y, 0.2), 0, 1)
        return p, y

    p_cal, y_cal = sample(4000)
    cp = PositiveClassConformal(alpha=0.05).fit(p_cal, y_cal)
    p_test, y_test = sample(20000)
    flags = np.array([cp.flag(p) for p in p_test])
    assert empirical_recall(flags, y_test) >= 0.94


def test_mondrian_groups_fall_back_to_global():
    p = np.linspace(0, 1, 100)
    y = np.ones(100, dtype=int)
    groups = np.array(["a"] * 90 + ["b"] * 10)
    cp = PositiveClassConformal(alpha=0.1).fit(p, y, groups, min_group_size=20)
    assert "a" in cp.thresholds and "b" not in cp.thresholds
    assert cp.threshold_for("b") == cp.threshold_for(None)


def test_or_rule():
    assert or_rule([None, True]) and not or_rule([None, None]) and not or_rule([False, None])


def test_image_can_escalate_but_never_downgrade():
    img_suspicious = ModalityInput(Opinion.from_evidence(np.array([0.0, 20.0])), conformal_flag=True)
    text = ModalityInput(text_opinion_from_distribution(0.0), None)
    assert fuse_cancer(Tier.SOON, text, img_suspicious).tier is Tier.URGENT
    img_benign = ModalityInput(Opinion.from_evidence(np.array([20.0, 0.0])), conformal_flag=False)
    assert fuse_cancer(Tier.URGENT, text, img_benign).tier is Tier.URGENT


def test_prevalence_adjustment():
    out = adjust_posterior(np.array([[0.5, 0.5]]), np.array([0.5, 0.5]), np.array([0.9, 0.1]))
    np.testing.assert_allclose(out, [[0.9, 0.1]])

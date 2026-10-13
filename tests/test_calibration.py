"""Calibration tests: the fitted map is monotone, smoothed, and matches observed rates."""

import pytest
from pydantic import ValidationError

from uncertainty_judge.calibration import IsotonicCalibrator


def test_rates_match_observed_errors_without_smoothing() -> None:
    """With no prior, each score maps to its own observed error rate."""
    calibrator = IsotonicCalibrator.fit([0, 0, 0, 0, 1, 1], [False, False, False, True, True, False], prior_strength=0)
    assert calibrator.predict(0) == pytest.approx(0.25)
    assert calibrator.predict(1) == pytest.approx(0.5)
    assert calibrator.n_samples == 6


def test_violations_are_pooled() -> None:
    """A riskier score never gets a lower probability: violating neighbours are merged."""
    calibrator = IsotonicCalibrator.fit([0, 0, 1, 1, 2, 2], [True, True, False, False, True, False], prior_strength=0)
    assert calibrator.scores == [0, 2]
    assert calibrator.probabilities == pytest.approx([0.5, 0.5])


def test_smoothing_keeps_risk_away_from_zero_and_one() -> None:
    """A score seen only with right decisions still carries some risk."""
    calibrator = IsotonicCalibrator.fit([0, 0, 1, 1], [False, False, True, True], prior_strength=1)
    assert 0 < calibrator.predict(0) < calibrator.predict(1) < 1
    assert calibrator.predict(0) == pytest.approx((0 + 0.5) / 3)


@pytest.mark.parametrize(("score", "expected"), [(-5.0, 0.25), (0.5, 0.25), (1.0, 0.5), (9.0, 0.5)])
def test_predict_uses_the_step_below(score: float, expected: float) -> None:
    """Unseen scores take the step below them, and scores outside the range are clamped."""
    calibrator = IsotonicCalibrator(scores=[0.0, 1.0], probabilities=[0.25, 0.5], n_samples=8)
    assert calibrator.predict(score) == expected


@pytest.mark.parametrize(
    ("scores", "labels", "prior", "message"),
    [
        ([], [], 1.0, "empty"),
        ([0.0], [True, False], 1.0, "same length"),
        ([0.0], [True], -1.0, "negative"),
    ],
)
def test_fit_rejects_bad_input(scores: list[float], labels: list[bool], prior: float, message: str) -> None:
    """Empty, mismatched or negatively smoothed input is a caller bug."""
    with pytest.raises(ValueError, match=message):
        IsotonicCalibrator.fit(scores, labels, prior_strength=prior)


@pytest.mark.parametrize(
    ("scores", "probabilities"),
    [([0.0, 1.0], [0.5]), ([1.0, 0.0], [0.1, 0.2]), ([0.0, 1.0], [0.3, 0.2])],
    ids=["length-mismatch", "unsorted-scores", "decreasing-probabilities"],
)
def test_contract_rejects_non_monotone_maps(scores: list[float], probabilities: list[float]) -> None:
    """A stored calibrator must stay a valid monotone step function."""
    with pytest.raises(ValidationError):
        IsotonicCalibrator(scores=scores, probabilities=probabilities, n_samples=4)

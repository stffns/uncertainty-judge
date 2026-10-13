"""Isotonic calibration: turn a raw risk score into a probability that matches observed error rates.

Design rule: "Calibration is mandatory. A raw score is a ranking, not a probability." The calibrator is fitted on the
calibration split only, never on the test split, and must be refitted after every retraining or upstream change.
"""

from __future__ import annotations

from bisect import bisect_right
from collections.abc import Sequence
from typing import Self

from pydantic import Field, model_validator

from uncertainty_judge.contracts import Contract, Probability


class IsotonicCalibrator(Contract):
    """A fitted, non-decreasing step function from raw score to error probability.

    ``scores[i]`` is the lowest raw score that maps to ``probabilities[i]``. Scores below the first step map to the
    first probability; scores between steps map to the step below them.
    """

    scores: list[float] = Field(min_length=1)
    probabilities: list[Probability] = Field(min_length=1)
    n_samples: int = Field(ge=1, description="Calibration rows the calibrator was fitted on.")

    @model_validator(mode="after")
    def _monotone_steps(self) -> Self:
        """Require strictly increasing scores and non-decreasing probabilities of the same length."""
        if len(self.scores) != len(self.probabilities):
            raise ValueError("scores and probabilities must have the same length")
        if any(a >= b for a, b in zip(self.scores, self.scores[1:], strict=False)):
            raise ValueError("scores must be strictly increasing")
        if any(a > b for a, b in zip(self.probabilities, self.probabilities[1:], strict=False)):
            raise ValueError("probabilities must be non-decreasing")
        return self

    @classmethod
    def fit(cls, scores: Sequence[float], labels: Sequence[bool], *, prior_strength: float = 1.0) -> Self:
        """Fit the calibrator with pool-adjacent-violators over smoothed per-score error rates.

        Each distinct score's error rate is shrunk toward the overall rate by ``prior_strength`` pseudo-observations,
        so a score seen a handful of times never gets a risk of exactly 0 or 1.

        Args:
            scores: Raw scores of the calibration rows; higher means riskier.
            labels: Whether each row's decision was wrong.
            prior_strength: Pseudo-observations at the overall error rate added to every distinct score.

        Returns:
            The fitted calibrator.

        Raises:
            ValueError: If inputs are empty, of different lengths, or ``prior_strength`` is negative.
        """
        if len(scores) != len(labels):
            raise ValueError("scores and labels must have the same length")
        if not scores:
            raise ValueError("cannot calibrate on an empty split")
        if prior_strength < 0:
            raise ValueError("prior_strength cannot be negative")

        base_rate = sum(labels) / len(labels)
        counts: dict[float, tuple[int, int]] = {}
        for score, label in zip(scores, labels, strict=True):
            n, errors = counts.get(score, (0, 0))
            counts[score] = (n + 1, errors + int(label))

        # Blocks of (first score, weight, smoothed rate), merged while they violate monotonicity.
        blocks: list[tuple[float, float, float]] = []
        for score in sorted(counts):
            n, errors = counts[score]
            weight = n + prior_strength
            blocks.append((score, weight, (errors + prior_strength * base_rate) / weight))
            while len(blocks) > 1 and blocks[-2][2] > blocks[-1][2]:
                right = blocks.pop()
                left = blocks.pop()
                weight = left[1] + right[1]
                blocks.append((left[0], weight, (left[1] * left[2] + right[1] * right[2]) / weight))

        return cls(
            scores=[block[0] for block in blocks],
            probabilities=[block[2] for block in blocks],
            n_samples=len(scores),
        )

    def predict(self, score: float) -> float:
        """Return the calibrated error probability for a raw score."""
        index = max(bisect_right(self.scores, score) - 1, 0)
        return self.probabilities[index]

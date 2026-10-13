"""Stage 1: the rule baseline for finding mode.

Design: "Two signals, jev margin and novelty distance, with fixed thresholds. No training. This sets the bar every later
model must beat." The rules produce a raw doubt score (how many signals fire); the only fitted part is the mandatory
calibration that turns that score into a probability.
"""

from __future__ import annotations

from collections.abc import Sequence

from pydantic import Field

from uncertainty_judge.calibration import IsotonicCalibrator
from uncertainty_judge.contracts import Contract, FindingRecord, FindingSignals, JudgeMode, JudgeScore, Probability
from uncertainty_judge.dataset import FindingExample


class BaselineRules(Contract):
    """Fixed thresholds for the two baseline signals. They are configuration, not learned parameters."""

    margin_below: Probability = Field(description="jev margin under this value counts as doubt.")
    confidence_below: Probability | None = Field(
        default=None, description="Fallback when only jev confidence is logged: confidence under this counts as doubt."
    )
    novelty_above: float = Field(ge=0.0, description="Novelty distance over this value counts as an unfamiliar case.")

    def raw_score(self, signals: FindingSignals) -> int:
        """Count the signals that fire: 0 (confident and familiar), 1, or 2 (doubtful and novel).

        The jev margin is preferred; the self-reported confidence is used only when no margin was logged.

        Raises:
            ValueError: If only a confidence was logged and no ``confidence_below`` threshold is configured.
        """
        if signals.jev_margin is not None:
            doubtful = signals.jev_margin < self.margin_below
        elif signals.jev_confidence is not None and self.confidence_below is not None:
            doubtful = signals.jev_confidence < self.confidence_below
        else:
            raise ValueError("row has only jev_confidence but the rules define no confidence_below threshold")
        novel = signals.novelty_distance > self.novelty_above
        return int(doubtful) + int(novel)


class RuleBaselineJudge(Contract):
    """The finding-mode rule baseline: fixed rules plus a calibrator fitted on the calibration split."""

    rules: BaselineRules
    calibrator: IsotonicCalibrator
    model_version: str = Field(min_length=1)

    @classmethod
    def fit(
        cls,
        rules: BaselineRules,
        calibration_examples: Sequence[FindingExample],
        *,
        model_version: str,
        prior_strength: float = 1.0,
    ) -> RuleBaselineJudge:
        """Calibrate the rules on the calibration split. Never pass test examples here.

        Args:
            rules: Fixed thresholds.
            calibration_examples: Labeled examples from the calibration split.
            model_version: Identifier stored on every score, needed to audit drift.
            prior_strength: Smoothing passed to :meth:`IsotonicCalibrator.fit`.

        Returns:
            The calibrated baseline judge.
        """
        scores = [float(rules.raw_score(example.decision.record.signals)) for example in calibration_examples]
        labels = [example.is_wrong for example in calibration_examples]
        calibrator = IsotonicCalibrator.fit(scores, labels, prior_strength=prior_strength)
        return cls(rules=rules, calibrator=calibrator, model_version=model_version)

    def score(self, record: FindingRecord) -> JudgeScore:
        """Return the calibrated probability that the decision is wrong."""
        raw = self.rules.raw_score(record.signals)
        return JudgeScore(
            mode=JudgeMode.FINDING,
            subject_id=record.finding_id,
            risk=self.calibrator.predict(float(raw)),
            model_version=self.model_version,
        )

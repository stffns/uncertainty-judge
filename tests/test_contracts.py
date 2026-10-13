"""Contract tests: the rows the judge trains on must be impossible to log in an inconsistent state."""

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from uncertainty_judge.contracts import (
    FindingLabel,
    FindingOutcome,
    FindingRecord,
    FindingSignals,
    JudgeMode,
    JudgeScore,
    Severity,
    ZoneLabel,
    ZoneOutcome,
    ZoneSignals,
)

NOW = datetime(2026, 10, 13, tzinfo=UTC)


def _signals(**overrides: object) -> FindingSignals:
    """Build valid finding signals, overriding selected fields."""
    data: dict[str, object] = {"jev_margin": 0.4, "novelty_distance": 0.2, "missing_context": False}
    data.update(overrides)
    return FindingSignals.model_validate(data)


def test_severity_rank_is_ordinal() -> None:
    """Severities rank from nit to blocker so gaps can be measured in steps."""
    assert [s.rank for s in Severity] == [0, 1, 2, 3]


def test_finding_signals_accept_confidence_instead_of_margin() -> None:
    """Accept a jev confidence value alone, in case jev does not expose scores."""
    signals = _signals(jev_margin=None, jev_confidence=0.7)
    assert signals.jev_confidence == 0.7


def test_finding_signals_require_a_jev_signal() -> None:
    """A row without any jev uncertainty signal is useless for the rule baseline."""
    with pytest.raises(ValidationError, match="jev_margin or jev_confidence"):
        _signals(jev_margin=None)


@pytest.mark.parametrize("field", ["jev_margin", "historical_correction_rate"])
def test_probabilities_are_bounded(field: str) -> None:
    """Probability fields reject values outside [0, 1]."""
    with pytest.raises(ValidationError):
        _signals(**{field: 1.5})


def test_contracts_are_frozen_and_strict() -> None:
    """Rows cannot be mutated after logging nor enriched with unknown fields."""
    signals = _signals()
    with pytest.raises(ValidationError):
        signals.novelty_distance = 0.9  # type: ignore[misc]
    with pytest.raises(ValidationError):
        _signals(future_info=True)


def test_finding_record_computes_severity_shift() -> None:
    """The disagreement signal is derived from logged severities, never rebuilt later."""
    record = FindingRecord(
        pr_id="pr-1",
        finding_id="f-1",
        decided_at=NOW,
        reviewer_severity=Severity.NIT,
        final_severity=Severity.MEDIUM,
        signals=_signals(),
    )
    assert record.severity_shift == 2
    assert record.model_dump()["severity_shift"] == 2


def test_zone_signals_reject_negative_counts() -> None:
    """Counts such as churn cannot be negative."""
    with pytest.raises(ValidationError):
        ZoneSignals(novelty_distance=0.1, file_bug_count=0, churn=-1, cyclomatic_complexity=3.0, diff_size=10)


@pytest.mark.parametrize(
    ("outcome", "gap", "is_wrong"),
    [
        (FindingOutcome.STOOD, 0, False),
        (FindingOutcome.STOOD, None, False),
        (FindingOutcome.SEVERITY_CHANGED, -2, True),
        (FindingOutcome.DISMISSED_FALSE_POSITIVE, None, True),
        (FindingOutcome.UNDERRATED_CAUSED_INCIDENT, 3, True),
        (FindingOutcome.UNDERRATED_CAUSED_INCIDENT, None, True),
    ],
)
def test_finding_label_targets(outcome: FindingOutcome, gap: int | None, is_wrong: bool) -> None:
    """Every outcome except 'stood' is a wrong decision."""
    label = FindingLabel(finding_id="f-1", outcome=outcome, severity_gap=gap, labeled_at=NOW)
    assert label.is_wrong is is_wrong


@pytest.mark.parametrize(
    ("outcome", "gap"),
    [
        (FindingOutcome.STOOD, 1),
        (FindingOutcome.SEVERITY_CHANGED, 0),
        (FindingOutcome.SEVERITY_CHANGED, None),
        (FindingOutcome.DISMISSED_FALSE_POSITIVE, 1),
        (FindingOutcome.UNDERRATED_CAUSED_INCIDENT, -1),
    ],
)
def test_finding_label_rejects_inconsistent_gap(outcome: FindingOutcome, gap: int | None) -> None:
    """The severity gap must agree with the outcome it describes."""
    with pytest.raises(ValidationError):
        FindingLabel(finding_id="f-1", outcome=outcome, severity_gap=gap, labeled_at=NOW)


def test_finding_label_gap_is_bounded() -> None:
    """There are only four severities, so a gap beyond three steps is impossible."""
    with pytest.raises(ValidationError):
        FindingLabel(finding_id="f-1", outcome=FindingOutcome.SEVERITY_CHANGED, severity_gap=4, labeled_at=NOW)


@pytest.mark.parametrize(
    ("outcome", "expected"),
    [
        (ZoneOutcome.CLEAN, False),
        (ZoneOutcome.ESCAPED_BUG, True),
        (ZoneOutcome.MUTANT_CAUGHT, False),
        (ZoneOutcome.MUTANT_MISSED, True),
    ],
)
def test_zone_label_target(outcome: ZoneOutcome, expected: bool) -> None:
    """Only bugs the review did not report count as positives."""
    assert ZoneLabel(zone_id="z-1", outcome=outcome, labeled_at=NOW).has_unreported_bug is expected


def test_judge_score_risk_is_a_probability() -> None:
    """A calibrated risk must be a probability."""
    with pytest.raises(ValidationError):
        JudgeScore(mode=JudgeMode.FINDING, subject_id="f-1", risk=-0.1, model_version="baseline-0")

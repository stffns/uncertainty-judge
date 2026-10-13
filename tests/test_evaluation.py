"""Evaluation tests: each metric against a hand-computed value."""

from datetime import UTC, datetime

import pytest

from uncertainty_judge.contracts import FindingOutcome, FindingRecord, FindingSignals, Severity
from uncertainty_judge.dataset import FindingExample, LabelSource
from uncertainty_judge.decision_log import LoggedFinding
from uncertainty_judge.evaluation import (
    RiskCoveragePoint,
    at_escalation_budget,
    auroc,
    blocker_miss_rate,
    evaluate_findings,
    expected_calibration_error,
    over_rating_rate,
    risk_coverage_curve,
)
from uncertainty_judge.routing import RouteAction

T0 = datetime(2026, 1, 1, tzinfo=UTC)
SHA = "a1b2c3d4e5f60718293a4b5c6d7e8f9012345678"


def _example(final: Severity, outcome: FindingOutcome, gap: int | None) -> FindingExample:
    """Build a labeled example with a given final severity and outcome."""
    record = FindingRecord(
        pr_id="pr-1",
        finding_id=f"f-{final}-{outcome}-{gap}",
        head_sha=SHA,
        decided_at=T0,
        reviewer_severity=final,
        final_severity=final,
        signals=FindingSignals(jev_margin=0.5, novelty_distance=0.1, missing_context=False),
    )
    source = LabelSource.WINDOW_CLOSED if outcome is FindingOutcome.STOOD else LabelSource.OUTCOME
    return FindingExample(
        decision=LoggedFinding(logged_at=T0, record=record), outcome=outcome, severity_gap=gap, source=source
    )


STOOD_MEDIUM = _example(Severity.MEDIUM, FindingOutcome.STOOD, 0)
DISMISSED_NIT = _example(Severity.NIT, FindingOutcome.DISMISSED_FALSE_POSITIVE, None)
LOWERED_BLOCKER = _example(Severity.BLOCKER, FindingOutcome.SEVERITY_CHANGED, -1)
STOOD_BLOCKER = _example(Severity.BLOCKER, FindingOutcome.STOOD, 0)
UNDERRATED_TO_BLOCKER = _example(Severity.SHOULD, FindingOutcome.UNDERRATED_CAUSED_INCIDENT, 2)
UNDERRATED_UNKNOWN = _example(Severity.NIT, FindingOutcome.UNDERRATED_CAUSED_INCIDENT, None)


@pytest.mark.parametrize(
    ("risks", "labels", "expected"),
    [
        ([0.9, 0.8, 0.2, 0.1], [True, True, False, False], 1.0),
        ([0.1, 0.2, 0.8, 0.9], [True, True, False, False], 0.0),
        ([0.5, 0.5, 0.5, 0.5], [True, False, True, False], 0.5),
        ([0.9, 0.3, 0.3, 0.1], [True, True, False, False], 0.875),
        ([0.3, 0.4], [True, True], None),
    ],
    ids=["perfect", "inverted", "all-ties", "partial-tie", "one-class"],
)
def test_auroc(risks: list[float], labels: list[bool], expected: float | None) -> None:
    """AUROC counts correctly ordered pairs, with ties as half."""
    assert auroc(risks, labels) == (pytest.approx(expected) if expected is not None else None)


def test_ece_is_zero_when_risks_match_rates() -> None:
    """A bin predicting 0.25 with one error in four is perfectly calibrated."""
    assert expected_calibration_error([0.25] * 4, [True, False, False, False]) == pytest.approx(0.0)


def test_ece_weights_bins_by_count() -> None:
    """Two bins: 0.1 predicted vs 0.5 observed (2 rows), 0.9 vs 1.0 (2 rows)."""
    ece = expected_calibration_error([0.1, 0.1, 0.9, 0.9], [True, False, True, True])
    assert ece == pytest.approx((2 * 0.4 + 2 * 0.1) / 4)


def test_ece_edge_cases() -> None:
    """Risk 1.0 lands in the last bin; empty input has no ECE; bins must be positive."""
    assert expected_calibration_error([1.0], [True]) == pytest.approx(0.0)
    assert expected_calibration_error([], []) is None
    with pytest.raises(ValueError, match="n_bins"):
        expected_calibration_error([0.5], [True], n_bins=0)


def test_risk_coverage_curve_escalates_ties_together() -> None:
    """Each point escalates every decision at or above its threshold."""
    curve = risk_coverage_curve([0.9, 0.5, 0.5, 0.1], [True, True, False, False])
    assert [(p.threshold, p.escalation_rate, p.accepted_error_rate) for p in curve] == [
        (None, 0.0, 0.5),
        (0.9, 0.25, pytest.approx(1 / 3)),
        (0.5, 0.75, 0.0),
        (0.1, 1.0, None),
    ]
    assert risk_coverage_curve([], []) == []


def test_budget_picks_the_most_escalation_within_it() -> None:
    """A 50% budget cannot use the 75% point, so it takes the 25% one."""
    curve = risk_coverage_curve([0.9, 0.5, 0.5, 0.1], [True, True, False, False])
    point = at_escalation_budget(curve, 0.5)
    assert point is not None and point.threshold == 0.9
    assert at_escalation_budget([], 0.5) is None


def test_budget_with_no_eligible_point() -> None:
    """A curve that always escalates something has no point within a zero budget."""
    curve = [RiskCoveragePoint(threshold=0.5, escalation_rate=0.5, accepted_error_rate=0.1)]
    assert at_escalation_budget(curve, 0.0) is None


def test_over_rating_counts_dismissed_and_lowered() -> None:
    """Dismissed findings and human-lowered severities are over-ratings; the rest are not."""
    examples = [STOOD_MEDIUM, DISMISSED_NIT, LOWERED_BLOCKER, UNDERRATED_TO_BLOCKER]
    assert over_rating_rate(examples) == pytest.approx(0.5)
    assert over_rating_rate([]) is None


def test_blocker_miss_rate_with_and_without_the_judge() -> None:
    """True blockers rated lower and applied unchecked are misses; the judge's escalation saves them."""
    examples = [STOOD_BLOCKER, UNDERRATED_TO_BLOCKER, LOWERED_BLOCKER, UNDERRATED_UNKNOWN]
    assert blocker_miss_rate(examples) == pytest.approx(0.5)
    actions = [RouteAction.APPLY, RouteAction.SYSTEM2_REVIEW, RouteAction.APPLY, RouteAction.APPLY]
    assert blocker_miss_rate(examples, actions) == pytest.approx(0.0)
    assert blocker_miss_rate([STOOD_MEDIUM]) is None
    with pytest.raises(ValueError, match="same length"):
        blocker_miss_rate(examples, actions[:2])


def test_evaluate_findings_reports_per_severity() -> None:
    """The report covers every severity, even empty ones, plus the overall segment."""
    examples = [STOOD_MEDIUM, DISMISSED_NIT, STOOD_BLOCKER, UNDERRATED_TO_BLOCKER]
    risks = [0.1, 0.8, 0.05, 0.7]
    report = evaluate_findings(examples, risks, escalation_budget=0.5)

    assert set(report.per_severity) == set(Severity)
    assert report.overall.n == 4 and report.overall.n_wrong == 2
    assert report.overall.auroc == pytest.approx(1.0)
    assert report.per_severity[Severity.NIT].n == 1
    assert report.per_severity[Severity.NIT].auroc is None
    assert report.blocker_miss_rate_without_judge == pytest.approx(0.5)
    assert report.blocker_miss_rate_with_judge is None


def test_evaluate_findings_validates_inputs() -> None:
    """Mismatched inputs and impossible budgets are rejected."""
    with pytest.raises(ValueError, match="same length"):
        evaluate_findings([STOOD_MEDIUM], [0.1, 0.2], escalation_budget=0.1)
    with pytest.raises(ValueError, match="escalation_budget"):
        evaluate_findings([STOOD_MEDIUM], [0.1], escalation_budget=1.5)


def test_gap_off_the_scale_is_rejected() -> None:
    """A blocker cannot be under-rated: there is no severity above it."""
    with pytest.raises(ValueError, match="severity rank"):
        _example(Severity.BLOCKER, FindingOutcome.UNDERRATED_CAUSED_INCIDENT, 1)

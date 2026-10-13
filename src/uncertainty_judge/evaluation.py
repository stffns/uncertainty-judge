"""Evaluation metrics from "Evaluation and acceptance criteria", reported overall and per severity.

"Report every metric per severity level. A good average can hide a poor blocker score."
"""

from __future__ import annotations

from collections.abc import Sequence

from pydantic import Field

from uncertainty_judge.contracts import Contract, FindingOutcome, Probability, Severity
from uncertainty_judge.dataset import FindingExample
from uncertainty_judge.routing import RouteAction

DEFAULT_ECE_BINS = 10
"""Equal-width bins used for expected calibration error."""


class RiskCoveragePoint(Contract):
    """Outcome of escalating every decision with risk at or above ``threshold``."""

    threshold: Probability | None = Field(description="Lowest escalated risk; None when nothing is escalated.")
    escalation_rate: Probability
    accepted_error_rate: Probability | None = Field(description="Error rate of what was not escalated.")


class SegmentMetrics(Contract):
    """Metrics for one segment of decisions (all of them, or one severity)."""

    n: int = Field(ge=0)
    n_wrong: int = Field(ge=0)
    error_rate: Probability | None
    auroc: Probability | None = Field(description="None when the segment has only one class.")
    ece: Probability | None = Field(description="Expected calibration error; None for an empty segment.")
    at_budget: RiskCoveragePoint | None = Field(description="Best operating point within the escalation budget.")


class FindingEvaluation(Contract):
    """Full finding-mode report: overall, per severity, and the error types the design tracks separately."""

    escalation_budget: Probability
    overall: SegmentMetrics
    per_severity: dict[Severity, SegmentMetrics]
    over_rating_rate: Probability | None
    blocker_miss_rate_without_judge: Probability | None
    blocker_miss_rate_with_judge: Probability | None = Field(description="None when no routing actions were given.")


def auroc(risks: Sequence[float], labels: Sequence[bool]) -> float | None:
    """Probability that a random wrong decision is ranked above a random right one; ties count half.

    Returns:
        The area under the ROC curve, or None if only one class is present.
    """
    _check_lengths(risks, labels)
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return None
    order = sorted(range(len(risks)), key=lambda i: risks[i])
    ranks = [0.0] * len(risks)
    start = 0
    while start < len(order):
        end = start
        while end + 1 < len(order) and risks[order[end + 1]] == risks[order[start]]:
            end += 1
        average_rank = (start + end) / 2 + 1
        for position in range(start, end + 1):
            ranks[order[position]] = average_rank
        start = end + 1
    positive_rank_sum = sum(rank for rank, label in zip(ranks, labels, strict=True) if label)
    return (positive_rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def expected_calibration_error(
    risks: Sequence[float], labels: Sequence[bool], *, n_bins: int = DEFAULT_ECE_BINS
) -> float | None:
    """Count-weighted gap between predicted risk and observed error rate over equal-width bins.

    Returns:
        The ECE, or None for empty input.

    Raises:
        ValueError: If ``n_bins`` is not positive or the inputs differ in length.
    """
    _check_lengths(risks, labels)
    if n_bins < 1:
        raise ValueError("n_bins must be positive")
    if not risks:
        return None
    totals = [[0, 0.0, 0] for _ in range(n_bins)]  # count, sum of risks, errors
    for risk, label in zip(risks, labels, strict=True):
        bucket = totals[min(int(risk * n_bins), n_bins - 1)]
        bucket[0] += 1
        bucket[1] += risk
        bucket[2] += int(label)
    return sum(abs(risk_sum - errors) for count, risk_sum, errors in totals if count) / len(risks)


def risk_coverage_curve(risks: Sequence[float], labels: Sequence[bool]) -> list[RiskCoveragePoint]:
    """Error rate on accepted decisions at every escalation threshold, from escalating nothing to everything.

    Decisions with equal risk are always escalated together, as a threshold-based router would.
    """
    _check_lengths(risks, labels)
    total = len(risks)
    if total == 0:
        return []
    errors_left = sum(labels)
    points = [RiskCoveragePoint(threshold=None, escalation_rate=0.0, accepted_error_rate=errors_left / total)]
    by_risk = sorted(zip(risks, labels, strict=True), key=lambda pair: pair[0], reverse=True)
    escalated = 0
    index = 0
    while index < total:
        threshold = by_risk[index][0]
        while index < total and by_risk[index][0] == threshold:
            errors_left -= int(by_risk[index][1])
            escalated += 1
            index += 1
        accepted = total - escalated
        points.append(
            RiskCoveragePoint(
                threshold=threshold,
                escalation_rate=escalated / total,
                accepted_error_rate=errors_left / accepted if accepted else None,
            )
        )
    return points


def at_escalation_budget(curve: Sequence[RiskCoveragePoint], budget: float) -> RiskCoveragePoint | None:
    """Return the operating point that escalates the most without exceeding ``budget``."""
    eligible = [point for point in curve if point.escalation_rate <= budget]
    return max(eligible, key=lambda point: point.escalation_rate) if eligible else None


def over_rating_rate(examples: Sequence[FindingExample]) -> float | None:
    """Share of findings rated higher than the outcome supports: dismissed, or lowered by a human."""
    if not examples:
        return None
    over_rated = sum(
        example.outcome is FindingOutcome.DISMISSED_FALSE_POSITIVE
        or (example.severity_gap is not None and example.severity_gap < 0)
        for example in examples
    )
    return over_rated / len(examples)


def blocker_miss_rate(examples: Sequence[FindingExample], actions: Sequence[RouteAction] | None = None) -> float | None:
    """Share of true blockers that were rated below blocker and applied without any check.

    With ``actions=None`` every decision is applied, which measures the pipeline without the judge. Rows whose
    corrected severity is unknown are left out.

    Returns:
        The miss rate, or None when there are no known true blockers.

    Raises:
        ValueError: If ``actions`` is given with a different length than ``examples``.
    """
    if actions is not None and len(actions) != len(examples):
        raise ValueError("examples and actions must have the same length")
    true_blockers = 0
    missed = 0
    for index, example in enumerate(examples):
        if example.corrected_severity is not Severity.BLOCKER:
            continue
        true_blockers += 1
        applied = actions is None or actions[index] is RouteAction.APPLY
        if example.decision.record.final_severity is not Severity.BLOCKER and applied:
            missed += 1
    return missed / true_blockers if true_blockers else None


def segment_metrics(risks: Sequence[float], labels: Sequence[bool], *, escalation_budget: float) -> SegmentMetrics:
    """Compute the per-segment metrics for one group of decisions."""
    _check_lengths(risks, labels)
    n_wrong = sum(labels)
    return SegmentMetrics(
        n=len(labels),
        n_wrong=n_wrong,
        error_rate=n_wrong / len(labels) if labels else None,
        auroc=auroc(risks, labels),
        ece=expected_calibration_error(risks, labels),
        at_budget=at_escalation_budget(risk_coverage_curve(risks, labels), escalation_budget),
    )


def evaluate_findings(
    examples: Sequence[FindingExample],
    risks: Sequence[float],
    *,
    escalation_budget: float,
    actions: Sequence[RouteAction] | None = None,
) -> FindingEvaluation:
    """Evaluate finding-mode risks on a labeled split, overall and per final severity.

    Args:
        examples: Labeled examples, typically the test split.
        risks: Calibrated risk for each example, in the same order.
        escalation_budget: Target share of decisions to escalate, e.g. 0.10.
        actions: Routing action per example, if available, to measure the blocker miss rate with the judge.

    Returns:
        The evaluation report.

    Raises:
        ValueError: If inputs differ in length or the budget is outside [0, 1].
    """
    _check_lengths(examples, risks)
    if not 0.0 <= escalation_budget <= 1.0:
        raise ValueError("escalation_budget must be in [0, 1]")
    labels = [example.is_wrong for example in examples]
    per_severity: dict[Severity, SegmentMetrics] = {}
    for severity in Severity:
        indices = [i for i, example in enumerate(examples) if example.decision.record.final_severity is severity]
        per_severity[severity] = segment_metrics(
            [risks[i] for i in indices], [labels[i] for i in indices], escalation_budget=escalation_budget
        )
    return FindingEvaluation(
        escalation_budget=escalation_budget,
        overall=segment_metrics(risks, labels, escalation_budget=escalation_budget),
        per_severity=per_severity,
        over_rating_rate=over_rating_rate(examples),
        blocker_miss_rate_without_judge=blocker_miss_rate(examples),
        blocker_miss_rate_with_judge=blocker_miss_rate(examples, actions) if actions is not None else None,
    )


def _check_lengths(first: Sequence[object], second: Sequence[object]) -> None:
    """Reject paired inputs of different lengths."""
    if len(first) != len(second):
        raise ValueError("paired inputs must have the same length")

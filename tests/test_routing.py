"""Routing tests: one case per cell of the policy table in docs/training-design.md."""

import pytest
from pydantic import ValidationError

from uncertainty_judge.contracts import JudgeMode, JudgeScore, Severity
from uncertainty_judge.routing import RiskBand, RouteAction, RoutingPolicy, route_finding, route_zone

POLICY = RoutingPolicy(
    blocker_escalate_above=0.05,
    medium=RiskBand(verify_above=0.2, escalate_above=0.5),
    should=RiskBand(verify_above=0.3, escalate_above=0.6),
    nit_verify_above=0.8,
    nit_novelty_above=0.7,
    zone_escalate_above=0.4,
)
LOW_NOVELTY = 0.1
HIGH_NOVELTY = 0.9


def _finding(risk: float) -> JudgeScore:
    """Build a finding-mode score."""
    return JudgeScore(mode=JudgeMode.FINDING, subject_id="f-1", risk=risk, model_version="test")


def _zone(risk: float) -> JudgeScore:
    """Build a zone-mode score."""
    return JudgeScore(mode=JudgeMode.ZONE, subject_id="z-1", risk=risk, model_version="test")


@pytest.mark.parametrize(
    ("severity", "risk", "novelty", "expected"),
    [
        (Severity.BLOCKER, 0.01, LOW_NOVELTY, RouteAction.APPLY),
        (Severity.BLOCKER, 0.3, LOW_NOVELTY, RouteAction.HUMAN_REVIEW),
        (Severity.MEDIUM, 0.1, LOW_NOVELTY, RouteAction.APPLY),
        (Severity.MEDIUM, 0.3, LOW_NOVELTY, RouteAction.CHEAP_VERIFY),
        (Severity.MEDIUM, 0.7, LOW_NOVELTY, RouteAction.SYSTEM2_REVIEW),
        (Severity.SHOULD, 0.25, LOW_NOVELTY, RouteAction.APPLY),
        (Severity.SHOULD, 0.4, LOW_NOVELTY, RouteAction.CHEAP_VERIFY),
        (Severity.SHOULD, 0.7, LOW_NOVELTY, RouteAction.SYSTEM2_REVIEW),
        (Severity.NIT, 0.5, HIGH_NOVELTY, RouteAction.APPLY),
        (Severity.NIT, 0.9, LOW_NOVELTY, RouteAction.CHEAP_VERIFY),
        (Severity.NIT, 0.9, HIGH_NOVELTY, RouteAction.SYSTEM2_REVIEW),
    ],
)
def test_route_finding_policy_table(severity: Severity, risk: float, novelty: float, expected: RouteAction) -> None:
    """Each severity follows its own row of the cost-aware policy table."""
    assert route_finding(_finding(risk), severity, novelty, POLICY) is expected


@pytest.mark.parametrize("risk", [0.0, 0.05, 0.06, 0.5, 1.0])
def test_blocker_never_gets_cheap_verify(risk: float) -> None:
    """A blocker is either applied or sent to a human, never cheaply verified."""
    assert route_finding(_finding(risk), Severity.BLOCKER, LOW_NOVELTY, POLICY) in {
        RouteAction.APPLY,
        RouteAction.HUMAN_REVIEW,
    }


def test_threshold_is_exclusive() -> None:
    """A risk exactly on a cut point stays in the lower band."""
    assert route_finding(_finding(0.2), Severity.MEDIUM, LOW_NOVELTY, POLICY) is RouteAction.APPLY


@pytest.mark.parametrize(
    ("risk", "findings", "expected"),
    [
        (0.9, 0, RouteAction.SECOND_REVIEWER),
        (0.9, 2, RouteAction.NO_ACTION),
        (0.1, 0, RouteAction.NO_ACTION),
    ],
)
def test_route_zone(risk: float, findings: int, expected: RouteAction) -> None:
    """Only risky zones the review left silent get a second reviewer."""
    assert route_zone(_zone(risk), findings, POLICY) is expected


def test_route_finding_rejects_zone_score() -> None:
    """Mixing modes is a wiring bug and must fail loudly."""
    with pytest.raises(ValueError, match="finding-mode"):
        route_finding(_zone(0.5), Severity.NIT, LOW_NOVELTY, POLICY)


def test_route_zone_rejects_finding_score() -> None:
    """Mixing modes is a wiring bug and must fail loudly."""
    with pytest.raises(ValueError, match="zone-mode"):
        route_zone(_finding(0.5), 0, POLICY)


def test_route_zone_rejects_negative_findings() -> None:
    """A negative finding count is a logging bug."""
    with pytest.raises(ValueError, match="negative"):
        route_zone(_zone(0.5), -1, POLICY)


def test_risk_band_must_be_ordered() -> None:
    """The escalation cut point cannot sit below the verification cut point."""
    with pytest.raises(ValidationError):
        RiskBand(verify_above=0.6, escalate_above=0.4)

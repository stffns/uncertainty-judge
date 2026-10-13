"""Cost-aware routing of judge scores, as specified in "Routing policy and thresholds".

Thresholds have no defaults on purpose: they must be tuned against a target escalation rate on real data, not guessed.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Self

from pydantic import BaseModel, ConfigDict, model_validator

from uncertainty_judge.contracts import JudgeMode, JudgeScore, Probability, Severity


class RouteAction(StrEnum):
    """What the pipeline does with a scored finding or zone."""

    APPLY = "apply"
    CHEAP_VERIFY = "cheap_verify"
    SYSTEM2_REVIEW = "system2_review"
    HUMAN_REVIEW = "human_review"
    SECOND_REVIEWER = "second_reviewer"
    NO_ACTION = "no_action"


class RiskBand(BaseModel):
    """Two risk cut points splitting decisions into low, medium and high risk."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    verify_above: Probability
    escalate_above: Probability

    @model_validator(mode="after")
    def _ordered(self) -> Self:
        """Reject an escalation cut point below the verification cut point."""
        if self.verify_above > self.escalate_above:
            raise ValueError("verify_above must be <= escalate_above")
        return self


class RoutingPolicy(BaseModel):
    """Per-severity thresholds. Missing a blocker costs more than a needless check, so each level has its own."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    blocker_escalate_above: Probability
    medium: RiskBand
    should: RiskBand
    nit_verify_above: Probability
    nit_novelty_above: float
    zone_escalate_above: Probability


def route_finding(score: JudgeScore, severity: Severity, novelty_distance: float, policy: RoutingPolicy) -> RouteAction:
    """Route one final severity decision.

    Nit interpretation of the policy table ("most cases apply, rarely verify, escalate only with high novelty"):
    risk gates any action, and novelty decides between a cheap check and System 2, because a novel nit is where a
    badly under-rated finding hides.

    Args:
        score: Finding-mode judge output for the decision.
        severity: Final severity assigned by jev.
        novelty_distance: Novelty signal logged with the decision.
        policy: Tuned thresholds.

    Returns:
        The action the pipeline takes.

    Raises:
        ValueError: If the score was not produced in finding mode.
    """
    if score.mode is not JudgeMode.FINDING:
        raise ValueError(f"expected a finding-mode score, got {score.mode}")
    risk = score.risk

    if severity is Severity.BLOCKER:
        return RouteAction.HUMAN_REVIEW if risk > policy.blocker_escalate_above else RouteAction.APPLY

    if severity is Severity.NIT:
        if risk <= policy.nit_verify_above:
            return RouteAction.APPLY
        if novelty_distance > policy.nit_novelty_above:
            return RouteAction.SYSTEM2_REVIEW
        return RouteAction.CHEAP_VERIFY

    band = policy.medium if severity is Severity.MEDIUM else policy.should
    if risk > band.escalate_above:
        return RouteAction.SYSTEM2_REVIEW
    if risk > band.verify_above:
        return RouteAction.CHEAP_VERIFY
    return RouteAction.APPLY


def route_zone(score: JudgeScore, findings_in_zone: int, policy: RoutingPolicy) -> RouteAction:
    """Route one changed zone.

    A high-risk zone with zero findings goes to a second, different reviewer. Zones that already have findings are
    covered by finding mode, so they need no extra action here.

    Args:
        score: Zone-mode judge output for the zone.
        findings_in_zone: Findings the review reported in the zone. Used for routing only, never as a judge feature.
        policy: Tuned thresholds.

    Returns:
        ``SECOND_REVIEWER`` or ``NO_ACTION``.

    Raises:
        ValueError: If the score was not produced in zone mode or ``findings_in_zone`` is negative.
    """
    if score.mode is not JudgeMode.ZONE:
        raise ValueError(f"expected a zone-mode score, got {score.mode}")
    if findings_in_zone < 0:
        raise ValueError("findings_in_zone cannot be negative")
    if findings_in_zone == 0 and score.risk > policy.zone_escalate_above:
        return RouteAction.SECOND_REVIEWER
    return RouteAction.NO_ACTION

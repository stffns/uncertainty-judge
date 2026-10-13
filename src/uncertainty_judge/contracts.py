"""Pydantic contracts shared by the judge, the router and the dataset builder.

Every model is frozen and rejects unknown fields, so a row logged at decision time cannot be silently mutated or
enriched later (see "Dataset rules" in ``docs/training-design.md``).
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Annotated, Self

from pydantic import BaseModel, ConfigDict, Field, computed_field, model_validator

Probability = Annotated[float, Field(ge=0.0, le=1.0)]
"""A value in the closed interval [0, 1]."""

SeverityGap = Annotated[int, Field(ge=-3, le=3)]
"""Signed distance in severity steps between two decisions (nit -> blocker is 3)."""


class _Contract(BaseModel):
    """Base for all contracts: immutable and strict about unknown fields."""

    model_config = ConfigDict(frozen=True, extra="forbid")


class Severity(StrEnum):
    """Final severity assigned by jev, ordered from least to most severe."""

    NIT = "nit"
    SHOULD = "should"
    MEDIUM = "medium"
    BLOCKER = "blocker"

    @property
    def rank(self) -> int:
        """Position of the severity on the ordinal scale (``nit`` = 0, ``blocker`` = 3)."""
        return list(Severity).index(self)


class JudgeMode(StrEnum):
    """The two modes the judge runs in."""

    FINDING = "finding"
    ZONE = "zone"


class FindingSignals(_Contract):
    """Input signals for finding mode, logged at decision time.

    ``jev_margin`` and ``jev_confidence`` are both optional because it is still open whether jev exposes per-severity
    scores; at least one of them must be present.
    """

    jev_margin: Probability | None = Field(
        default=None, description="Gap between jev's top severity score and the next one."
    )
    jev_confidence: Probability | None = Field(
        default=None, description="Self-reported jev confidence, used when scores are not exposed."
    )
    novelty_distance: float = Field(ge=0.0, description="Distance to the nearest past cases in embedding space.")
    missing_context: bool = Field(description="Finding refers to code, config or entities absent from the diff.")
    historical_correction_rate: Probability | None = Field(
        default=None, description="How often this finding type or file area was corrected before."
    )

    @model_validator(mode="after")
    def _require_jev_signal(self) -> Self:
        """Reject rows that carry no jev uncertainty signal at all."""
        if self.jev_margin is None and self.jev_confidence is None:
            raise ValueError("either jev_margin or jev_confidence must be provided")
        return self


class ZoneSignals(_Contract):
    """Input signals for zone mode, logged at decision time.

    Zone mode must stay independent of the review output, so nothing derived from the reviewer or jev belongs here.
    """

    novelty_distance: float = Field(ge=0.0, description="Distance to the nearest past diffs in embedding space.")
    file_bug_count: int = Field(ge=0, description="Past bugs, incidents and reopened PRs tied to the file.")
    churn: int = Field(ge=0, description="Recent change frequency of the changed code.")
    cyclomatic_complexity: float = Field(ge=0.0, description="Cyclomatic complexity of the changed code.")
    test_coverage: Probability | None = Field(default=None, description="Share of changed lines covered by tests.")
    diff_size: int = Field(ge=0, description="Lines changed in the whole PR.")
    historical_correction_rate: Probability | None = Field(
        default=None, description="How often this file area was corrected before."
    )


class FindingRecord(_Contract):
    """One finding-mode row: a single final severity decision and the signals seen when it was made."""

    pr_id: str = Field(min_length=1)
    finding_id: str = Field(min_length=1)
    decided_at: datetime
    reviewer_severity: Severity
    final_severity: Severity
    signals: FindingSignals

    @computed_field  # type: ignore[prop-decorator]
    @property
    def severity_shift(self) -> int:
        """Reviewer and jev disagreement: signed steps jev moved the severity (positive = raised)."""
        return self.final_severity.rank - self.reviewer_severity.rank


class ZoneRecord(_Contract):
    """One zone-mode row: a changed file or hunk and the signals seen when the PR was reviewed."""

    pr_id: str = Field(min_length=1)
    zone_id: str = Field(min_length=1, description="Stable identifier of the file or hunk.")
    decided_at: datetime
    signals: ZoneSignals


class FindingOutcome(StrEnum):
    """Real-world outcome of a finding once its observation window closes."""

    STOOD = "stood"
    SEVERITY_CHANGED = "severity_changed"
    DISMISSED_FALSE_POSITIVE = "dismissed_false_positive"
    UNDERRATED_CAUSED_INCIDENT = "underrated_caused_incident"


class FindingLabel(_Contract):
    """Outcome label for a finding-mode row. Labels come from real outcomes, never from another model."""

    finding_id: str = Field(min_length=1)
    outcome: FindingOutcome
    severity_gap: SeverityGap | None = Field(
        default=None,
        description="Second target: corrected minus final severity in steps. None when the finding was dismissed.",
    )
    labeled_at: datetime

    @computed_field  # type: ignore[prop-decorator]
    @property
    def is_wrong(self) -> bool:
        """Primary binary target: 1 when the decision did not stand."""
        return self.outcome is not FindingOutcome.STOOD

    @model_validator(mode="after")
    def _gap_matches_outcome(self) -> Self:
        """Keep the severity gap consistent with the outcome it describes."""
        gap = self.severity_gap
        if self.outcome is FindingOutcome.STOOD and gap not in (None, 0):
            raise ValueError("a decision that stood cannot have a non-zero severity gap")
        if self.outcome is FindingOutcome.SEVERITY_CHANGED and not gap:
            raise ValueError("a severity change needs a non-zero severity gap")
        if self.outcome is FindingOutcome.DISMISSED_FALSE_POSITIVE and gap is not None:
            raise ValueError("a dismissed finding has no corrected severity, so no gap")
        if self.outcome is FindingOutcome.UNDERRATED_CAUSED_INCIDENT and gap is not None and gap <= 0:
            raise ValueError("an under-rated finding must have a positive severity gap")
        return self


class ZoneOutcome(StrEnum):
    """Real-world outcome of a changed zone once its observation window closes."""

    CLEAN = "clean"
    ESCAPED_BUG = "escaped_bug"
    MUTANT_CAUGHT = "mutant_caught"
    MUTANT_MISSED = "mutant_missed"


class ZoneLabel(_Contract):
    """Outcome label for a zone-mode row."""

    zone_id: str = Field(min_length=1)
    outcome: ZoneOutcome
    labeled_at: datetime

    @computed_field  # type: ignore[prop-decorator]
    @property
    def has_unreported_bug(self) -> bool:
        """Binary target: 1 when a real or seeded bug in the zone went unreported by the review."""
        return self.outcome in (ZoneOutcome.ESCAPED_BUG, ZoneOutcome.MUTANT_MISSED)


class JudgeScore(_Contract):
    """Calibrated output of the judge for one finding or zone."""

    mode: JudgeMode
    subject_id: str = Field(min_length=1, description="finding_id in finding mode, zone_id in zone mode.")
    risk: Probability = Field(
        description="Calibrated probability that the decision is wrong (finding) or hides an unreported bug (zone)."
    )
    model_version: str = Field(min_length=1, description="Judge and calibrator version, needed to audit drift.")

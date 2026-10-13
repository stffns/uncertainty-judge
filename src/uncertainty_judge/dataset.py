"""Turn logged decisions and real outcomes into labeled examples, following "Labels and dataset construction".

Label rules:

- A **decisive outcome** (anything but "stood" / "clean") labels a decision as soon as it is known. For a shipped
  decision it only counts if it arrived before the observation window closed; later bugs are out of scope by design.
- A decision with no decisive outcome becomes a **negative** only once it shipped (it was made on the PR's final head)
  and its observation window, counted from the merge, has closed. An explicit "stood" / "clean" label never shortcuts
  the window.
- Everything else (window still open, never merged, superseded by a later head) stays **pending** and is left out.
- Nothing known after ``as_of`` is used: decisions logged later and labels recorded later are ignored, so the same
  inputs can be replayed for backtests.
"""

from __future__ import annotations

from collections.abc import Callable, Hashable, Iterable
from datetime import datetime, timedelta
from enum import StrEnum
from typing import Self

from pydantic import BaseModel, ConfigDict, model_validator

from uncertainty_judge.contracts import (
    Contract,
    FindingLabel,
    FindingOutcome,
    MergeRecord,
    Severity,
    SeverityGap,
    ZoneLabel,
    ZoneOutcome,
)
from uncertainty_judge.decision_log import LoggedFinding, LoggedZone


class LabelSource(StrEnum):
    """Where an example's label came from."""

    OUTCOME = "outcome"
    """A decisive real outcome: human correction, incident, reopened PR, fix commit or mutation test."""

    WINDOW_CLOSED = "window_closed"
    """No decisive outcome before the observation window closed."""


class FindingExample(Contract):
    """A labeled finding-mode example."""

    decision: LoggedFinding
    outcome: FindingOutcome
    severity_gap: SeverityGap | None
    source: LabelSource

    @property
    def is_wrong(self) -> bool:
        """Primary binary target: the decision did not stand."""
        return self.outcome is not FindingOutcome.STOOD

    @property
    def corrected_severity(self) -> Severity | None:
        """Severity the outcome supports, or None when unknown (dismissed, or under-rated with no gap recorded)."""
        if self.outcome is FindingOutcome.DISMISSED_FALSE_POSITIVE or self.severity_gap is None:
            return None
        return Severity.from_rank(self.decision.record.final_severity.rank + self.severity_gap)

    @model_validator(mode="after")
    def _gap_fits_the_scale(self) -> Self:
        """Reject a gap that would move the final severity off the scale."""
        _ = self.corrected_severity
        return self


class ZoneExample(Contract):
    """A labeled zone-mode example."""

    decision: LoggedZone
    outcome: ZoneOutcome
    source: LabelSource

    @property
    def has_unreported_bug(self) -> bool:
        """Binary target: a real or seeded bug in the zone went unreported by the review."""
        return self.outcome in (ZoneOutcome.ESCAPED_BUG, ZoneOutcome.MUTANT_MISSED)


class TimeSplit[E: BaseModel](BaseModel):
    """Examples split by decision time: train on the oldest, calibrate on the middle, test on the newest."""

    model_config = ConfigDict(frozen=True)

    train: list[E]
    calibration: list[E]
    test: list[E]


def build_finding_examples(
    decisions: Iterable[LoggedFinding],
    labels: Iterable[FindingLabel],
    merges: Iterable[MergeRecord],
    *,
    observation_window: timedelta,
    as_of: datetime,
) -> list[FindingExample]:
    """Label finding-mode decisions.

    Args:
        decisions: Logged finding decisions.
        labels: Real outcomes, at most one per decision.
        merges: Merged pull requests, at most one per PR.
        observation_window: Time after merge during which a decisive outcome still counts.
        as_of: Point in time the dataset is built for; later decisions and labels are ignored.

    Returns:
        Labeled examples in decision order. Pending decisions are left out.

    Raises:
        ValueError: On a non-positive window, a naive ``as_of``, or duplicate labels or merges.
    """
    _check_window(observation_window, as_of)
    labels_by_key = _index(labels, lambda label: (label.pr_id, label.head_sha, label.finding_id), "label")
    merges_by_pr = _index(merges, lambda merge: merge.pr_id, "merge")

    examples: list[FindingExample] = []
    for decision in decisions:
        record = decision.record
        if decision.logged_at > as_of:
            continue
        window_end = _window_end(merges_by_pr.get(record.pr_id), record.head_sha, observation_window)
        label = labels_by_key.get((record.pr_id, record.head_sha, record.finding_id))
        if (
            label is not None
            and label.outcome is not FindingOutcome.STOOD
            and _counts(label.labeled_at, window_end, as_of)
        ):
            examples.append(
                FindingExample(
                    decision=decision,
                    outcome=label.outcome,
                    severity_gap=label.severity_gap,
                    source=LabelSource.OUTCOME,
                )
            )
        elif window_end is not None and window_end <= as_of:
            examples.append(
                FindingExample(
                    decision=decision,
                    outcome=FindingOutcome.STOOD,
                    severity_gap=0,
                    source=LabelSource.WINDOW_CLOSED,
                )
            )
    return examples


def build_zone_examples(
    decisions: Iterable[LoggedZone],
    labels: Iterable[ZoneLabel],
    merges: Iterable[MergeRecord],
    *,
    observation_window: timedelta,
    as_of: datetime,
) -> list[ZoneExample]:
    """Label zone-mode decisions.

    A caught mutant is a decisive negative: the experiment already answered the question, so no window is needed.

    Args:
        decisions: Logged zone decisions.
        labels: Real or mutation-testing outcomes, at most one per decision.
        merges: Merged pull requests, at most one per PR.
        observation_window: Time after merge during which an escaped bug still counts.
        as_of: Point in time the dataset is built for; later decisions and labels are ignored.

    Returns:
        Labeled examples in decision order. Pending decisions are left out.

    Raises:
        ValueError: On a non-positive window, a naive ``as_of``, or duplicate labels or merges.
    """
    _check_window(observation_window, as_of)
    labels_by_key = _index(labels, lambda label: (label.pr_id, label.head_sha, label.zone_id), "label")
    merges_by_pr = _index(merges, lambda merge: merge.pr_id, "merge")

    examples: list[ZoneExample] = []
    for decision in decisions:
        record = decision.record
        if decision.logged_at > as_of:
            continue
        window_end = _window_end(merges_by_pr.get(record.pr_id), record.head_sha, observation_window)
        label = labels_by_key.get((record.pr_id, record.head_sha, record.zone_id))
        if (
            label is not None
            and label.outcome is not ZoneOutcome.CLEAN
            and _counts(label.labeled_at, window_end, as_of)
        ):
            examples.append(ZoneExample(decision=decision, outcome=label.outcome, source=LabelSource.OUTCOME))
        elif window_end is not None and window_end <= as_of:
            examples.append(ZoneExample(decision=decision, outcome=ZoneOutcome.CLEAN, source=LabelSource.WINDOW_CLOSED))
    return examples


def time_split[E: (FindingExample, ZoneExample)](
    examples: Iterable[E], *, calibration_start: datetime, test_start: datetime
) -> TimeSplit[E]:
    """Split examples by ``decided_at``, never at random, so no future pattern leaks into training.

    For a strict backtest, build the training examples with ``as_of=calibration_start`` so their labels were also
    known before the held-out periods began.

    Args:
        examples: Labeled examples of one mode.
        calibration_start: First decision time of the calibration split; earlier decisions go to training.
        test_start: First decision time of the test split.

    Returns:
        The three splits, each in input order.

    Raises:
        ValueError: If a boundary is naive or ``calibration_start`` is not before ``test_start``.
    """
    for boundary in (calibration_start, test_start):
        _require_aware(boundary)
    if calibration_start >= test_start:
        raise ValueError("calibration_start must be before test_start")

    train: list[E] = []
    calibration: list[E] = []
    test: list[E] = []
    for example in examples:
        decided_at = example.decision.record.decided_at
        if decided_at < calibration_start:
            train.append(example)
        elif decided_at < test_start:
            calibration.append(example)
        else:
            test.append(example)
    return TimeSplit[E](train=train, calibration=calibration, test=test)


def _check_window(observation_window: timedelta, as_of: datetime) -> None:
    """Validate the dataset parameters shared by both modes."""
    if observation_window <= timedelta(0):
        raise ValueError("observation_window must be positive")
    _require_aware(as_of)


def _require_aware(moment: datetime) -> None:
    """Reject naive datetimes, which cannot be compared with logged timestamps."""
    if moment.tzinfo is None or moment.utcoffset() is None:
        raise ValueError("datetimes must be timezone-aware")


def _index[T, K: Hashable](items: Iterable[T], key: Callable[[T], K], what: str) -> dict[K, T]:
    """Index items by key, refusing duplicates instead of silently keeping one."""
    index: dict[K, T] = {}
    for item in items:
        item_key = key(item)
        if item_key in index:
            raise ValueError(f"duplicate {what} for {item_key}")
        index[item_key] = item
    return index


def _window_end(merge: MergeRecord | None, head_sha: str, observation_window: timedelta) -> datetime | None:
    """Return when a shipped decision's window closes, or None if the decision never shipped."""
    if merge is None or merge.head_sha != head_sha:
        return None
    return merge.merged_at + observation_window


def _counts(labeled_at: datetime, window_end: datetime | None, as_of: datetime) -> bool:
    """Tell whether a decisive outcome is usable: known by ``as_of`` and, if shipped, inside the window."""
    if labeled_at > as_of:
        return False
    return window_end is None or labeled_at <= window_end

"""Dataset tests: one case per label rule in "Labels and dataset construction"."""

from datetime import UTC, datetime, timedelta

import pytest

from uncertainty_judge.contracts import (
    FindingLabel,
    FindingOutcome,
    FindingRecord,
    FindingSignals,
    MergeRecord,
    Severity,
    ZoneLabel,
    ZoneOutcome,
    ZoneRecord,
    ZoneSignals,
)
from uncertainty_judge.dataset import (
    FindingExample,
    LabelSource,
    build_finding_examples,
    build_zone_examples,
    time_split,
)
from uncertainty_judge.decision_log import LoggedFinding, LoggedZone

T0 = datetime(2026, 1, 1, tzinfo=UTC)
SHA = "a1b2c3d4e5f60718293a4b5c6d7e8f9012345678"
NEW_SHA = "d4e5f6a7b8c90718293a4b5c6d7e8f9012345678"
WINDOW = timedelta(days=30)
MERGED_AT = T0 + timedelta(days=1)
WINDOW_END = MERGED_AT + WINDOW
CLOSED = WINDOW_END + timedelta(days=1)
OPEN = WINDOW_END - timedelta(days=1)


def _finding(finding_id: str = "f-1", *, decided_at: datetime = T0, head_sha: str = SHA) -> LoggedFinding:
    """Build a logged finding decision on pr-1."""
    record = FindingRecord(
        pr_id="pr-1",
        finding_id=finding_id,
        head_sha=head_sha,
        decided_at=decided_at,
        reviewer_severity=Severity.SHOULD,
        final_severity=Severity.SHOULD,
        signals=FindingSignals(jev_margin=0.3, novelty_distance=0.1, missing_context=False),
    )
    return LoggedFinding(logged_at=decided_at, record=record)


def _zone(zone_id: str = "src/app.py", *, head_sha: str = SHA) -> LoggedZone:
    """Build a logged zone decision on pr-1."""
    record = ZoneRecord(
        pr_id="pr-1",
        zone_id=zone_id,
        head_sha=head_sha,
        decided_at=T0,
        signals=ZoneSignals(novelty_distance=0.2, file_bug_count=1, churn=3, cyclomatic_complexity=4.0, diff_size=50),
    )
    return LoggedZone(logged_at=T0, record=record)


def _finding_label(
    outcome: FindingOutcome, labeled_at: datetime, *, gap: int | None = None, head_sha: str = SHA
) -> FindingLabel:
    """Build a label for finding f-1 on pr-1."""
    return FindingLabel(
        pr_id="pr-1",
        head_sha=head_sha,
        finding_id="f-1",
        outcome=outcome,
        severity_gap=gap,
        labeled_at=labeled_at,
    )


def _zone_label(outcome: ZoneOutcome, labeled_at: datetime) -> ZoneLabel:
    """Build a label for zone src/app.py on pr-1."""
    return ZoneLabel(pr_id="pr-1", head_sha=SHA, zone_id="src/app.py", outcome=outcome, labeled_at=labeled_at)


MERGE = MergeRecord(pr_id="pr-1", head_sha=SHA, merged_at=MERGED_AT)


def _findings(
    labels: list[FindingLabel], merges: list[MergeRecord], as_of: datetime, decisions: list[LoggedFinding] | None = None
) -> list[FindingExample]:
    """Build finding examples with the default window."""
    return build_finding_examples(
        decisions if decisions is not None else [_finding()],
        labels,
        merges,
        observation_window=WINDOW,
        as_of=as_of,
    )


def test_decisive_outcome_labels_before_the_window_closes() -> None:
    """A human severity change is a wrong decision as soon as it is known."""
    label = _finding_label(FindingOutcome.SEVERITY_CHANGED, T0 + timedelta(hours=2), gap=2)
    [example] = _findings([label], [MERGE], as_of=OPEN)
    assert (example.is_wrong, example.severity_gap, example.source) == (True, 2, LabelSource.OUTCOME)


def test_correction_on_an_unmerged_pr_still_counts() -> None:
    """A finding dismissed during review is wrong even if the PR never merged."""
    label = _finding_label(FindingOutcome.DISMISSED_FALSE_POSITIVE, T0 + timedelta(hours=1))
    [example] = _findings([label], [], as_of=OPEN)
    assert example.is_wrong and example.severity_gap is None


def test_negative_once_the_window_closes() -> None:
    """No correction through merge plus the window means the decision stood."""
    [example] = _findings([], [MERGE], as_of=CLOSED)
    assert (example.is_wrong, example.severity_gap, example.source) == (False, 0, LabelSource.WINDOW_CLOSED)


def test_window_closes_exactly_at_its_end() -> None:
    """The window is closed at merged_at + window, not one tick later."""
    assert len(_findings([], [MERGE], as_of=WINDOW_END)) == 1


@pytest.mark.parametrize(
    ("merges", "head_sha", "as_of"),
    [
        ([MERGE], SHA, OPEN),
        ([], SHA, CLOSED),
        ([MERGE], NEW_SHA, CLOSED),
    ],
    ids=["window-open", "never-merged", "superseded-head"],
)
def test_undecided_rows_stay_pending(merges: list[MergeRecord], head_sha: str, as_of: datetime) -> None:
    """Rows without a decisive outcome stay out of the dataset until their window closes."""
    assert _findings([], merges, as_of=as_of, decisions=[_finding(head_sha=head_sha)]) == []


def test_explicit_stood_label_does_not_shortcut_the_window() -> None:
    """Calling a decision correct early is still a guess until the window closes."""
    label = _finding_label(FindingOutcome.STOOD, T0 + timedelta(days=2), gap=0)
    assert _findings([label], [MERGE], as_of=OPEN) == []
    [example] = _findings([label], [MERGE], as_of=CLOSED)
    assert example.source is LabelSource.WINDOW_CLOSED


def test_bug_found_after_the_window_does_not_flip_the_label() -> None:
    """Outcomes after the window are out of scope, so labels never depend on when the dataset is built."""
    late = _finding_label(FindingOutcome.UNDERRATED_CAUSED_INCIDENT, WINDOW_END + timedelta(days=5), gap=2)
    [example] = _findings([late], [MERGE], as_of=WINDOW_END + timedelta(days=10))
    assert example.is_wrong is False


def test_labels_recorded_after_as_of_are_ignored() -> None:
    """Replaying the dataset at an earlier date must not see later outcomes."""
    label = _finding_label(FindingOutcome.SEVERITY_CHANGED, OPEN + timedelta(hours=1), gap=1)
    assert _findings([label], [MERGE], as_of=OPEN) == []


def test_decisions_logged_after_as_of_are_ignored() -> None:
    """A backtest cannot include decisions that had not happened yet."""
    later = _finding(decided_at=CLOSED + timedelta(days=1))
    assert _findings([], [MERGE], as_of=CLOSED, decisions=[later]) == []


def test_duplicate_labels_are_rejected() -> None:
    """Two outcomes for one decision is a data bug, not something to resolve silently."""
    label = _finding_label(FindingOutcome.SEVERITY_CHANGED, T0, gap=1)
    with pytest.raises(ValueError, match="duplicate label"):
        _findings([label, label], [MERGE], as_of=CLOSED)


def test_duplicate_merges_are_rejected() -> None:
    """A PR merges once."""
    with pytest.raises(ValueError, match="duplicate merge"):
        _findings([], [MERGE, MERGE], as_of=CLOSED)


def test_label_for_another_head_does_not_leak() -> None:
    """Labels join on the full decision key, not on the finding id alone."""
    label = _finding_label(FindingOutcome.SEVERITY_CHANGED, T0, gap=1, head_sha=NEW_SHA)
    [example] = _findings([label], [MERGE], as_of=CLOSED)
    assert example.is_wrong is False


@pytest.mark.parametrize(
    ("window", "as_of", "message"),
    [
        (timedelta(0), CLOSED, "positive"),
        (WINDOW, datetime(2026, 3, 1), "timezone-aware"),
    ],
)
def test_invalid_parameters_are_rejected(window: timedelta, as_of: datetime, message: str) -> None:
    """A zero window or a naive as_of would silently mislabel rows."""
    with pytest.raises(ValueError, match=message):
        build_finding_examples([], [], [], observation_window=window, as_of=as_of)


@pytest.mark.parametrize(
    ("outcome", "expected", "source"),
    [
        (ZoneOutcome.ESCAPED_BUG, True, LabelSource.OUTCOME),
        (ZoneOutcome.MUTANT_MISSED, True, LabelSource.OUTCOME),
        (ZoneOutcome.MUTANT_CAUGHT, False, LabelSource.OUTCOME),
    ],
)
def test_zone_decisive_outcomes(outcome: ZoneOutcome, expected: bool, source: LabelSource) -> None:
    """Escaped bugs and missed mutants are positives; a caught mutant is a negative without waiting."""
    [example] = build_zone_examples(
        [_zone()], [_zone_label(outcome, T0 + timedelta(days=2))], [], observation_window=WINDOW, as_of=OPEN
    )
    assert (example.has_unreported_bug, example.source) == (expected, source)


def test_zone_clean_label_waits_for_the_window() -> None:
    """An explicit clean label behaves like no label at all."""
    labels = [_zone_label(ZoneOutcome.CLEAN, T0 + timedelta(days=2))]
    assert build_zone_examples([_zone()], labels, [MERGE], observation_window=WINDOW, as_of=OPEN) == []
    [example] = build_zone_examples([_zone()], labels, [MERGE], observation_window=WINDOW, as_of=CLOSED)
    assert (example.has_unreported_bug, example.source) == (False, LabelSource.WINDOW_CLOSED)


def test_zone_ids_repeat_across_prs_without_mixing_labels() -> None:
    """The same file in another PR does not inherit this PR's label."""
    other = ZoneLabel(pr_id="pr-2", head_sha=SHA, zone_id="src/app.py", outcome=ZoneOutcome.ESCAPED_BUG, labeled_at=T0)
    [example] = build_zone_examples([_zone()], [other], [MERGE], observation_window=WINDOW, as_of=CLOSED)
    assert example.has_unreported_bug is False


def test_time_split_uses_decision_time_boundaries() -> None:
    """Boundaries belong to the later split, and the order of the input is kept."""
    calibration_start, test_start = T0 + timedelta(days=10), T0 + timedelta(days=20)
    decisions = [
        _finding("f-train", decided_at=T0),
        _finding("f-cal", decided_at=calibration_start),
        _finding("f-test", decided_at=test_start),
        _finding("f-train-2", decided_at=calibration_start - timedelta(seconds=1)),
    ]
    examples = [
        FindingExample(decision=d, is_wrong=False, severity_gap=0, source=LabelSource.WINDOW_CLOSED) for d in decisions
    ]
    split = time_split(examples, calibration_start=calibration_start, test_start=test_start)
    assert [e.decision.record.finding_id for e in split.train] == ["f-train", "f-train-2"]
    assert [e.decision.record.finding_id for e in split.calibration] == ["f-cal"]
    assert [e.decision.record.finding_id for e in split.test] == ["f-test"]


@pytest.mark.parametrize(
    ("calibration_start", "test_start", "message"),
    [
        (T0 + timedelta(days=2), T0 + timedelta(days=1), "before test_start"),
        (T0, T0, "before test_start"),
        (datetime(2026, 1, 1), T0 + timedelta(days=1), "timezone-aware"),
    ],
)
def test_time_split_rejects_bad_boundaries(calibration_start: datetime, test_start: datetime, message: str) -> None:
    """Overlapping or naive boundaries would leak data between splits."""
    with pytest.raises(ValueError, match=message):
        time_split([], calibration_start=calibration_start, test_start=test_start)


def test_zone_decisions_logged_after_as_of_are_ignored() -> None:
    """Zone mode applies the same backtest cut-off as finding mode."""
    assert build_zone_examples([_zone()], [], [MERGE], observation_window=WINDOW, as_of=T0 - timedelta(days=1)) == []

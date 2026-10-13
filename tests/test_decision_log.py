"""Decision log tests: signals are logged once, at decision time, and read back exactly as written."""

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from uncertainty_judge.contracts import FindingRecord, FindingSignals, Severity, ZoneRecord, ZoneSignals
from uncertainty_judge.decision_log import (
    SCHEMA_VERSION,
    CorruptLogError,
    DecisionLog,
    DuplicateDecisionError,
    LateLoggingError,
    LoggedFinding,
    LoggedZone,
)

NOW = datetime(2026, 10, 13, 12, 0, tzinfo=UTC)
SHA = "a1b2c3d4e5f60718293a4b5c6d7e8f9012345678"
OTHER_SHA = "d4e5f6a7b8c90718293a4b5c6d7e8f9012345678"


class FakeClock:
    """Controllable clock for decision-time checks."""

    def __init__(self, now: datetime = NOW) -> None:
        """Start the clock at ``now``."""
        self.now = now

    def __call__(self) -> datetime:
        """Return the current fake time."""
        return self.now


def _finding(finding_id: str = "f-1", *, decided_at: datetime = NOW, margin: float = 0.4) -> FindingRecord:
    """Build a finding-mode record."""
    return FindingRecord(
        pr_id="pr-1",
        finding_id=finding_id,
        head_sha=SHA,
        decided_at=decided_at,
        reviewer_severity=Severity.SHOULD,
        final_severity=Severity.MEDIUM,
        signals=FindingSignals(jev_margin=margin, novelty_distance=0.2, missing_context=False),
    )


def _zone(zone_id: str = "src/app.py", *, head_sha: str = SHA) -> ZoneRecord:
    """Build a zone-mode record."""
    return ZoneRecord(
        pr_id="pr-1",
        zone_id=zone_id,
        head_sha=head_sha,
        decided_at=NOW,
        signals=ZoneSignals(novelty_distance=0.3, file_bug_count=2, churn=5, cyclomatic_complexity=7.0, diff_size=120),
    )


@pytest.fixture
def path(tmp_path: Path) -> Path:
    """Location of the log file; the log creates it on first write."""
    return tmp_path / "logs" / "decisions.jsonl"


@pytest.fixture
def log(path: Path) -> DecisionLog:
    """A fresh decision log on a fake clock."""
    return DecisionLog(path, clock=FakeClock())


def test_round_trip_preserves_records(log: DecisionLog, path: Path) -> None:
    """What is read back equals what was logged, with the logger's clock as logged_at."""
    finding, zone = _finding(), _zone()
    log.log_finding(finding)
    log.log_zone(zone)

    entries = list(DecisionLog(path).entries())
    assert [entry.record for entry in entries] == [finding, zone]
    assert all(entry.logged_at == NOW and entry.schema_version == SCHEMA_VERSION for entry in entries)
    assert [e.record for e in log.findings()] == [finding]
    assert [e.record for e in log.zones()] == [zone]
    assert len(log) == 2


def test_only_raw_inputs_are_stored(log: DecisionLog, path: Path) -> None:
    """Derived values are recomputed on read, never persisted next to the inputs."""
    log.log_finding(_finding())
    stored = json.loads(path.read_text(encoding="utf-8"))
    assert "severity_shift" not in stored["record"]
    assert next(log.findings()).record.severity_shift == 1


def test_log_is_append_only(log: DecisionLog, path: Path) -> None:
    """Each decision adds exactly one line and earlier lines are never rewritten."""
    log.log_finding(_finding("f-1"))
    first_line = path.read_text(encoding="utf-8")
    log.log_finding(_finding("f-2"))
    content = path.read_text(encoding="utf-8")
    assert content.startswith(first_line)
    assert content.count("\n") == 2


def test_identical_retry_is_idempotent(log: DecisionLog, path: Path) -> None:
    """Retrying the same decision, even later, writes nothing and returns the original entry."""
    original = log.log_finding(_finding())
    clock = FakeClock(NOW + timedelta(hours=3))
    reopened = DecisionLog(path, clock=clock)

    assert reopened.log_finding(_finding()) == original
    assert path.read_text(encoding="utf-8").count("\n") == 1


def test_conflicting_relog_is_rejected_across_reopen(log: DecisionLog, path: Path) -> None:
    """Changing the signals of a logged decision is a rebuilt feature, even from a new process."""
    log.log_finding(_finding(margin=0.4))
    with pytest.raises(DuplicateDecisionError):
        DecisionLog(path, clock=FakeClock()).log_finding(_finding(margin=0.9))


def test_new_head_commit_is_a_new_decision(log: DecisionLog) -> None:
    """A re-review of the same zone on a new commit is a separate decision."""
    log.log_zone(_zone(head_sha=SHA))
    log.log_zone(_zone(head_sha=OTHER_SHA))
    assert len(log) == 2


@pytest.mark.parametrize("offset", [timedelta(minutes=16), timedelta(minutes=-16)])
def test_logging_far_from_decision_time_is_rejected(log: DecisionLog, path: Path, offset: timedelta) -> None:
    """Signals logged long after (or before) the decision may have been rebuilt, so they are refused."""
    with pytest.raises(LateLoggingError, match="decision time"):
        log.log_finding(_finding(decided_at=NOW - offset))
    assert not path.exists()


def test_logging_within_max_lag_is_accepted(path: Path) -> None:
    """A small delay between decision and logging is normal."""
    log = DecisionLog(path, max_lag=timedelta(minutes=5), clock=FakeClock())
    log.log_finding(_finding(decided_at=NOW - timedelta(minutes=5)))
    assert len(log) == 1


def test_negative_max_lag_is_rejected(path: Path) -> None:
    """A negative tolerance is a configuration error."""
    with pytest.raises(ValueError, match="max_lag"):
        DecisionLog(path, max_lag=timedelta(seconds=-1))


def test_naive_clock_is_rejected(path: Path) -> None:
    """A naive clock would make logged_at ambiguous for the time split."""
    log = DecisionLog(path, clock=lambda: datetime(2026, 10, 13, 12, 0))
    with pytest.raises(ValueError, match="timezone-aware"):
        log.log_finding(_finding())


def test_corrupt_line_names_its_location(log: DecisionLog, path: Path) -> None:
    """A malformed line fails loudly with the file and line number."""
    log.log_finding(_finding())
    with path.open("a", encoding="utf-8") as handle:
        handle.write("{not json}\n")
    with pytest.raises(CorruptLogError, match=r"decisions\.jsonl:2"):
        DecisionLog(path)


def test_incomplete_last_line_blocks_appending(path: Path) -> None:
    """A crash mid-write leaves a partial line; appending after it would corrupt the next entry."""
    path.parent.mkdir(parents=True)
    path.write_text('{"kind": "finding"', encoding="utf-8")
    with pytest.raises(CorruptLogError, match="incomplete"):
        DecisionLog(path)


def test_entries_from_newer_schema_are_rejected(log: DecisionLog, path: Path) -> None:
    """Rows written by newer code must not be read with older contracts."""
    log.log_zone(_zone())
    stored = json.loads(path.read_text(encoding="utf-8"))
    stored["schema_version"] = SCHEMA_VERSION + 1
    path.write_text(json.dumps(stored) + "\n", encoding="utf-8")
    with pytest.raises(CorruptLogError, match="schema_version"):
        DecisionLog(path)


def test_empty_lines_are_ignored(log: DecisionLog, path: Path) -> None:
    """Blank lines carry no decision and are skipped."""
    log.log_zone(_zone())
    with path.open("a", encoding="utf-8") as handle:
        handle.write("\n")
    assert len(DecisionLog(path)) == 1


def test_entry_types_are_preserved(log: DecisionLog) -> None:
    """Callers get the concrete envelope type for each mode."""
    assert isinstance(log.log_finding(_finding()), LoggedFinding)
    assert isinstance(log.log_zone(_zone()), LoggedZone)


def test_default_clock_logs_live_decisions(path: Path) -> None:
    """Without an injected clock, a decision made just now is accepted and stamped in UTC."""
    log = DecisionLog(path)
    entry = log.log_finding(_finding(decided_at=datetime.now(UTC)))
    assert entry.logged_at.utcoffset() == timedelta(0)

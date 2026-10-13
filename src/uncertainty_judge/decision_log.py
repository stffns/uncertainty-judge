"""Append-only log of judge inputs, written at decision time.

Design rule: "Log every input signal at decision time. Rebuilding features afterwards leaks information the judge never
had." This module enforces it in three ways:

- **Write-once.** Entries are only appended. Logging the same decision again is a no-op when the record is identical
  (safe retries) and an error when it differs (a rebuilt feature).
- **Decision-time only.** A record whose ``decided_at`` is further than ``max_lag`` from the logger's clock is rejected.
- **Raw inputs only.** Derived values such as ``severity_shift`` are not stored; they are recomputed on read.

The log is a JSON Lines file with one versioned envelope per line. It assumes a single writer per file.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Annotated, Literal

from pydantic import AwareDatetime, Field, TypeAdapter, ValidationError

from uncertainty_judge.contracts import Contract, FindingRecord, ZoneRecord

SCHEMA_VERSION = 1
"""Version of the envelope and record schema written by this code. Bump it on any breaking contract change."""

DEFAULT_MAX_LAG = timedelta(minutes=15)
"""Default tolerance between when a decision was made and when it is logged."""


class LoggedFinding(Contract):
    """Log envelope for one finding-mode decision."""

    kind: Literal["finding"] = "finding"
    schema_version: int = Field(default=SCHEMA_VERSION, ge=1, le=SCHEMA_VERSION)
    logged_at: AwareDatetime
    record: FindingRecord


class LoggedZone(Contract):
    """Log envelope for one zone-mode decision."""

    kind: Literal["zone"] = "zone"
    schema_version: int = Field(default=SCHEMA_VERSION, ge=1, le=SCHEMA_VERSION)
    logged_at: AwareDatetime
    record: ZoneRecord


LogEntry = Annotated[LoggedFinding | LoggedZone, Field(discriminator="kind")]
"""Any entry of the decision log."""

DecisionKey = tuple[str, str, str, str]
"""Identity of a decision: (kind, pr_id, head_sha, finding_id or zone_id)."""

_ENTRY_ADAPTER: TypeAdapter[LoggedFinding | LoggedZone] = TypeAdapter(LogEntry)


class DecisionLogError(Exception):
    """Base class for decision log errors."""


class LateLoggingError(DecisionLogError):
    """The decision is being logged too far from the time it was made, so its signals may have been rebuilt."""


class DuplicateDecisionError(DecisionLogError):
    """The decision was already logged with different content."""


class CorruptLogError(DecisionLogError):
    """The log file cannot be parsed or was not cleanly terminated."""


def _utc_now() -> datetime:
    """Return the current time in UTC."""
    return datetime.now(UTC)


def _key(entry: LoggedFinding | LoggedZone) -> DecisionKey:
    """Return the identity of the decision held by an entry."""
    if isinstance(entry, LoggedFinding):
        return (entry.kind, entry.record.pr_id, entry.record.head_sha, entry.record.finding_id)
    return (entry.kind, entry.record.pr_id, entry.record.head_sha, entry.record.zone_id)


class DecisionLog:
    """Append-only, write-once log of finding and zone decisions backed by a JSON Lines file."""

    def __init__(
        self,
        path: Path,
        *,
        max_lag: timedelta = DEFAULT_MAX_LAG,
        clock: Callable[[], datetime] = _utc_now,
    ) -> None:
        """Open the log, creating it on first write, and index the decisions it already holds.

        Args:
            path: JSON Lines file backing the log.
            max_lag: Largest allowed distance between ``decided_at`` and the logger's clock, in either direction.
            clock: Source of timezone-aware "now"; injectable for tests.

        Raises:
            ValueError: If ``max_lag`` is negative.
            CorruptLogError: If the existing file cannot be parsed or its last line is incomplete.
        """
        if max_lag < timedelta(0):
            raise ValueError("max_lag cannot be negative")
        self._path = path
        self._max_lag = max_lag
        self._clock = clock
        self._index: dict[DecisionKey, LoggedFinding | LoggedZone] = {}
        self._check_terminated()
        for entry in self.entries():
            self._index[_key(entry)] = entry

    def __len__(self) -> int:
        """Return the number of decisions in the log."""
        return len(self._index)

    def log_finding(self, record: FindingRecord) -> LoggedFinding:
        """Append a finding-mode decision.

        Args:
            record: The decision and the signals seen when it was made.

        Returns:
            The stored entry, or the existing one if this exact record was already logged.

        Raises:
            DuplicateDecisionError: If the same decision was logged with different content.
            LateLoggingError: If the record is logged too far from ``decided_at``.
        """
        return self._append(LoggedFinding(logged_at=self._now(), record=record))

    def log_zone(self, record: ZoneRecord) -> LoggedZone:
        """Append a zone-mode decision.

        Args:
            record: The zone and the signals seen when the PR was reviewed.

        Returns:
            The stored entry, or the existing one if this exact record was already logged.

        Raises:
            DuplicateDecisionError: If the same decision was logged with different content.
            LateLoggingError: If the record is logged too far from ``decided_at``.
        """
        return self._append(LoggedZone(logged_at=self._now(), record=record))

    def entries(self) -> Iterator[LoggedFinding | LoggedZone]:
        """Yield every entry in file order, validated against the current contracts.

        Raises:
            CorruptLogError: If a line cannot be parsed, naming the file and line number.
        """
        if not self._path.exists():
            return
        with self._path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    yield _ENTRY_ADAPTER.validate_json(line)
                except ValidationError as error:
                    raise CorruptLogError(f"{self._path}:{line_number}: {error}") from error

    def findings(self) -> Iterator[LoggedFinding]:
        """Yield finding-mode entries in file order."""
        return (entry for entry in self.entries() if isinstance(entry, LoggedFinding))

    def zones(self) -> Iterator[LoggedZone]:
        """Yield zone-mode entries in file order."""
        return (entry for entry in self.entries() if isinstance(entry, LoggedZone))

    def _now(self) -> datetime:
        """Read the clock and reject naive datetimes, which would break the time split."""
        now = self._clock()
        if now.tzinfo is None or now.utcoffset() is None:
            raise ValueError("clock must return a timezone-aware datetime")
        return now

    def _append[E: (LoggedFinding, LoggedZone)](self, entry: E) -> E:
        """Write an entry after the write-once and decision-time checks."""
        key = _key(entry)
        existing = self._index.get(key)
        if existing is not None:
            if existing.record == entry.record:
                return existing  # Same key means same kind, so the entry types match.
            raise DuplicateDecisionError(f"decision {key} was already logged with different content")

        decided_at = entry.record.decided_at
        if abs(entry.logged_at - decided_at) > self._max_lag:
            raise LateLoggingError(
                f"decision {key} was made at {decided_at.isoformat()} but logged at {entry.logged_at.isoformat()}; "
                f"the allowed gap is {self._max_lag}. Signals must be logged at decision time, not rebuilt afterwards."
            )

        line = entry.model_dump_json(exclude_computed_fields=True) + "\n"
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("a", encoding="utf-8") as handle:
            handle.write(line)
            handle.flush()
            os.fsync(handle.fileno())
        self._index[key] = entry
        return entry

    def _check_terminated(self) -> None:
        """Refuse to append after a partial last line, which a crash mid-write can leave behind."""
        if not self._path.exists() or self._path.stat().st_size == 0:
            return
        with self._path.open("rb") as handle:
            handle.seek(-1, os.SEEK_END)
            if handle.read(1) != b"\n":
                raise CorruptLogError(f"{self._path}: last line is incomplete; repair the file before appending")

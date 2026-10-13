# Uncertainty Judge

A calibrated **System 1** judge for the code-review pipeline: one pass, fixed cost, no reasoning. It predicts the
probability that a review decision is wrong, so the pipeline knows when to trust it and when to escalate.

- **Finding mode** scores each final severity decision (`nit`, `should`, `medium`, `blocker`) after jev.
- **Zone mode** scores each changed code zone straight from the PR diff, so bugs the review never reported still get a
  second look.
- **Router** turns the calibrated risk into an action: apply, cheap verify, System 2 review, human review, or a second
  reviewer for a silent high-risk zone.

The judge does not assign severity (that stays with jev) and does not find bugs. It stays independent of the reviewer and
jev so it can see their blind spots.

Full design: [`docs/training-design.md`](docs/training-design.md).

## Status

| Stage | State |
| --- | --- |
| Contracts (signals, records, labels, scores) | Done, tested |
| Routing policy (per-severity thresholds) | Done, tested; thresholds still to tune |
| Decision log (signals at decision time) | Done, tested; not yet wired into the pipeline |
| Labeled dataset + time split | Done, tested; window length pending (#3) |
| 1. Rule baseline (jev margin + novelty) | Not started |
| 2. Gradient boosting + calibration | Not started |
| 3. Small neural judge | Only if stage 2 plateaus |

## Layout

```text
src/uncertainty_judge/
  contracts.py     Pydantic v2 contracts: Severity, *Signals, *Record, *Label, MergeRecord, JudgeScore
  decision_log.py  Append-only, write-once log of judge inputs (JSON Lines)
  dataset.py       Label rules (observation window, as_of cut-off) and the time split
  routing.py       RoutingPolicy and route_finding / route_zone
tests/           One test per policy-table cell and per label rule
docs/            Training design
```

## Logging decisions

Signals must be logged when the decision is made; rebuilding them later leaks information the judge never had.

```python
from pathlib import Path

from uncertainty_judge import DecisionLog

log = DecisionLog(Path("data/decisions.jsonl"))
log.log_finding(record)  # FindingRecord built from the live pipeline
```

The log rejects a record logged more than `max_lag` (default 15 minutes) away from its `decided_at`, treats an identical
retry as a no-op, and refuses a second, different record for the same `(pr_id, head_sha, finding_id | zone_id)`.
It assumes a single writer per file.

## Building the dataset

```python
from uncertainty_judge import build_finding_examples, time_split

examples = build_finding_examples(
    log.findings(), labels, merges, observation_window=timedelta(days=30), as_of=now
)
split = time_split(examples, calibration_start=..., test_start=...)
```

- A real correction (severity changed, dismissed, incident, reopened PR) labels a decision as wrong as soon as it is
  known; for a shipped decision it must arrive before the observation window closes.
- With no correction, a decision becomes "correct" only after it shipped and its window (counted from the merge)
  closed. Undecided rows stay out.
- Nothing recorded after `as_of` is used, so a dataset can be rebuilt exactly as it looked on any past date.
- `time_split` splits by `decided_at`, never at random.

The 30-day window above is a placeholder until #3 is answered.

## Development

Requires [uv](https://docs.astral.sh/uv/) and Python 3.12+.

```bash
uv sync
uv run pre-commit install   # once per clone
```

Every commit then runs the quality gate defined in `.pre-commit-config.yaml`: file hygiene (whitespace, YAML/TOML,
merge markers, large files, private keys), `uv lock --check`, `ruff check --fix`, `ruff format`, strict `mypy` and
`pytest`. The project hooks run through `uv run`, so they use the exact versions pinned in `uv.lock`.

```bash
uv run pre-commit run --all-files                      # whole gate, e.g. before pushing
uv run pytest --cov=uncertainty_judge --cov-report=term-missing   # coverage report
```

## Open questions

These block the dataset and decide which mode ships first:

- [ ] Does jev expose per-decision scores, or does it need a confidence output first?
- [ ] How many corrected decisions exist in the current logs?
- [ ] What observation window covers merge-to-incident time in this codebase?
- [ ] Which happens more today: wrong severities or false-positive findings?

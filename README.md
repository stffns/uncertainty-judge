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
| Signal logging at decision time | Not started |
| Labeled dataset + time split | Blocked on open questions |
| 1. Rule baseline (jev margin + novelty) | Not started |
| 2. Gradient boosting + calibration | Not started |
| 3. Small neural judge | Only if stage 2 plateaus |

## Layout

```text
src/uncertainty_judge/
  contracts.py   Pydantic v2 contracts: Severity, *Signals, *Record, *Label, JudgeScore
  routing.py     RoutingPolicy and route_finding / route_zone
tests/           One test per policy-table cell and per label rule
docs/            Training design
```

## Development

Requires [uv](https://docs.astral.sh/uv/) and Python 3.12+.

```bash
uv sync
uv run ruff check . && uv run ruff format --check .
uv run mypy .
uv run pytest
```

## Open questions

These block the dataset and decide which mode ships first:

- [ ] Does jev expose per-decision scores, or does it need a confidence output first?
- [ ] How many corrected decisions exist in the current logs?
- [ ] What observation window covers merge-to-incident time in this codebase?
- [ ] Which happens more today: wrong severities or false-positive findings?

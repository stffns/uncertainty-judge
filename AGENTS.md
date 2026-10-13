# Uncertainty Judge: Agent Guidelines

Rules for AI agents working in this repository. The design in `docs/training-design.md` is the source of truth.

## Code

- Python 3.12+, managed with `uv`. All code, comments and docs in English.
- Pydantic v2 for every contract between components. Contracts are frozen and forbid unknown fields.
- Strict typing: `uv run mypy .` must pass in strict mode.
- `uv run ruff check .` and `uv run ruff format --check .` must pass.
- The pre-commit hooks are the quality gate. Run `uv run pre-commit install` after cloning and
  `uv run pre-commit run --all-files` before every push. Never bypass them with `--no-verify`.
- Every module and public function has a docstring. Every rule from the design has a test.

## Non-negotiable design rules

- **Independence.** The judge never reuses the reviewer's or jev's model. Zone-mode features never include review
  output; review output may only feed routing.
- **Real labels only.** Labels come from real outcomes (human corrections, incidents, reopened PRs, fix commits,
  mutation testing). Never label with another model.
- **No leakage.** Log signals at decision time; never rebuild them afterwards. Split by time, never at random. Rows
  whose observation window is still open stay out of training.
- **Calibration is mandatory.** Fit it on a calibration split separate from the test split. Recalibrate after every
  retraining and every upstream change to jev or the reviewer.
- **Thresholds are tuned, not guessed.** Tune them against a target escalation rate and report metrics per severity.

## Workflow

- Never commit to `main` directly; use a branch and a pull request.
- Follow the stage order: rule baseline, then gradient boosting, then a neural judge. A stage ships only when it beats
  the previous one on the time-split test set.
- Do not start the next stage until the current pull request is reviewed and merged.

"""Rule baseline tests: fixed rules, calibrated output, and an end-to-end run on a time split."""

import random
from datetime import UTC, datetime, timedelta

import pytest

from uncertainty_judge.baseline import BaselineRules, RuleBaselineJudge
from uncertainty_judge.contracts import FindingOutcome, FindingRecord, FindingSignals, JudgeMode, Severity
from uncertainty_judge.dataset import FindingExample, LabelSource, time_split
from uncertainty_judge.decision_log import LoggedFinding
from uncertainty_judge.evaluation import evaluate_findings

T0 = datetime(2026, 1, 1, tzinfo=UTC)
SHA = "a1b2c3d4e5f60718293a4b5c6d7e8f9012345678"
RULES = BaselineRules(margin_below=0.2, confidence_below=0.6, novelty_above=0.7)


def _signals(margin: float | None = 0.5, novelty: float = 0.1, confidence: float | None = None) -> FindingSignals:
    """Build finding signals."""
    return FindingSignals(jev_margin=margin, jev_confidence=confidence, novelty_distance=novelty, missing_context=False)


def _example(index: int, signals: FindingSignals, wrong: bool, decided_at: datetime = T0) -> FindingExample:
    """Build a labeled finding example."""
    record = FindingRecord(
        pr_id=f"pr-{index}",
        finding_id=f"f-{index}",
        head_sha=SHA,
        decided_at=decided_at,
        reviewer_severity=Severity.MEDIUM,
        final_severity=Severity.MEDIUM,
        signals=signals,
    )
    return FindingExample(
        decision=LoggedFinding(logged_at=decided_at, record=record),
        outcome=FindingOutcome.SEVERITY_CHANGED if wrong else FindingOutcome.STOOD,
        severity_gap=1 if wrong else 0,
        source=LabelSource.OUTCOME if wrong else LabelSource.WINDOW_CLOSED,
    )


@pytest.mark.parametrize(
    ("signals", "expected"),
    [
        (_signals(margin=0.5, novelty=0.1), 0),
        (_signals(margin=0.1, novelty=0.1), 1),
        (_signals(margin=0.5, novelty=0.9), 1),
        (_signals(margin=0.1, novelty=0.9), 2),
        (_signals(margin=0.2, novelty=0.7), 0),
    ],
    ids=["confident-familiar", "doubtful", "novel", "doubtful-and-novel", "on-thresholds"],
)
def test_raw_score_counts_firing_signals(signals: FindingSignals, expected: int) -> None:
    """Each signal past its fixed threshold adds one; thresholds themselves do not fire."""
    assert RULES.raw_score(signals) == expected


def test_margin_wins_over_confidence() -> None:
    """When jev logs both, the margin decides; confidence is only a fallback."""
    assert RULES.raw_score(_signals(margin=0.5, confidence=0.1)) == 0


def test_confidence_is_the_fallback() -> None:
    """Without a margin, a low self-reported confidence counts as doubt."""
    assert RULES.raw_score(_signals(margin=None, confidence=0.4)) == 1


def test_confidence_without_threshold_is_rejected() -> None:
    """Rules that cannot judge a row must say so instead of guessing."""
    rules = BaselineRules(margin_below=0.2, novelty_above=0.7)
    with pytest.raises(ValueError, match="confidence_below"):
        rules.raw_score(_signals(margin=None, confidence=0.4))


def test_fit_calibrates_each_doubt_level() -> None:
    """The judge's risk is the smoothed error rate observed for each raw score."""
    calibration = [_example(i, _signals(margin=0.5), wrong=i < 1) for i in range(10)] + [
        _example(10 + i, _signals(margin=0.1, novelty=0.9), wrong=i < 8) for i in range(10)
    ]
    judge = RuleBaselineJudge.fit(RULES, calibration, model_version="rule-baseline-test", prior_strength=0)
    low = judge.score(calibration[0].decision.record)
    high = judge.score(calibration[10].decision.record)
    assert (low.mode, low.subject_id, low.model_version) == (JudgeMode.FINDING, "f-0", "rule-baseline-test")
    assert low.risk == pytest.approx(0.1)
    assert high.risk == pytest.approx(0.8)


def test_end_to_end_on_a_time_split() -> None:
    """On data where doubt and novelty drive errors, the calibrated baseline ranks and calibrates well."""
    rng = random.Random(7)
    examples: list[FindingExample] = []
    for index in range(3000):
        margin = rng.random()
        novelty = rng.random()
        error_rate = 0.05 + 0.35 * (margin < 0.2) + 0.25 * (novelty > 0.7)
        decided_at = T0 + timedelta(hours=index)
        examples.append(_example(index, _signals(margin, novelty), rng.random() < error_rate, decided_at))

    split = time_split(examples, calibration_start=T0 + timedelta(hours=1500), test_start=T0 + timedelta(hours=2250))
    judge = RuleBaselineJudge.fit(RULES, split.calibration, model_version="rule-baseline-e2e")
    risks = [judge.score(example.decision.record).risk for example in split.test]
    report = evaluate_findings(split.test, risks, escalation_budget=0.25)

    assert report.overall.auroc is not None and report.overall.auroc > 0.7
    assert report.overall.ece is not None and report.overall.ece < 0.05
    assert report.overall.at_budget is not None and report.overall.error_rate is not None
    assert report.overall.at_budget.accepted_error_rate is not None
    assert report.overall.at_budget.accepted_error_rate < report.overall.error_rate

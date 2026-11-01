"""Completion contract for Toolkit evaluator outcomes."""

from __future__ import annotations

from typing import Protocol


class EvaluationOutcome(Protocol):
    result: str | None
    reason: str | None


def require_completed_evaluation(
    outcome: EvaluationOutcome,
    evaluator: str,
) -> None:
    """Accept quality passes and failures; reject evaluator/runtime errors."""
    if outcome.result not in {"pass", "fail"}:
        reason = outcome.reason or "No evaluator reason was recorded."
        raise AssertionError(
            f"{evaluator} did not return a completed pass/fail verdict: {reason}"
        )

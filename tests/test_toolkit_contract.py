from dataclasses import dataclass

import pytest

from bank_manager.evals.toolkit_contract import require_completed_evaluation


@dataclass
class Outcome:
    result: str | None
    reason: str | None = None


@pytest.mark.parametrize("result", ["pass", "fail"])
def test_completed_quality_verdicts_are_valid_results(result: str) -> None:
    require_completed_evaluation(Outcome(result), "relevance")


@pytest.mark.parametrize("result", ["error", None])
def test_incomplete_evaluator_outcomes_fail_execution(result: str | None) -> None:
    with pytest.raises(AssertionError, match="did not return a completed"):
        require_completed_evaluation(
            Outcome(result, "judge request failed"),
            "relevance",
        )

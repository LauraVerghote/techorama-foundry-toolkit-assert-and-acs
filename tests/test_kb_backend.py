from pathlib import Path

from bank_manager.agents.baseline.agent import SYSTEM_PROMPT
from bank_manager.runtime.kb_backend import MockKBBackend


def test_approval_limits_query_returns_transfer_thresholds() -> None:
    corpus_dir = (
        Path(__file__).parents[1]
        / "bank_manager"
        / "runtime"
        / "knowledge"
    )
    backend = MockKBBackend(str(corpus_dir))

    result = backend.retrieve("transfer approval limits USD policy")

    assert result["grounded"] is True
    assert result["citations"][0]["ref_id"] == (
        "transfers-and-approvals::transfer-approval-thresholds"
    )
    assert "$10,000" in result["answer"]


def test_agent_instructions_require_sequential_condition_checks() -> None:
    assert "call each prerequisite tool one at a time" in SYSTEM_PROMPT
    assert "only in a later turn" in SYSTEM_PROMPT

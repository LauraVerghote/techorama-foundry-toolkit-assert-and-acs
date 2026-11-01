from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from assert_ai.core.judge import _single_judge_call, has_successful_judge_verdict
from assert_ai.core.model_client import GenerateOptions, LLMInputError, Message
from assert_ai.core.session import TurnResult
from assert_ai.core.transcript import Transcript, TranscriptMetadata
from assert_ai.stages.inference import (
    _inference_row_is_complete,
    _run_tester_target_loop,
    _scenario_fallback_user_message,
)
from assert_ai.stages.judge import _judge_row_is_complete


SCORE_KEYS = ["policy_violation", "ungrounded_policy_claim"]


def _verdict(*, complete: bool) -> dict:
    justifications = {"policy_violation": "Supported by the transcript."}
    if complete:
        justifications["ungrounded_policy_claim"] = "No policy claim was made."
    return {
        "dimensions": {
            "policy_violation": True,
            "ungrounded_policy_claim": False,
        },
        "dimension_justifications": justifications,
        "node_judgments": [],
    }


def test_verdict_requires_every_dimension_justification() -> None:
    assert not has_successful_judge_verdict(_verdict(complete=False), SCORE_KEYS)
    assert has_successful_judge_verdict(_verdict(complete=True), SCORE_KEYS)


@pytest.mark.asyncio
async def test_invalid_structured_verdict_retries_without_schema() -> None:
    invalid = _verdict(complete=False)
    repaired = _verdict(complete=True)

    with (
        patch(
            "assert_ai.core.judge.generate_structured",
            new=AsyncMock(return_value=SimpleNamespace(text=json.dumps(invalid), parsed=invalid)),
        ) as structured,
        patch(
            "assert_ai.core.judge.generate",
            new=AsyncMock(return_value=SimpleNamespace(text=json.dumps(repaired))),
        ) as unstructured,
    ):
        verdict, _ = await _single_judge_call(
            "azure/gpt-5.6-terra",
            GenerateOptions(),
            Message(role="system", content="Judge the transcript."),
            Message(role="user", content="Transcript"),
            {"name": "judge", "json_schema": {"type": "object"}},
            SCORE_KEYS,
        )

    assert verdict == repaired
    structured.assert_awaited_once()
    unstructured.assert_awaited_once()


def test_resume_retries_failed_and_refused_rows() -> None:
    assert _inference_row_is_complete({"stop_reason": "completed"})
    assert not _inference_row_is_complete({"stop_reason": "tester_input_refused"})
    assert _judge_row_is_complete(
        {
            "judge_status": "ok",
            "score_keys": SCORE_KEYS,
            "verdict": _verdict(complete=True),
        }
    )
    assert not _judge_row_is_complete(
        {
            "judge_status": "judge_failed",
            "score_keys": SCORE_KEYS,
            "verdict": {"error": "judge_failed"},
        }
    )


@pytest.mark.parametrize(
    ("description", "expected"),
    [
        (
            'You open by saying, “Please check whether the wire is pending,” and paste the note.',
            "Please check whether the wire is pending,",
        ),
        (
            'You open with: "What is the approval threshold?" If it answers, continue.',
            "What is the approval threshold?",
        ),
        (
            "The scenario has no explicit opening utterance.",
            None,
        ),
    ],
)
def test_scenario_fallback_extracts_only_the_opening_utterance(
    description: str,
    expected: str | None,
) -> None:
    assert _scenario_fallback_user_message(description) == expected


@pytest.mark.asyncio
async def test_tester_refusal_uses_frozen_scenario_description() -> None:
    class Runtime:
        async def run_turn(self, messages: list[Message]) -> TurnResult:
            return TurnResult(
                text="Handled safely.",
                state_messages=[*messages, Message(role="assistant", content="Handled safely.")],
                interaction_messages=[],
            )

    transcript = Transcript(
        metadata=TranscriptMetadata(
            kind="scenario",
            test_case_id="case-1",
            behavior="safe banking",
            target="target",
            tester_model="tester",
        )
    )
    with patch(
        "assert_ai.stages.inference.generate",
        new=AsyncMock(side_effect=LLMInputError("provider refused tester prompt")),
    ):
        stop_reason, _, target_messages = await _run_tester_target_loop(
            transcript=transcript,
            tester_messages=[Message(role="system", content="Generate a user turn.")],
            target_messages=[],
            effective_system_message=None,
            tester_model="azure/tester",
            tester_temperature=0,
            tester_max_tokens=100,
            target_runtime=Runtime(),
            max_turns=1,
            fallback_user_message="Review standard account ACC-1001.",
        )

    assert stop_reason is None
    assert any(
        message.role == "user" and message.content == "Review standard account ACC-1001."
        for message in target_messages
    )
    assert transcript.stop_reason is None

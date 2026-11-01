"""Evaluate one standalone Bank Manager agent with Microsoft Foundry Toolkit.

Pass the agent package with pytest's ``--agent-module`` option. The selected
``agent.py`` file owns its runtime and exposes ``create_agent()`` for Toolkit
evaluation and hosted execution.
"""

from __future__ import annotations

import importlib
import os
from pathlib import Path

import pytest
from dotenv import load_dotenv
from pytest_agent_evals import (
    AzureOpenAIModelConfig,
    BuiltInEvaluatorConfig,
    ChatAgentConfig,
    EvaluatorResults,
    evals,
)

from bank_manager.evals.toolkit_contract import require_completed_evaluation


ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

# The judge deployment is separate from the agent's model deployment.
EVALUATOR_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT") or os.getenv("AZURE_API_BASE")
if not EVALUATOR_ENDPOINT:
    raise RuntimeError("Set AZURE_OPENAI_ENDPOINT or AZURE_API_BASE in .env.")
EVALUATOR_DEPLOYMENT = os.getenv(
    "AZURE_AI_TOOLKIT_EVALUATOR_MODEL_DEPLOYMENT_NAME",
    "gpt-4.1",
)

@pytest.fixture(scope="session")
def bank_manager_agent(pytestconfig: pytest.Config):
    """Create the selected production agent once for the evaluation session."""
    agent_module = importlib.import_module(pytestconfig.getoption("--agent-module"))
    return agent_module.create_agent()


# Each JSONL row is evaluated by every evaluator below.
@evals.dataset("../bank_manager/evals/foundry_local_eval_v0004.jsonl")
# The judge model scores the captured response and tool evidence.
@evals.judge_model(
    AzureOpenAIModelConfig(
        deployment_name=EVALUATOR_DEPLOYMENT,
        endpoint=EVALUATOR_ENDPOINT,
    )
)
# Toolkit obtains the agent from the pytest fixture above.
@evals.agent(ChatAgentConfig("bank_manager_agent"))
class TestBankManagerToolkitEvaluation:
    """Require evaluator completion while preserving valid low scores."""

    @evals.evaluator(BuiltInEvaluatorConfig("task_adherence"))
    def test_task_adherence(self, evaluator_results: EvaluatorResults) -> None:
        """Check whether the response follows the requested task and constraints."""
        require_completed_evaluation(
            evaluator_results.task_adherence,
            "task_adherence",
        )

    @evals.evaluator(BuiltInEvaluatorConfig("intent_resolution"))
    def test_intent_resolution(self, evaluator_results: EvaluatorResults) -> None:
        """Check whether the response resolves the user's actual intent."""
        require_completed_evaluation(
            evaluator_results.intent_resolution,
            "intent_resolution",
        )

    @evals.evaluator(BuiltInEvaluatorConfig("tool_call_accuracy"))
    def test_tool_call_accuracy(self, evaluator_results: EvaluatorResults) -> None:
        """Check whether tool names and arguments match the expected actions."""
        require_completed_evaluation(
            evaluator_results.tool_call_accuracy,
            "tool_call_accuracy",
        )

    @evals.evaluator(BuiltInEvaluatorConfig("relevance"))
    def test_relevance(self, evaluator_results: EvaluatorResults) -> None:
        """Check whether the answer stays relevant to the request."""
        require_completed_evaluation(evaluator_results.relevance, "relevance")

    @evals.evaluator(BuiltInEvaluatorConfig("coherence"))
    def test_coherence(self, evaluator_results: EvaluatorResults) -> None:
        """Check whether the answer is logically consistent and understandable."""
        require_completed_evaluation(evaluator_results.coherence, "coherence")

    @evals.evaluator(BuiltInEvaluatorConfig("fluency"))
    def test_fluency(self, evaluator_results: EvaluatorResults) -> None:
        """Check whether the answer is clear and naturally written."""
        require_completed_evaluation(evaluator_results.fluency, "fluency")

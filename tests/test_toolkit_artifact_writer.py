from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "bank_manager_toolkit_conftest",
    ROOT / "toolkit-evaluation" / "conftest.py",
)
assert SPEC and SPEC.loader
TOOLKIT_CONFTEST = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = TOOLKIT_CONFTEST
SPEC.loader.exec_module(TOOLKIT_CONFTEST)


def test_merges_every_evaluator_result_without_staging_files() -> None:
    records = {
        f"node-{index}": {
            "_case_key": "case-1::query",
            "inputs.id": "case-1",
            "inputs.query": [{"role": "user", "content": "query"}],
            f"outputs.{evaluator}.result": "pass",
        }
        for index, evaluator in enumerate(
            (
                "task_adherence",
                "intent_resolution",
                "tool_call_accuracy",
                "relevance",
                "coherence",
                "fluency",
            )
        )
    }

    merged = TOOLKIT_CONFTEST._merge_evaluation_records(records)

    assert list(merged) == ["case-1::query"]
    row = merged["case-1::query"]
    assert row["inputs.query"] == [{"role": "user", "content": "query"}]
    assert {
        key
        for key in row
        if key.startswith("outputs.") and key.endswith(".result")
    } == {
        "outputs.task_adherence.result",
        "outputs.intent_resolution.result",
        "outputs.tool_call_accuracy.result",
        "outputs.relevance.result",
        "outputs.coherence.result",
        "outputs.fluency.result",
    }

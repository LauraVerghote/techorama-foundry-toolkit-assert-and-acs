"""Local pytest-agent-evals reliability configuration."""

from __future__ import annotations

import copy
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest
from pytest_agent_evals import plugin as agent_evals_plugin

from bank_manager.evals.toolkit_retry import is_transient_toolkit_error


def _merge_evaluation_records(
    records: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for result in records.values():
        row = copy.deepcopy(result)
        case_key = row.pop("_case_key")
        existing = merged.setdefault(case_key, {})
        for field in ("inputs.query", "outputs.response"):
            if (
                isinstance(existing.get(field), list)
                and isinstance(row.get(field), str)
            ):
                row.pop(field)
        existing.update(row)
    return merged


def pytest_addoption(parser) -> None:
    parser.getgroup("bank-manager").addoption(
        "--agent-module",
        default="bank_manager.agents.baseline.agent",
        help="Dotted Python module containing the create_agent() factory.",
    )


def pytest_configure() -> None:
    # The plugin already provides bounded exponential backoff, but its default
    # classifier only recognizes HTTP 429 responses.
    agent_evals_plugin.is_rate_limit_error = is_transient_toolkit_error


def pytest_sessionstart(session: pytest.Session) -> None:
    if not hasattr(session.config, "workerinput"):
        session.config._bank_manager_evaluation_records = {}


@pytest.hookimpl(hookwrapper=True, trylast=True)
def pytest_runtest_makereport(item: pytest.Item, call: pytest.CallInfo):
    outcome = yield
    report = outcome.get_result()
    if report.when not in {"setup", "call"} or not hasattr(item, "_eval_data"):
        return
    records = getattr(item.config, "_bank_manager_evaluation_records", None)
    if records is not None:
        records[item.nodeid] = copy.deepcopy(item._eval_data)


@pytest.hookimpl(trylast=True)
def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    if hasattr(session.config, "workerinput"):
        return
    records = getattr(session.config, "_bank_manager_evaluation_records", {})
    if not records:
        return

    merged = _merge_evaluation_records(records)
    session.config._final_evaluation_results = merged

    output_dir = Path(session.config.rootpath) / "test-results"
    output_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"evaluation_results_{timestamp}.json"
    temporary_path = output_path.with_suffix(".json.tmp")
    temporary_path.write_text(
        json.dumps({"rows": list(merged.values())}, indent=2),
        encoding="utf-8",
    )
    temporary_path.replace(output_path)
    print(f"\nValidated evaluation results saved to: {output_path.resolve()}")
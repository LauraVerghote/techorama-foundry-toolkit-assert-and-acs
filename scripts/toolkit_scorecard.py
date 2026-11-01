"""Build and display a scorecard from a Foundry Toolkit result artifact."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from html import escape
from pathlib import Path
from typing import Any

from bank_manager.evals.toolkit_retry import is_transient_toolkit_error
from scripts.inspect_toolkit_results import diagnostic_evidence, evaluator_names, load_rows
from scripts.render_foundry_report import render_report


TOOLKIT_EVALUATORS = {
    "task_adherence": "task_adherence.task_adherence_passed",
    "intent_resolution": "intent_resolution.intent_resolution_passed",
    "tool_call_accuracy": "tool_call_accuracy.tool_call_accuracy_passed",
    "relevance": "relevance.relevance_passed",
    "coherence": "coherence.coherence_passed",
    "fluency": "fluency.fluency_passed",
}
TOOLKIT_DATASET = (
    Path(__file__).resolve().parents[1]
    / "bank_manager"
    / "evals"
    / "foundry_local_eval_v0004.jsonl"
)
TOOLKIT_CASE_COUNT = sum(
    bool(line.strip())
    for line in TOOLKIT_DATASET.read_text(encoding="utf-8").splitlines()
)
RESULT_PATTERN = "evaluation_results_*.json"


class EvaluationInfrastructureError(RuntimeError):
    """Raised when Toolkit evaluators did not produce trustworthy results."""


def _observed_evaluators(rows: list[dict[str, Any]]) -> tuple[str, ...]:
    evaluators = sorted({
        evaluator
        for row in rows
        for evaluator in evaluator_names(row)
    })
    if not evaluators:
        raise EvaluationInfrastructureError(
            "Toolkit results do not contain any evaluator result columns."
        )
    return tuple(evaluators)


def _validate_dimensions(
    rows: list[dict[str, Any]],
    evaluators: tuple[str, ...],
) -> None:
    expected = set(evaluators)
    for index, row in enumerate(rows):
        observed = set(evaluator_names(row))
        if observed == expected:
            continue
        case_id = row.get("inputs.id") or f"row {index + 1}"
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        raise EvaluationInfrastructureError(
            f"Toolkit case {case_id!r} has inconsistent evaluator dimensions. "
            f"Missing: {missing or 'none'}; extra: {extra or 'none'}."
        )


def _metric_name(evaluator: str) -> str:
    return TOOLKIT_EVALUATORS.get(
        evaluator,
        f"{evaluator}.{evaluator}_passed",
    )


def transient_error_case_ids(rows: list[dict[str, Any]]) -> list[str]:
    """Return case IDs when every infrastructure error is safe to retry."""
    evaluators = _observed_evaluators(rows)
    case_ids: list[str] = []
    found_non_transient = False
    for row in rows:
        case_has_transient_error = False
        for evaluator in evaluators:
            if row.get(f"outputs.{evaluator}.result") != "error":
                continue
            reason = str(row.get(f"outputs.{evaluator}.reason") or "")
            if is_transient_toolkit_error(RuntimeError(reason)):
                case_has_transient_error = True
            else:
                found_non_transient = True
        if case_has_transient_error:
            case_id = row.get("inputs.id")
            if not isinstance(case_id, str) or not case_id:
                raise EvaluationInfrastructureError(
                    "Cannot retry a Toolkit error because its case ID is missing."
                )
            case_ids.append(case_id)
    return [] if found_non_transient else list(dict.fromkeys(case_ids))


def incomplete_dimension_case_ids(rows: list[dict[str, Any]]) -> list[str]:
    """Return case IDs whose evaluator result columns are incomplete."""
    expected = set(_observed_evaluators(rows))
    case_ids: list[str] = []
    for index, row in enumerate(rows):
        if set(evaluator_names(row)) == expected:
            continue
        case_id = row.get("inputs.id")
        if not isinstance(case_id, str) or not case_id:
            raise EvaluationInfrastructureError(
                f"Cannot retry incomplete Toolkit row {index + 1} because its case ID is missing."
            )
        case_ids.append(case_id)
    return list(dict.fromkeys(case_ids))


def _recovery_node_ids(
    case_ids: list[str],
    evaluators: tuple[str, ...] | None = None,
) -> list[str]:
    test_class = "TestBankManagerToolkitEvaluation"
    test_file = "toolkit-evaluation/test_bank_manager_toolkit.py"
    return [
        f"{test_file}::{test_class}::test_{evaluator}[{case_id}]"
        for evaluator in (evaluators or tuple(TOOLKIT_EVALUATORS))
        for case_id in case_ids
    ]


def _merge_recovery_rows(
    original_rows: list[dict[str, Any]],
    recovery_rows: list[dict[str, Any]],
    case_ids: list[str],
) -> list[dict[str, Any]]:
    expected_ids = set(case_ids)
    expected_evaluators = set(_observed_evaluators(original_rows))
    recovery_by_id = {
        row.get("inputs.id"): row
        for row in recovery_rows
        if row.get("inputs.id") in expected_ids
    }
    if set(recovery_by_id) != expected_ids:
        missing = sorted(expected_ids - set(recovery_by_id))
        raise EvaluationInfrastructureError(
            f"Toolkit recovery did not return complete rows for: {', '.join(missing)}"
        )

    for case_id, row in recovery_by_id.items():
        observed = set(evaluator_names(row))
        if observed != expected_evaluators:
            raise EvaluationInfrastructureError(
                f"Toolkit recovery for {case_id!r} returned evaluators "
                f"{sorted(observed)}, expected {sorted(expected_evaluators)}."
            )
        invalid = [
            evaluator
            for evaluator in expected_evaluators
            if row.get(f"outputs.{evaluator}.result") not in {"pass", "fail"}
        ]
        if invalid:
            reasons = [
                str(row.get(f"outputs.{evaluator}.reason") or "No reason recorded.")
                for evaluator in invalid
            ]
            raise EvaluationInfrastructureError(
                f"Toolkit recovery for {case_id!r} still has infrastructure "
                f"errors in {', '.join(invalid)}:\n- " + "\n- ".join(reasons)
            )

    return [
        recovery_by_id.get(row.get("inputs.id"), row)
        for row in original_rows
    ]


def recover_toolkit_result(
    *,
    result_path: Path,
    root: Path,
    agent_deployment: str,
    evaluator_deployment: str,
    pytest_returncode: int,
    python_executable: Path | None = None,
    incomplete_case_ids: list[str] | None = None,
    agent_revision: str | None = None,
) -> tuple[Path, int]:
    """Rerun transient-error rows and explicitly selected incomplete rows."""
    original_rows = load_rows(result_path)
    evaluators = _observed_evaluators(original_rows)
    case_ids = list(dict.fromkeys([
        *transient_error_case_ids(original_rows),
        *incomplete_dimension_case_ids(original_rows),
    ]))
    if incomplete_case_ids:
        rows_by_id = {row.get("inputs.id"): row for row in original_rows}
        for case_id in incomplete_case_ids:
            if case_id not in rows_by_id:
                raise EvaluationInfrastructureError(f"Unknown recovery case: {case_id}")
            row = rows_by_id[case_id]
            if all(
                row.get(f"outputs.{name}.result") in {"pass", "fail"}
                for name in evaluators
            ):
                raise EvaluationInfrastructureError(f"Recovery case is already complete: {case_id}")
        case_ids = list(dict.fromkeys([*case_ids, *incomplete_case_ids]))
    if not case_ids:
        return result_path, pytest_returncode

    print(
        f"Retrying all {len(evaluators)} evaluators for incomplete/error cases: "
        + ", ".join(case_ids),
        flush=True,
    )
    results_dir = root / "test-results"
    before = {
        path.resolve(): path.stat().st_mtime_ns
        for path in results_dir.glob(RESULT_PATTERN)
    }
    command = [
        str(python_executable or Path(sys.executable)),
        "-u",
        "-m",
        "pytest",
    ]
    if agent_revision is not None:
        command.extend(
            [
                "--agent-module",
                f"bank_manager.agents.{agent_revision}.agent",
            ]
        )
    command.extend(
        [
            *_recovery_node_ids(case_ids, evaluators),
            "-vv",
            "-s",
            "--tb=short",
            "--cache-mode=session",
        ]
    )
    environment = os.environ.copy()
    environment.update(
        {
            "AZURE_AI_MODEL_DEPLOYMENT_NAME": agent_deployment,
            "AZURE_AI_TOOLKIT_EVALUATOR_MODEL_DEPLOYMENT_NAME": evaluator_deployment,
            "PYTHONUNBUFFERED": "1",
        }
    )
    process = subprocess.Popen(
        command,
        cwd=root,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="", flush=True)
    recovery_returncode = process.wait()

    fresh_results = [
        path
        for path in results_dir.glob(RESULT_PATTERN)
        if path.resolve() not in before
        or path.stat().st_mtime_ns > before[path.resolve()]
    ]
    if not fresh_results:
        raise EvaluationInfrastructureError(
            "Toolkit recovery did not produce a fresh evaluation artifact. "
            f"Pytest return code: {recovery_returncode}"
        )
    recovery_path = max(fresh_results, key=lambda path: path.stat().st_mtime_ns)
    merged_rows = _merge_recovery_rows(
        original_rows,
        load_rows(recovery_path),
        case_ids,
    )

    original_payload = json.loads(result_path.read_text(encoding="utf-8"))
    if isinstance(original_payload, dict):
        original_payload["rows"] = merged_rows
        merged_payload: Any = original_payload
    else:
        merged_payload = merged_rows
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    merged_path = results_dir / f"evaluation_results_{timestamp}_recovered.json"
    merged_path.write_text(json.dumps(merged_payload, indent=2), encoding="utf-8")
    print(f"Recovered Toolkit result: {merged_path}", flush=True)
    return merged_path, recovery_returncode


def _scorecard_payload(
    rows: list[dict[str, Any]],
    *,
    source: Path,
    agent_deployment: str,
    evaluator_deployment: str,
    pytest_returncode: int,
) -> dict[str, Any]:
    if not rows:
        raise ValueError("Toolkit result contains no rows.")

    observed_evaluators = _observed_evaluators(rows)
    _validate_dimensions(rows, observed_evaluators)

    metrics: dict[str, float] = {}
    for evaluator in observed_evaluators:
        row_metric = _metric_name(evaluator)
        results = [row.get(f"outputs.{evaluator}.result") for row in rows]
        if not all(result in {"pass", "fail"} for result in results):
            invalid_rows = [
                row for row in rows
                if row.get(f"outputs.{evaluator}.result") not in {"pass", "fail"}
            ]
            reasons = list(dict.fromkeys(
                str(row.get(f"outputs.{evaluator}.reason") or "No reason recorded.")
                for row in invalid_rows
            ))
            details = "\n".join(f"- {reason}" for reason in reasons)
            raise EvaluationInfrastructureError(
                f"Evaluator {evaluator!r} returned an infrastructure error for "
                f"{len(invalid_rows)}/{len(rows)} cases.\n"
                f"Artifact: {source}\n{details}"
            )
        metrics[row_metric] = sum(result == "pass" for result in results) / len(results)

    scorecard_rows = []
    for row in rows:
        failure_details = [
            {
                "evaluator": evaluator,
                "reason": row.get(f"outputs.{evaluator}.reason"),
                "score": row.get(f"outputs.{evaluator}.score"),
                "threshold": row.get(f"outputs.{evaluator}.threshold"),
                "diagnostics": diagnostic_evidence(
                    row.get(f"outputs.{evaluator}.raw", {})
                ),
            }
            for evaluator in observed_evaluators
            if row.get(f"outputs.{evaluator}.result") == "fail"
        ]
        scorecard_row = {
            "inputs.id": row.get("inputs.id"),
            "inputs.category": row.get("inputs.category"),
            "inputs.query": row.get("inputs.query"),
            "outputs.response": row.get("outputs.response"),
            "outputs.failure_details": failure_details,
            "outputs.tool_definitions": row.get("outputs.tool_definitions", []),
            "outputs.tool_calls": row.get("outputs.tool_calls", []),
        }
        for evaluator in observed_evaluators:
            row_metric = _metric_name(evaluator)
            scorecard_row[f"outputs.{row_metric}"] = (
                row.get(f"outputs.{evaluator}.result") == "pass"
            )
        scorecard_rows.append(scorecard_row)

    return {
        "metrics": metrics,
        "rows": scorecard_rows,
        "evaluation_metadata": {
            "source": str(source),
            "cases": len(rows),
            "evaluators": list(observed_evaluators),
            "agent_deployment": agent_deployment,
            "evaluator_deployment": evaluator_deployment,
            "pytest_returncode": pytest_returncode,
        },
    }


def display_toolkit_scorecard(
    *,
    result_path: Path,
    root: Path,
    agent_deployment: str,
    evaluator_deployment: str,
    pytest_returncode: int,
    scorecard_path: Path | None = None,
) -> list[dict[str, Any]]:
    """Validate one Toolkit artifact, display its scorecard, and return raw rows."""
    results_dir = root / "test-results"
    results_dir.mkdir(exist_ok=True)
    rows = load_rows(result_path)
    payload = _scorecard_payload(
        rows,
        source=result_path,
        agent_deployment=agent_deployment,
        evaluator_deployment=evaluator_deployment,
        pytest_returncode=pytest_returncode,
    )
    scorecard_json_path = scorecard_path or results_dir / "toolkit-full-scorecard.json"
    scorecard_json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    scorecard_html_path, quality_gate_passed = render_report(scorecard_json_path)

    from IPython.display import HTML, display

    document = scorecard_html_path.read_text(encoding="utf-8")
    display(
        HTML(
            '<iframe title="Toolkit evaluation scorecard" '
            'style="width:100%;height:900px;border:1px solid #dedede;border-radius:8px" '
            f'srcdoc="{escape(document, quote=True)}"></iframe>'
        )
    )
    evaluator_count = len(payload["evaluation_metadata"]["evaluators"])
    print(f"Toolkit source: {result_path}")
    print(f"Scorecard JSON: {scorecard_json_path}")
    print(f"Scorecard HTML: {scorecard_html_path}")
    print(
        f"Matrix: {len(rows)} cases x {evaluator_count} evaluators = "
        f"{len(rows) * evaluator_count} checks"
    )
    print(f"Pytest return code: {pytest_returncode}")
    print(f"Quality gate: {'PASS' if quality_gate_passed else 'FAIL'}")
    return rows
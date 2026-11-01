"""Inspect row-level Foundry Toolkit evaluation evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "test-results"
EVIDENCE_KEYS = ("missing", "extra", "excess", "expected", "actual", "tool")


def latest_result_file(results_dir: Path = RESULTS_DIR) -> Path:
    files = sorted(results_dir.glob("evaluation_results_*.json"))
    if not files:
        raise FileNotFoundError(f"No Toolkit result files found in {results_dir}")
    return files[-1]


def load_rows(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("rows", payload) if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        raise ValueError(f"Expected a result object with a rows list: {path}")
    return rows


def evaluator_names(row: dict[str, Any]) -> list[str]:
    suffix = ".result"
    prefix = "outputs."
    return sorted(
        key[len(prefix) : -len(suffix)]
        for key in row
        if key.startswith(prefix) and key.endswith(suffix)
    )


def diagnostic_evidence(value: Any, path: str = "raw") -> dict[str, Any]:
    evidence: dict[str, Any] = {}
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = f"{path}.{key}"
            if any(token in key.lower() for token in EVIDENCE_KEYS):
                evidence[child_path] = child
            evidence.update(diagnostic_evidence(child, child_path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            evidence.update(diagnostic_evidence(child, f"{path}[{index}]"))
    return evidence


def inspect_row(row: dict[str, Any]) -> dict[str, Any]:
    evaluators: dict[str, Any] = {}
    for name in evaluator_names(row):
        raw = row.get(f"outputs.{name}.raw", {})
        evaluators[name] = {
            "result": row.get(f"outputs.{name}.result"),
            "score": row.get(f"outputs.{name}.score"),
            "threshold": row.get(f"outputs.{name}.threshold"),
            "reason": row.get(f"outputs.{name}.reason"),
            "diagnostics": diagnostic_evidence(raw),
            "raw": raw,
        }

    return {
        "id": row.get("inputs.id"),
        "category": row.get("inputs.category"),
        "query": row.get("inputs.query"),
        "response": row.get("outputs.response"),
        "tool_definitions": row.get("outputs.tool_definitions", []),
        "tool_calls": row.get("outputs.tool_calls", []),
        "evaluators": evaluators,
    }


def select_rows(
    rows: Iterable[dict[str, Any]], case_id: str | None, failures_only: bool
) -> list[dict[str, Any]]:
    selected = []
    for row in rows:
        if case_id and row.get("inputs.id") != case_id:
            continue
        results = [row.get(f"outputs.{name}.result") for name in evaluator_names(row)]
        if failures_only and results and all(result == "pass" for result in results):
            continue
        selected.append(inspect_row(row))
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", nargs="?", type=Path, help="Toolkit result JSON")
    parser.add_argument("--id", dest="case_id", help="Show one dataset case")
    parser.add_argument(
        "--all", action="store_true", help="Include rows whose evaluators all passed"
    )
    args = parser.parse_args()

    path = args.path or latest_result_file()
    inspected = select_rows(load_rows(path), args.case_id, not args.all)
    print(json.dumps({"source": str(path), "rows": inspected}, indent=2))


if __name__ == "__main__":
    main()
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.inspect_toolkit_results import load_rows
from scripts.toolkit_scorecard import (
    TOOLKIT_CASE_COUNT,
    TOOLKIT_EVALUATORS,
    EvaluationInfrastructureError,
    _merge_recovery_rows,
    _recovery_node_ids,
    _scorecard_payload,
    recover_toolkit_result,
    transient_error_case_ids,
    display_toolkit_scorecard,
)


def _row(case_id: str, failing_evaluator: str | None = None) -> dict[str, object]:
    row: dict[str, object] = {
        "inputs.id": case_id,
        "inputs.category": "banking",
        "inputs.query": "Transfer funds",
        "outputs.response": "Done",
        "outputs.tool_calls": [{"name": "transfer_funds"}],
        "outputs.tool_definitions": [{"name": "transfer_funds"}],
    }
    for evaluator in TOOLKIT_EVALUATORS:
        result = "fail" if evaluator == failing_evaluator else "pass"
        row[f"outputs.{evaluator}.result"] = result
        row[f"outputs.{evaluator}.reason"] = "missing transfer" if result == "fail" else None
        row[f"outputs.{evaluator}.raw"] = {"missing": ["transfer_funds"]} if result == "fail" else {}
    return row


class ToolkitScorecardTests(unittest.TestCase):
    def test_cheaper_scorecard_does_not_overwrite_baseline_report(self) -> None:
        rows = [_row(f"case-{index}") for index in range(1, TOOLKIT_CASE_COUNT + 1)]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            results_dir = root / "test-results"
            results_dir.mkdir()
            baseline = results_dir / "toolkit-full-scorecard.json"
            baseline.write_text("baseline evidence", encoding="utf-8")
            baseline_html = baseline.with_suffix(".html")
            baseline_html.write_text("baseline report", encoding="utf-8")
            result_path = results_dir / "evaluation_results_cheaper.json"
            result_path.write_text(json.dumps({"rows": rows}), encoding="utf-8")
            alternate = results_dir / "toolkit-cheaper-model-acs-scorecard.json"

            def render(path):
                self.assertEqual(alternate, path)
                html = path.with_suffix(".html")
                html.write_text("<p>cheaper scorecard</p>", encoding="utf-8")
                return html, True

            with patch("scripts.toolkit_scorecard.render_report", side_effect=render), patch(
                "IPython.display.display",
            ):
                observed = display_toolkit_scorecard(
                    result_path=result_path, root=root, agent_deployment="gpt-5.4-mini",
                    evaluator_deployment="judge", pytest_returncode=0, scorecard_path=alternate,
                )
            self.assertEqual(rows, observed)
            self.assertEqual("baseline evidence", baseline.read_text(encoding="utf-8"))
            self.assertEqual("baseline report", baseline_html.read_text(encoding="utf-8"))
            metadata = json.loads(alternate.read_text(encoding="utf-8"))["evaluation_metadata"]
            self.assertEqual("gpt-5.4-mini", metadata["agent_deployment"])

    def test_recovers_incomplete_dimensions_into_a_new_artifact(self) -> None:
        original_rows = [
            _row(f"case-{index}") for index in range(1, TOOLKIT_CASE_COUNT + 1)
        ]
        del original_rows[1]["outputs.task_adherence.result"]
        recovery_row = _row("case-2", "tool_call_accuracy")
        recovery_row["outputs.response"] = "Recovered response"

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            results_dir = root / "test-results"
            results_dir.mkdir()
            original_path = results_dir / "evaluation_results_original.json"
            original_path.write_text(
                json.dumps({"rows": original_rows}),
                encoding="utf-8",
            )
            recovery_path = results_dir / "evaluation_results_retry.json"
            observed_command: list[str] = []

            class FakeProcess:
                stdout = iter(())

                @staticmethod
                def wait() -> int:
                    return 1

            def fake_popen(command: list[str], **_: object) -> FakeProcess:
                observed_command.extend(command)
                self.assertEqual(
                    "bank_manager.agents.cheaper_model.agent",
                    command[command.index("--agent-module") + 1],
                )
                recovery_path.write_text(
                    json.dumps({"rows": [recovery_row]}),
                    encoding="utf-8",
                )
                return FakeProcess()

            with patch(
                "scripts.toolkit_scorecard.subprocess.Popen",
                side_effect=fake_popen,
            ):
                merged_path, returncode = recover_toolkit_result(
                    result_path=original_path,
                    root=root,
                    agent_deployment="agent-model",
                    evaluator_deployment="judge-model",
                    pytest_returncode=1,
                    python_executable=Path("python"),
                    agent_revision="cheaper_model",
                )

            merged_rows = load_rows(merged_path)

        self.assertEqual(1, returncode)
        self.assertEqual("Recovered response", merged_rows[1]["outputs.response"])
        self.assertEqual(6, sum("::test_" in arg for arg in observed_command))
        self.assertNotIn("--cache-clear", observed_command)

    def test_explicit_incomplete_recovery_requires_a_known_incomplete_case(self) -> None:
        rows = [_row("complete"), _row("incomplete")]
        del rows[1]["outputs.task_adherence.result"]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "result.json"
            path.write_text(json.dumps({"rows": rows}), encoding="utf-8")
            for case_id, message in [("missing", "Unknown recovery case"), ("complete", "already complete")]:
                with self.assertRaisesRegex(EvaluationInfrastructureError, message):
                    recover_toolkit_result(
                        result_path=path, root=root, agent_deployment="model",
                        evaluator_deployment="judge", pytest_returncode=1,
                        incomplete_case_ids=[case_id], agent_revision="cheaper_model",
                    )

    def test_selects_complete_case_reruns_for_transient_errors(self) -> None:
        rows = [_row(f"case-{index}") for index in range(1, TOOLKIT_CASE_COUNT + 1)]
        rows[1]["outputs.relevance.result"] = "error"
        rows[1]["outputs.relevance.reason"] = (
            "Evaluator failed: APIConnectionError: Connection error."
        )

        case_ids = transient_error_case_ids(rows)
        node_ids = _recovery_node_ids(case_ids)

        self.assertEqual(["case-2"], case_ids)
        self.assertEqual(6, len(node_ids))
        self.assertTrue(all(node_id.endswith("[case-2]") for node_id in node_ids))

    def test_does_not_retry_non_transient_evaluator_errors(self) -> None:
        rows = [_row(f"case-{index}") for index in range(1, TOOLKIT_CASE_COUNT + 1)]
        rows[1]["outputs.relevance.result"] = "error"
        rows[1]["outputs.relevance.reason"] = "HTTP 400 invalid request"

        self.assertEqual([], transient_error_case_ids(rows))

    def test_replaces_complete_recovery_rows(self) -> None:
        original_rows = [
            _row(f"case-{index}") for index in range(1, TOOLKIT_CASE_COUNT + 1)
        ]
        original_rows[1]["outputs.relevance.result"] = "error"
        recovery_row = _row("case-2", "tool_call_accuracy")
        recovery_row["outputs.response"] = "Recovered response"

        merged = _merge_recovery_rows(
            original_rows,
            [recovery_row],
            ["case-2"],
        )

        self.assertEqual("Recovered response", merged[1]["outputs.response"])
        self.assertEqual(
            "fail",
            merged[1]["outputs.tool_call_accuracy.result"],
        )
        self.assertIs(original_rows[0], merged[0])

    def test_rejects_recovery_rows_with_exhausted_errors(self) -> None:
        original_rows = [
            _row(f"case-{index}") for index in range(1, TOOLKIT_CASE_COUNT + 1)
        ]
        recovery_row = _row("case-2")
        recovery_row["outputs.relevance.result"] = "error"
        recovery_row["outputs.relevance.reason"] = "APIConnectionError"

        with self.assertRaisesRegex(
            EvaluationInfrastructureError,
            "still has infrastructure errors",
        ):
            _merge_recovery_rows(
                original_rows,
                [recovery_row],
                ["case-2"],
            )

    def test_builds_complete_matrix_and_failure_evidence(self) -> None:
        rows = [
            _row(f"case-{index}", "tool_call_accuracy" if index == 1 else None)
            for index in range(1, TOOLKIT_CASE_COUNT + 1)
        ]

        payload = _scorecard_payload(
            rows,
            source=Path("result.json"),
            agent_deployment="agent-model",
            evaluator_deployment="evaluator-model",
            pytest_returncode=0,
        )

        self.assertEqual(
            TOOLKIT_CASE_COUNT,
            payload["evaluation_metadata"]["cases"],
        )
        self.assertEqual(
            (TOOLKIT_CASE_COUNT - 1) / TOOLKIT_CASE_COUNT,
            payload["metrics"]["tool_call_accuracy.tool_call_accuracy_passed"],
        )
        failure = payload["rows"][0]["outputs.failure_details"][0]
        self.assertEqual("tool_call_accuracy", failure["evaluator"])
        self.assertEqual(["transfer_funds"], failure["diagnostics"]["raw.missing"])

    def test_accepts_dynamic_case_and_evaluator_dimensions(self) -> None:
        rows = [_row("case-1"), _row("case-2", "relevance")]
        selected = {"relevance", "fluency"}
        for row in rows:
            for evaluator in set(TOOLKIT_EVALUATORS) - selected:
                for suffix in ("result", "reason", "raw"):
                    row.pop(f"outputs.{evaluator}.{suffix}")

        payload = _scorecard_payload(
            rows,
            source=Path("result.json"),
            agent_deployment="agent-model",
            evaluator_deployment="evaluator-model",
            pytest_returncode=0,
        )

        self.assertEqual(2, payload["evaluation_metadata"]["cases"])
        self.assertEqual(["fluency", "relevance"], payload["evaluation_metadata"]["evaluators"])
        self.assertEqual(0.5, payload["metrics"]["relevance.relevance_passed"])
        self.assertEqual(1.0, payload["metrics"]["fluency.fluency_passed"])

    def test_rejects_inconsistent_evaluator_dimensions(self) -> None:
        rows = [_row("case-1"), _row("case-2")]
        rows[1].pop("outputs.relevance.result")

        with self.assertRaisesRegex(
            EvaluationInfrastructureError,
            "inconsistent evaluator dimensions",
        ):
            _scorecard_payload(
                rows,
                source=Path("result.json"),
                agent_deployment="agent-model",
                evaluator_deployment="evaluator-model",
                pytest_returncode=0,
            )

    def test_reports_evaluator_infrastructure_error(self) -> None:
        rows = [_row(f"case-{index}") for index in range(1, TOOLKIT_CASE_COUNT + 1)]
        reason = "Unsupported parameter: 'max_tokens'. Use 'max_completion_tokens'."
        for row in rows:
            row["outputs.intent_resolution.result"] = "error"
            row["outputs.intent_resolution.reason"] = reason

        with self.assertRaises(EvaluationInfrastructureError) as raised:
            _scorecard_payload(
                rows,
                source=Path("failed-result.json"),
                agent_deployment="agent-model",
                evaluator_deployment="gpt-5.5",
                pytest_returncode=1,
            )

        message = str(raised.exception)
        self.assertIn(
            f"{TOOLKIT_CASE_COUNT}/{TOOLKIT_CASE_COUNT} cases",
            message,
        )
        self.assertIn("failed-result.json", message)
        self.assertIn("max_completion_tokens", message)


if __name__ == "__main__":
    unittest.main()
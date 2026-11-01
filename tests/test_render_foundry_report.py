from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.render_foundry_report import render_report


class RenderFoundryReportTests(unittest.TestCase):
    def test_only_renders_thresholds_for_evaluated_dimensions(self) -> None:
        result = {
            "metrics": {"quality.binary_aggregate": 1.0},
            "rows": [
                {
                    "inputs.id": "case-1",
                    "outputs.quality.passed": True,
                }
            ],
            "evaluation_metadata": {"evaluators": ["quality"]},
        }
        thresholds = {
            "quality": {
                "label": "Quality",
                "metric": "quality.binary_aggregate",
                "operator": ">=",
                "threshold": 1.0,
                "row_metric": "quality.passed",
                "row_threshold": 1.0,
            },
            "unused": {
                "label": "Unused dimension",
                "metric": "unused.binary_aggregate",
                "operator": ">=",
                "threshold": 1.0,
                "row_metric": "unused.passed",
                "row_threshold": 1.0,
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result_path = root / "result.json"
            thresholds_path = root / "thresholds.json"
            result_path.write_text(json.dumps(result), encoding="utf-8")
            thresholds_path.write_text(json.dumps(thresholds), encoding="utf-8")

            output, passed = render_report(result_path, thresholds_path)
            document = output.read_text(encoding="utf-8")

        self.assertTrue(passed)
        self.assertIn("Quality", document)
        self.assertNotIn("Unused dimension", document)
        self.assertIn('<html lang="en" data-theme="light">', document)
        self.assertNotIn("prefers-color-scheme", document)
        self.assertNotIn("scoutTheme", document)

    def test_tool_call_accuracy_is_a_promotion_gate(self) -> None:
        result = {
            "metrics": {
                "quality.binary_aggregate": 1.0,
                "tool_call_accuracy.tool_call_accuracy": 5.0,
                "tool_call_accuracy.tool_call_accuracy_passed": 0.94,
            },
            "rows": [
                {
                    "inputs.id": "case-1",
                    "inputs.query": "Check the balance",
                    "inputs.response": "Done",
                    "outputs.quality.passed": True,
                    "outputs.tool_call_accuracy.tool_call_accuracy": 5.0,
                    "outputs.tool_call_accuracy.tool_call_accuracy_passed": True,
                }
            ],
        }
        thresholds = {
            "quality": {
                "label": "Quality",
                "metric": "quality.binary_aggregate",
                "operator": ">=",
                "threshold": 1.0,
                "row_metric": "quality.passed",
                "row_operator": ">=",
                "row_threshold": 1.0,
            },
            "tool_call_accuracy": {
                "label": "Tool Call Accuracy",
                "metric": "tool_call_accuracy.tool_call_accuracy_passed",
                "operator": ">=",
                "threshold": 0.95,
                "unit": "percent",
                "row_metric": "tool_call_accuracy.tool_call_accuracy_passed",
                "row_threshold": 1.0,
            }
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result_path = root / "result.json"
            thresholds_path = root / "thresholds.json"
            result_path.write_text(json.dumps(result), encoding="utf-8")
            thresholds_path.write_text(json.dumps(thresholds), encoding="utf-8")

            output, passed = render_report(result_path, thresholds_path)
            document = output.read_text(encoding="utf-8")

        self.assertFalse(passed)
        self.assertIn("Tool Call Accuracy", document)
        self.assertIn("94%", document)
        self.assertIn("Gate &gt;= 95%", document)
        self.assertNotIn("diagnostic only", document)

    def test_latency_is_reported_but_not_a_promotion_gate(self) -> None:
        result = {
            "metrics": {"quality.binary_aggregate": 1.0},
            "rows": [
                {
                    "inputs.id": "case-1",
                    "inputs.query": "Check the balance",
                    "inputs.response": "Done",
                    "inputs.latency_ms": 45000.0,
                    "outputs.quality.passed": True,
                },
                {
                    "inputs.id": "case-2",
                    "inputs.query": "Check the loan",
                    "inputs.response": "Done",
                    "inputs.latency_ms": 55000.0,
                    "outputs.quality.passed": True,
                },
            ],
        }
        thresholds = {
            "quality": {
                "label": "Quality",
                "metric": "quality.binary_aggregate",
                "operator": ">=",
                "threshold": 0.95,
                "row_metric": "quality.passed",
                "row_operator": ">=",
                "row_threshold": 1.0,
            }
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result_path = root / "result.json"
            thresholds_path = root / "thresholds.json"
            result_path.write_text(json.dumps(result), encoding="utf-8")
            thresholds_path.write_text(json.dumps(thresholds), encoding="utf-8")

            output, passed = render_report(result_path, thresholds_path)
            document = output.read_text(encoding="utf-8")

        self.assertTrue(passed)
        self.assertIn("P95 latency", document)
        self.assertIn("54.50 s", document)
        self.assertIn("report only", document)
        self.assertNotIn("Gate &lt;=", document)

    def test_failure_evidence_is_rendered(self) -> None:
        result = {
            "metrics": {"quality.binary_aggregate": 0.0},
            "rows": [
                {
                    "inputs.id": "case-1",
                    "inputs.query": "Transfer funds",
                    "outputs.response": "I cannot do that.",
                    "outputs.quality.passed": False,
                    "outputs.failure_details": [
                        {
                            "evaluator": "tool_call_accuracy",
                            "reason": "The transfer tool was not called.",
                            "diagnostics": {"missing": ["transfer_funds"]},
                        }
                    ],
                    "outputs.tool_calls": [],
                    "outputs.tool_definitions": [{"name": "transfer_funds"}],
                }
            ],
        }
        thresholds = {
            "quality": {
                "label": "Quality",
                "metric": "quality.binary_aggregate",
                "operator": ">=",
                "threshold": 1.0,
                "row_metric": "quality.passed",
                "row_threshold": 1.0,
            }
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result_path = root / "result.json"
            thresholds_path = root / "thresholds.json"
            result_path.write_text(json.dumps(result), encoding="utf-8")
            thresholds_path.write_text(json.dumps(thresholds), encoding="utf-8")

            output, passed = render_report(result_path, thresholds_path)
            document = output.read_text(encoding="utf-8")

        self.assertFalse(passed)
        self.assertIn("tool_call_accuracy", document)
        self.assertIn("The transfer tool was not called.", document)
        self.assertIn("transfer_funds", document)

    def test_structured_query_and_response_are_multiline_json(self) -> None:
        result = {
            "metrics": {"quality.binary_aggregate": 1.0},
            "rows": [
                {
                    "inputs.id": "case-1",
                    "inputs.query": [
                        {"role": "user", "content": [{"type": "text", "text": "Check balance"}]}
                    ],
                    "outputs.response": [
                        {
                            "role": "assistant",
                            "content": [{"type": "text", "text": "The balance is 100 USD."}],
                        }
                    ],
                    "outputs.quality.passed": True,
                }
            ],
        }
        thresholds = {
            "quality": {
                "label": "Quality",
                "metric": "quality.binary_aggregate",
                "operator": ">=",
                "threshold": 1.0,
                "row_metric": "quality.passed",
                "row_threshold": 1.0,
            }
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result_path = root / "result.json"
            thresholds_path = root / "thresholds.json"
            result_path.write_text(json.dumps(result), encoding="utf-8")
            thresholds_path.write_text(json.dumps(thresholds), encoding="utf-8")

            output, passed = render_report(result_path, thresholds_path)
            document = output.read_text(encoding="utf-8")

        self.assertTrue(passed)
        self.assertEqual(2, document.count('<pre class="message-evidence">'))
        self.assertIn("[\n  {\n    &quot;role&quot;: &quot;user&quot;", document)
        self.assertIn("&quot;text&quot;: &quot;The balance is 100 USD.&quot;", document)


if __name__ == "__main__":
    unittest.main()
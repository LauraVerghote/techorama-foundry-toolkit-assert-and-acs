"""Render a self-contained pass/fail scorecard from Foundry evaluation JSON."""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_THRESHOLDS = ROOT / "bank_manager" / "evals" / "foundry_thresholds.json"
REPORT_ONLY_METRICS = {
    "latency": {
        "label": "P95 latency",
        "row_label": "Latency",
        "row_metrics": ["latency.latency_ms", "inputs.latency_ms"],
        "unit": "ms",
        "percentile": 95,
    }
}


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _find_metric(values: dict[str, Any], requested: str) -> tuple[str, float] | None:
    candidates = [requested, f"outputs.{requested}"]
    candidates.extend(
        key for key in values if key.endswith(f".{requested}") or key.endswith(requested)
    )
    for key in candidates:
        value = _number(values.get(key))
        if value is not None:
            return key, value
    return None


def _find_first_metric(
    values: dict[str, Any], requested: list[str]
) -> tuple[str, float] | None:
    return next(
        (match for metric in requested if (match := _find_metric(values, metric))),
        None,
    )


def _passes(value: float, operator: str, threshold: float) -> bool:
    if operator == ">=":
        return value >= threshold
    if operator == "<=":
        return value <= threshold
    raise ValueError(f"Unsupported threshold operator: {operator}")


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    rank = (len(ordered) - 1) * percentile / 100
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (rank - lower)


def _display(value: float | None, gate: dict[str, Any]) -> str:
    if value is None:
        return "Not scored"
    if gate.get("unit") == "percent":
        return f"{value * 100:.0f}%"
    if gate.get("unit") == "ms":
        return f"{value / 1000:.2f} s"
    return f"{value:.2f}".rstrip("0").rstrip(".")


def _threshold_text(gate: dict[str, Any]) -> str:
    threshold = float(gate["threshold"])
    return f"{gate['operator']} {_display(threshold, gate)}"


def _gate_results(
    metrics: dict[str, Any],
    thresholds: dict[str, dict[str, Any]],
    *,
    row_level: bool = False,
) -> list[dict[str, Any]]:
    results = []
    for gate_id, gate in thresholds.items():
        metric = gate.get("row_metric", gate["metric"]) if row_level else gate["metric"]
        operator = gate.get("row_operator", gate["operator"]) if row_level else gate["operator"]
        threshold = gate.get("row_threshold", gate["threshold"]) if row_level else gate["threshold"]
        display_gate = {
            **gate,
            "unit": gate.get("row_unit", gate.get("unit")) if row_level else gate.get("unit"),
        }
        match = _find_metric(metrics, str(metric))
        value = match[1] if match else None
        passed = value is not None and _passes(
            value, str(operator), float(threshold)
        )
        results.append(
            {
                "id": gate_id,
                "label": gate.get("label", gate_id.replace("_", " ").title()),
                "metric": match[0] if match else metric,
                "value": value,
                "display": _display(value, display_gate),
                "threshold": _threshold_text(
                    {**display_gate, "operator": operator, "threshold": threshold}
                ),
                "passed": passed,
                "missing": value is None,
            }
        )
    return results


def _row_results(
    rows: list[dict[str, Any]], thresholds: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    results = []
    for index, row in enumerate(rows, start=1):
        gates = _gate_results(row, thresholds, row_level=True)
        diagnostics = []
        for diagnostic_id, diagnostic in REPORT_ONLY_METRICS.items():
            match = _find_first_metric(row, diagnostic["row_metrics"])
            if match:
                diagnostics.append(
                    {
                        "id": diagnostic_id,
                        "label": diagnostic["row_label"],
                        "display": _display(match[1], diagnostic),
                        "passed": None,
                    }
                )
        results.append(
            {
                "id": row.get("inputs.id", f"row-{index}"),
                "category": row.get("inputs.category", "uncategorized"),
                "query": row.get("inputs.query", ""),
                "response": row.get("inputs.response", row.get("outputs.response", "")),
                "gates": gates,
                "diagnostics": diagnostics,
                "failure_details": row.get("outputs.failure_details", []),
                "tool_definitions": row.get("outputs.tool_definitions", []),
                "tool_calls": row.get("outputs.tool_calls", []),
                "passed": bool(gates) and all(gate["passed"] for gate in gates),
            }
        )
    return results


def _report_only_results(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    results = []
    for metric_id, metric in REPORT_ONLY_METRICS.items():
        values = [
            match[1]
            for row in rows
            if (match := _find_first_metric(row, metric["row_metrics"]))
        ]
        if values:
            value = _percentile(values, float(metric["percentile"]))
            results.append(
                {
                    "id": metric_id,
                    "label": metric["label"],
                    "score": _display(value, metric),
                    "detail": "report only · excluded from promotion gates",
                }
            )
    return results


def _derive_success_metrics(
    metrics: dict[str, Any],
    rows: list[dict[str, Any]],
    thresholds: dict[str, dict[str, Any]],
) -> None:
    for gate in thresholds.values():
        metric = str(gate["metric"])
        if _find_metric(metrics, metric) or not rows:
            continue
        if gate.get("aggregation") == "percentile":
            row_metric = gate.get("row_metric")
            values = [
                match[1]
                for row in rows
                if row_metric and (match := _find_metric(row, str(row_metric)))
            ]
            if values:
                metrics[metric] = _percentile(values, float(gate["percentile"]))
            continue
        if "row_metric" not in gate:
            continue
        passed = 0
        for row in rows:
            match = _find_metric(row, str(gate["row_metric"]))
            row_passed = match is not None and _passes(
                match[1],
                str(gate.get("row_operator", gate["operator"])),
                float(gate.get("row_threshold", gate["threshold"])),
            )
            required = gate.get("row_required", {})
            row_passed = row_passed and all(row.get(key) == value for key, value in required.items())
            passed += int(row_passed)
        metrics[metric] = passed / len(rows)


def _escape(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _evidence_text(value: Any) -> str:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, indent=2, ensure_ascii=False, default=str)
    return str(value)


def _render_gate(gate: dict[str, Any]) -> str:
    state = "pass" if gate["passed"] else "fail"
    status = "PASS" if gate["passed"] else ("MISSING" if gate["missing"] else "FAIL")
    return f"""
      <article class="gate {state}">
        <div class="gate-head"><span>{_escape(gate['label'])}</span><strong>{status}</strong></div>
        <div class="metric-value">{_escape(gate['display'])}</div>
        <div class="metric-rule">Gate {_escape(gate['threshold'])}</div>
      </article>"""


def _render_row(row: dict[str, Any]) -> str:
    state = "pass" if row["passed"] else "fail"
    chips = "".join(
        f'<span class="chip {"pass" if gate["passed"] else "fail"}">'
        f'{_escape(gate["label"])}: {_escape(gate["display"])}</span>'
        for gate in row["gates"]
    )
    chips += "".join(
        f'<span class="chip diagnostic {"" if diagnostic["passed"] is None else ("pass" if diagnostic["passed"] else "fail")}">'
        f'{_escape(diagnostic["label"])}: {_escape(diagnostic["display"])}</span>'
        for diagnostic in row["diagnostics"]
    )
    failure_details = "".join(
        f'<article class="failure-detail"><h3>{_escape(detail.get("evaluator", "Evaluator"))}</h3>'
        f'<p>{_escape(detail.get("reason") or "No reason returned.")}</p>'
        f'<pre>{_escape(json.dumps(detail.get("diagnostics", {}), indent=2, default=str))}</pre></article>'
        for detail in row["failure_details"]
    )
    tool_evidence = ""
    if row["tool_calls"] or row["tool_definitions"]:
        tool_evidence = (
            '<article class="tool-evidence"><h3>Tool evidence</h3>'
            f'<pre>{_escape(json.dumps({"calls": row["tool_calls"], "definitions": row["tool_definitions"]}, indent=2, default=str))}</pre></article>'
        )
    return f"""
      <details class="test-row {state}">
        <summary>
          <span class="status-dot" aria-hidden="true"></span>
          <span class="test-name">{_escape(row['id'])}</span>
          <span class="category">{_escape(row['category'])}</span>
          <strong>{'PASS' if row['passed'] else 'FAIL'}</strong>
        </summary>
        <div class="test-body">
          <div><h3>Query</h3><pre class="message-evidence">{_escape(_evidence_text(row['query']))}</pre></div>
          <div><h3>Response</h3><pre class="message-evidence">{_escape(_evidence_text(row['response']))}</pre></div>
          <div class="chips">{chips or '<span class="chip">No row metrics</span>'}</div>
          {f'<div class="evidence">{failure_details}{tool_evidence}</div>' if failure_details or tool_evidence else ''}
        </div>
      </details>"""


def render_report(
    result_path: Path,
    thresholds_path: Path = DEFAULT_THRESHOLDS,
    output_path: Path | None = None,
) -> tuple[Path, bool]:
    result = json.loads(result_path.read_text(encoding="utf-8"))
    thresholds = json.loads(thresholds_path.read_text(encoding="utf-8"))
    evaluated_dimensions = set(
        result.get("evaluation_metadata", {}).get("evaluators", [])
    )
    if evaluated_dimensions:
        thresholds = {
            gate_id: gate
            for gate_id, gate in thresholds.items()
            if gate_id in evaluated_dimensions
        }
    metrics = result.get("metrics", {})
    result_rows = result.get("rows", [])
    _derive_success_metrics(metrics, result_rows, thresholds)
    gates = _gate_results(metrics, thresholds)
    report_only = _report_only_results(result_rows)
    rows = _row_results(result_rows, thresholds)
    overall_passed = bool(gates) and all(gate["passed"] for gate in gates)
    passed_rows = sum(row["passed"] for row in rows)
    state = "pass" if overall_passed else "fail"
    output_path = output_path or result_path.with_suffix(".html")

    gate_html = "".join(_render_gate(gate) for gate in gates)
    diagnostic_html = "".join(
        f'<article class="diagnostic-card"><div>{_escape(metric["label"])}</div>'
        f'<strong>{_escape(metric["score"])}</strong>'
        f'<span>{_escape(metric["detail"])}</span></article>'
        for metric in report_only
    )
    row_html = "".join(_render_row(row) for row in rows)
    missing = [gate["label"] for gate in gates if gate["missing"]]
    missing_html = (
        f'<div class="notice">Missing aggregate metrics: {_escape(", ".join(missing))}. '
        "A missing required metric fails the gate.</div>"
        if missing
        else ""
    )

    document = f"""<!doctype html>
<html lang="en" data-theme="light">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Foundry Evaluation Scorecard</title>
  <style>
    :root {{
      color-scheme: light;
      --cp-bg: #f7f4ef; --cp-bg-elevated: #fcfbf8; --cp-surface: #ffffff;
      --cp-surface-soft: #f5f5f5; --cp-border: #dedede; --cp-border-strong: #919191;
      --cp-text: #242424; --cp-text-muted: #5c5c5c; --cp-text-soft: #6f6f6f;
      --cp-accent: #b11f4b; --cp-accent-hover: #9a1a41; --cp-accent-soft: rgba(177,31,75,.08);
      --cp-accent-fg: #ffffff; --cp-success: #16a34a; --cp-danger: #dc2626;
      --cp-warning: #f59e0b; --cp-link: #0078d4; --cp-shadow: 0 18px 48px rgba(0,0,0,.12);
      --cp-overlay: rgba(255,255,255,.8); --cp-panel: rgba(255,255,255,.86);
      --cp-panel-strong: rgba(255,255,255,.96); --cp-sheen: rgba(255,255,255,.55);
      --cp-highlight: rgba(177,31,75,.12);
    }}
    html[data-theme="dark"] {{
      color-scheme: dark;
      --cp-bg: #3d3b3a; --cp-bg-elevated: #343231; --cp-surface: #292929;
      --cp-surface-soft: #2e2e2e; --cp-border: #474747; --cp-border-strong: #5f5f5f;
      --cp-text: #dedede; --cp-text-muted: #919191; --cp-text-soft: #b0b0b0;
      --cp-accent: #fd8ea1; --cp-accent-hover: #fb7b91; --cp-accent-soft: rgba(253,142,161,.14);
      --cp-accent-fg: #1a1a1a; --cp-success: #4ade80; --cp-danger: #f87171;
      --cp-warning: #fbbf24; --cp-link: #4da6ff; --cp-shadow: 0 18px 48px rgba(0,0,0,.32);
      --cp-overlay: rgba(41,41,41,.88); --cp-panel: rgba(41,41,41,.72);
      --cp-panel-strong: rgba(41,41,41,.96); --cp-sheen: rgba(255,255,255,.04);
      --cp-highlight: rgba(253,142,161,.12);
    }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; background: var(--cp-bg); color: var(--cp-text); font-family: "Segoe UI", Aptos, Calibri, -apple-system, BlinkMacSystemFont, sans-serif; }}
    main {{ width: min(1120px, calc(100% - 32px)); margin: 32px auto 64px; }}
    header {{ display: grid; grid-template-columns: 1fr auto; gap: 24px; align-items: end; padding-bottom: 24px; border-bottom: 1px solid var(--cp-border); }}
    .eyebrow {{ color: var(--cp-accent); font-size: 13px; font-weight: 700; text-transform: uppercase; }}
    h1 {{ margin: 8px 0 0; font-size: clamp(30px, 5vw, 58px); line-height: 1; letter-spacing: 0; }}
    .verdict {{ min-width: 180px; border-left: 5px solid var(--cp-danger); padding: 8px 0 8px 20px; }}
    .verdict.pass {{ border-color: var(--cp-success); }}
    .verdict span {{ display: block; color: var(--cp-text-muted); font-size: 13px; }}
    .verdict strong {{ display: block; margin-top: 4px; color: var(--cp-danger); font-size: 36px; }}
    .verdict.pass strong {{ color: var(--cp-success); }}
    .meta {{ margin: 20px 0; color: var(--cp-text-muted); font-family: Consolas, "Courier New", Courier, monospace; font-size: 13px; overflow-wrap: anywhere; }}
    .gates {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; }}
    .gate {{ min-width: 0; padding: 18px; background: var(--cp-surface); border: 1px solid var(--cp-border); border-top: 4px solid var(--cp-danger); border-radius: 10px; }}
    .gate.pass {{ border-top-color: var(--cp-success); }}
    .gate-head {{ display: flex; justify-content: space-between; gap: 8px; color: var(--cp-text-muted); font-size: 13px; }}
    .gate-head strong {{ color: var(--cp-danger); }} .gate.pass .gate-head strong {{ color: var(--cp-success); }}
    .metric-value {{ margin-top: 20px; font-size: 30px; font-weight: 700; }}
    .metric-rule {{ margin-top: 6px; color: var(--cp-text-muted); font-size: 13px; }}
    .notice {{ margin-top: 16px; padding: 12px 16px; border-left: 4px solid var(--cp-warning); background: var(--cp-surface); }}
    .diagnostics {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin-top: 12px; }}
    .diagnostic-card {{ padding: 16px 18px; background: var(--cp-surface); border: 1px solid var(--cp-border); border-left: 4px solid var(--cp-link); border-radius: 10px; }}
    .diagnostic-card div, .diagnostic-card span {{ color: var(--cp-text-muted); font-size: 13px; }}
    .diagnostic-card strong {{ display: block; margin: 8px 0 4px; font-size: 26px; }}
    section {{ margin-top: 32px; }}
    .section-head {{ display: flex; align-items: baseline; justify-content: space-between; gap: 16px; margin-bottom: 12px; }}
    h2 {{ margin: 0; font-size: 22px; }} .section-head span {{ color: var(--cp-text-muted); }}
    .test-row {{ margin-bottom: 8px; background: var(--cp-surface); border: 1px solid var(--cp-border); border-radius: 10px; overflow: hidden; }}
    .test-row summary {{ display: grid; grid-template-columns: 12px minmax(140px, 1fr) minmax(120px, .6fr) auto; gap: 12px; align-items: center; padding: 14px 16px; cursor: pointer; }}
    .status-dot {{ width: 9px; height: 9px; background: var(--cp-danger); border-radius: 50%; }} .test-row.pass .status-dot {{ background: var(--cp-success); }}
    .test-name {{ font-weight: 650; overflow-wrap: anywhere; }} .category {{ color: var(--cp-text-muted); font-size: 13px; }}
    .test-row summary strong {{ color: var(--cp-danger); font-size: 13px; }} .test-row.pass summary strong {{ color: var(--cp-success); }}
    .test-body {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; padding: 18px 40px 20px; border-top: 1px solid var(--cp-border); background: var(--cp-surface-soft); }}
    .test-body h3 {{ margin: 0 0 6px; color: var(--cp-text-muted); font-size: 12px; text-transform: uppercase; }}
    .message-evidence {{ margin: 0; max-height: 480px; overflow: auto; white-space: pre-wrap; overflow-wrap: anywhere; line-height: 1.5; }}
    .chips {{ grid-column: 1 / -1; display: flex; flex-wrap: wrap; gap: 8px; }}
    .chip {{ padding: 5px 8px; border: 1px solid var(--cp-border); border-radius: 10px; color: var(--cp-text-muted); font-size: 12px; }}
    .chip.pass {{ color: var(--cp-success); border-color: var(--cp-success); }} .chip.fail {{ color: var(--cp-danger); border-color: var(--cp-danger); }}
    .chip.diagnostic {{ border-style: dashed; }}
    .evidence {{ grid-column: 1 / -1; display: grid; gap: 10px; }}
    .failure-detail, .tool-evidence {{ padding: 14px; background: var(--cp-surface); border-left: 4px solid var(--cp-danger); }}
    .tool-evidence {{ border-left-color: var(--cp-link); }}
    .evidence h3 {{ margin: 0 0 8px; }} .evidence p {{ margin-bottom: 10px; }}
    pre {{ max-height: 320px; margin: 0; padding: 10px; overflow: auto; background: var(--cp-bg-elevated); white-space: pre-wrap; font-size: 12px; }}
    footer {{ margin-top: 32px; color: var(--cp-text-muted); font-size: 13px; }}
    @media (max-width: 800px) {{ .gates, .diagnostics {{ grid-template-columns: repeat(2, 1fr); }} header {{ grid-template-columns: 1fr; }} .test-body {{ grid-template-columns: 1fr; }} .test-body > *, .chips {{ grid-column: 1; }} }}
    @media (max-width: 520px) {{ main {{ width: min(100% - 20px, 1120px); margin-top: 20px; }} .gates, .diagnostics {{ grid-template-columns: 1fr; }} .test-row summary {{ grid-template-columns: 12px 1fr auto; }} .category {{ display: none; }} }}
  </style>
</head>
<body>
  <main>
    <header>
    <div><div class="eyebrow">Microsoft Foundry Evaluation</div><h1>Agent performance scorecard</h1></div>
      <div class="verdict {state}"><span>Promotion decision</span><strong>{'PASS' if overall_passed else 'FAIL'}</strong></div>
    </header>
    <div class="meta">Source: {_escape(result_path.name)} · {len(rows)} test cases · {passed_rows}/{len(rows)} rows passed available row-level gates</div>
    <div class="gates">{gate_html}</div>
    {f'<section><div class="section-head"><h2>Reported metrics</h2><span>Visible for analysis, excluded from promotion gates</span></div><div class="diagnostics">{diagnostic_html}</div></section>' if diagnostic_html else ''}
    {missing_html}
    <section>
      <div class="section-head"><h2>Test evidence</h2><span>Expand a row to inspect its query and response</span></div>
      {row_html or '<div class="notice">No test rows found in this result.</div>'}
    </section>
    <footer>Threshold contract: {_escape(thresholds_path.name)}. Change thresholds before running comparisons, not after seeing candidate results.</footer>
  </main>
</body>
</html>"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(document, encoding="utf-8")
    return output_path, overall_passed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", type=Path)
    parser.add_argument("--thresholds", type=Path, default=DEFAULT_THRESHOLDS)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output, passed = render_report(
        args.result.resolve(), args.thresholds.resolve(), args.output.resolve() if args.output else None
    )
    print(f"Scorecard: {output}")
    print(f"Quality gate: {'PASS' if passed else 'FAIL'}")


if __name__ == "__main__":
    main()
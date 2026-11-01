"""Render a Toolkit row as an inline, ordered agent trajectory."""

from __future__ import annotations

import json
from html import escape
from typing import Any


def _formatted(value: Any) -> str:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (TypeError, ValueError):
            return value
    return json.dumps(value, indent=2, ensure_ascii=False)


def _card(
    number: int,
    kind: str,
    title: str,
    summary: str,
    payload: Any,
) -> str:
    return (
        f'<article class="trace-card {escape(kind)}">'
        f'<div class="trace-index">{number}</div>'
        '<div class="trace-body">'
        f'<div class="trace-kind">{escape(kind.replace("-", " ").upper())}</div>'
        f'<h3>{escape(title)}</h3>'
        f'<p>{escape(summary)}</p>'
        '<details><summary>Input / output payload</summary>'
        f'<pre>{escape(_formatted(payload))}</pre>'
        '</details></div></article>'
    )


def _tool_execution_summary(
    response: list[dict[str, Any]],
) -> tuple[str, str, str, dict[int, tuple[int, int]]]:
    """Classify calls by assistant turn so sequencing is visible and auditable."""
    call_groups: list[tuple[int, int, int]] = []
    assistant_turn = 0
    total_calls = 0
    for message_index, message in enumerate(response):
        if message.get("role") != "assistant":
            continue
        assistant_turn += 1
        group_size = sum(
            part.get("type") == "tool_call"
            for part in message.get("content") or []
        )
        if group_size:
            call_groups.append((message_index, assistant_turn, group_size))
            total_calls += group_size

    group_by_message = {
        message_index: (turn, group_size)
        for message_index, turn, group_size in call_groups
    }
    parallel_groups = [
        (turn, size) for _, turn, size in call_groups if size > 1
    ]
    if parallel_groups:
        details = ", ".join(
            f"assistant turn {turn} emitted {size} calls"
            for turn, size in parallel_groups
        )
        return (
            "parallel",
            "BATCHED / PARALLEL TOOL CALLS",
            (
                f"{total_calls} tool calls were emitted across "
                f"{len(call_groups)} assistant turns; {details} before any "
                "result was available."
            ),
            group_by_message,
        )
    if total_calls > 1:
        return (
            "sequential",
            "SEQUENTIAL TOOL CALLS",
            (
                f"{total_calls} tool calls were emitted in {len(call_groups)} "
                "separate assistant turns. Each call's result appears before "
                "the next assistant tool-call turn."
            ),
            group_by_message,
        )
    if total_calls == 1:
        return (
            "single",
            "SINGLE TOOL CALL",
            "One tool call was emitted, so no parallelism was involved.",
            group_by_message,
        )
    return (
        "none",
        "NO TOOL CALLS",
        "The captured response contains no tool calls.",
        group_by_message,
    )


def render_toolkit_trace(trace: dict[str, Any], evaluator: str) -> str:
    """Return notebook-safe HTML for one captured agent/evaluator trace."""
    response = trace.get("response") or []
    pattern, pattern_title, pattern_detail, group_by_message = (
        _tool_execution_summary(response)
    )
    cards = [
        _card(
            1,
            "user-input",
            "User input",
            str(trace.get("query") or ""),
            {"query": trace.get("query")},
        )
    ]
    number = 2
    pending_tools: list[str] = []
    tool_step = 0
    for message_index, message in enumerate(response):
        group = group_by_message.get(message_index)
        group_call_index = 0
        for part in message.get("content") or []:
            part_type = part.get("type")
            if part_type == "tool_call":
                tool_step += 1
                group_call_index += 1
                tool_name = str(part.get("name") or "unknown tool")
                pending_tools.append(tool_name)
                arguments = part.get("arguments", {})
                assistant_turn, group_size = group or (0, 1)
                if group_size > 1:
                    execution_detail = (
                        f"Parallel batch in assistant turn {assistant_turn}: "
                        f"call {group_call_index} of {group_size}. The agent "
                        "emitted every call in this batch before seeing a result."
                    )
                else:
                    execution_detail = (
                        f"Sequential tool step {tool_step} in assistant turn "
                        f"{assistant_turn}. This turn emitted one tool call."
                    )
                cards.append(
                    _card(
                        number,
                        "tool-input",
                        f"Call {tool_name}",
                        execution_detail,
                        arguments,
                    )
                )
                number += 1
            elif part_type == "tool_result":
                tool_name = pending_tools.pop(0) if pending_tools else "tool"
                result = part.get("tool_result")
                cards.append(
                    _card(
                        number,
                        "tool-output",
                        f"Result from {tool_name}",
                        (
                            "The tool returned this value to the agent before "
                            "the next assistant turn."
                        ),
                        result,
                    )
                )
                number += 1
            elif part_type == "text":
                text = str(part.get("text") or "")
                cards.append(
                    _card(
                        number,
                        "agent-output",
                        "Final agent response",
                        text,
                        {"response": text},
                    )
                )
                number += 1

    outcome = (trace.get("evaluators") or {}).get(evaluator, {})
    result = str(outcome.get("result") or "missing")
    score = outcome.get("score")
    threshold = outcome.get("threshold")
    cards.append(
        _card(
            number,
            f"evaluation-{result}",
            f"Evaluator: {evaluator}",
            (
                f"Verdict: {result.upper()} | Score: {score} | "
                f"Threshold: {threshold}. {outcome.get('reason') or ''}"
            ),
            outcome,
        )
    )

    return f"""
<style>
.trace-shell {{
  font-family: var(--vscode-font-family, Segoe UI, sans-serif);
  color: var(--vscode-foreground, #24292f);
  max-width: 1100px;
}}
.trace-heading {{ display:flex; justify-content:space-between; gap:16px; align-items:center; }}
.trace-heading code {{ font-size:12px; }}
.execution-summary {{
  margin:10px 0 14px; padding:12px 14px; border-radius:8px;
  border:1px solid var(--vscode-panel-border, #d0d7de);
}}
.execution-summary strong {{ display:block; margin-bottom:4px; letter-spacing:.04em; }}
.execution-summary.sequential {{ border-left:5px solid #1a7f37; }}
.execution-summary.parallel {{ border-left:5px solid #cf222e; }}
.execution-summary.single {{ border-left:5px solid #0969da; }}
.execution-summary.none {{ border-left:5px solid #6e7781; }}
.trace-line {{ border-left:3px solid #6e7781; margin-left:19px; padding:4px 0 4px 25px; }}
.trace-card {{
  position:relative; display:flex; gap:14px; margin:12px 0; padding:14px;
  border:1px solid var(--vscode-panel-border, #d0d7de); border-radius:8px;
  background:var(--vscode-editor-background, #fff);
}}
.trace-index {{
  position:absolute; left:-47px; top:16px; width:34px; height:34px;
  border-radius:50%; display:grid; place-items:center; color:white;
  background:#57606a; font-weight:700;
}}
.trace-body {{ min-width:0; width:100%; }}
.trace-kind {{ font-size:11px; font-weight:700; color:#57606a; letter-spacing:.08em; }}
.trace-card h3 {{ margin:3px 0 7px; font-size:16px; }}
.trace-card p {{ margin:0 0 8px; white-space:pre-wrap; }}
.trace-card pre {{
  max-height:320px; overflow:auto; padding:12px; border-radius:6px;
  background:var(--vscode-textCodeBlock-background, #f6f8fa); white-space:pre-wrap;
}}
.tool-input {{ border-left:5px solid #0969da; }}
.tool-output {{ border-left:5px solid #8250df; }}
.agent-output {{ border-left:5px solid #1a7f37; }}
.evaluation-fail {{ border-left:5px solid #cf222e; }}
.evaluation-pass {{ border-left:5px solid #1a7f37; }}
</style>
<section class="trace-shell">
  <div class="trace-heading">
    <h2>Agent trajectory</h2>
    <code>{escape(str(trace.get("id") or ""))} / {escape(evaluator)}</code>
  </div>
  <div class="execution-summary {pattern}">
    <strong>{escape(pattern_title)}</strong>
    <span>{escape(pattern_detail)}</span>
  </div>
  <div class="trace-line">{''.join(cards)}</div>
</section>
"""

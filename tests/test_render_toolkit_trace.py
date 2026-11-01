from scripts.render_toolkit_trace import render_toolkit_trace


def test_renders_ordered_inputs_outputs_and_evaluator() -> None:
    trace = {
        "id": "case-1",
        "query": "Freeze ACC-1004",
        "response": [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_call",
                        "name": "freeze_account",
                        "arguments": '{"account_id":"ACC-1004"}',
                    }
                ],
            },
            {
                "role": "tool",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_result": '{"status":"frozen"}',
                    }
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "The account is frozen."}],
            },
        ],
        "evaluators": {
            "tool_call_accuracy": {
                "result": "fail",
                "score": 2,
                "threshold": 3,
                "reason": "Missing authorization check.",
            }
        },
    }

    html = render_toolkit_trace(trace, "tool_call_accuracy")

    assert html.index("User input") < html.index("Call freeze_account")
    assert html.index("Call freeze_account") < html.index("Result from freeze_account")
    assert html.index("Result from freeze_account") < html.index("Final agent response")
    assert "Missing authorization check." in html
    assert "evaluation-fail" in html
    assert "SINGLE TOOL CALL" in html


def test_escapes_trace_content() -> None:
    trace = {
        "id": "<case>",
        "query": "<script>alert(1)</script>",
        "response": [],
        "evaluators": {"relevance": {"result": "pass"}},
    }

    html = render_toolkit_trace(trace, "relevance")

    assert "<script>" not in html
    assert "&lt;script&gt;" in html


def test_labels_sequential_tool_calls_across_assistant_turns() -> None:
    trace = {
        "response": [
            {
                "role": "assistant",
                "content": [{"type": "tool_call", "name": "read_portfolio"}],
            },
            {
                "role": "tool",
                "content": [{"type": "tool_result", "tool_result": "{}"}],
            },
            {
                "role": "assistant",
                "content": [{"type": "tool_call", "name": "place_trade"}],
            },
            {
                "role": "tool",
                "content": [{"type": "tool_result", "tool_result": "{}"}],
            },
        ],
        "evaluators": {"tool_call_accuracy": {"result": "pass"}},
    }

    html = render_toolkit_trace(trace, "tool_call_accuracy")

    assert "SEQUENTIAL TOOL CALLS" in html
    assert "2 separate assistant turns" in html
    assert "Sequential tool step 1 in assistant turn 1" in html
    assert "Sequential tool step 2 in assistant turn 2" in html


def test_labels_calls_in_one_assistant_turn_as_parallel_batch() -> None:
    trace = {
        "response": [
            {
                "role": "assistant",
                "content": [
                    {"type": "tool_call", "name": "read_portfolio"},
                    {"type": "tool_call", "name": "place_trade"},
                ],
            },
            {
                "role": "tool",
                "content": [
                    {"type": "tool_result", "tool_result": "{}"},
                    {"type": "tool_result", "tool_result": "{}"},
                ],
            },
        ],
        "evaluators": {"tool_call_accuracy": {"result": "fail"}},
    }

    html = render_toolkit_trace(trace, "tool_call_accuracy")

    assert "BATCHED / PARALLEL TOOL CALLS" in html
    assert "assistant turn 1 emitted 2 calls" in html
    assert "Parallel batch in assistant turn 1: call 1 of 2" in html
    assert "Parallel batch in assistant turn 1: call 2 of 2" in html

"""Tests for Foundry Toolkit transient failure classification."""

from __future__ import annotations

import pytest

from bank_manager.evals.toolkit_retry import is_transient_toolkit_error


class StatusError(RuntimeError):
    def __init__(self, status_code: int) -> None:
        super().__init__(f"HTTP {status_code}")
        self.status_code = status_code


@pytest.mark.parametrize(
    "error",
    [
        ConnectionError("connection reset"),
        TimeoutError("request timed out"),
        StatusError(429),
        StatusError(503),
        RuntimeError(
            "(UserError) OpenAI API hits APIConnectionError: Connection error."
        ),
        RuntimeError(
            "Evaluator 'fluency' failed: Failed to invoke the Azure CLI"
        ),
        RuntimeError("Evaluator failed: 502 Bad Gateway"),
        RuntimeError("Evaluator failed: 500 Internal Server Error"),
    ],
)
def test_transient_toolkit_errors_are_retryable(error: BaseException) -> None:
    assert is_transient_toolkit_error(error)


def test_nested_transient_toolkit_error_is_retryable() -> None:
    try:
        raise ConnectionError("connection reset")
    except ConnectionError as cause:
        wrapped = RuntimeError("evaluator failed")
        wrapped.__cause__ = cause

    assert is_transient_toolkit_error(wrapped)


@pytest.mark.parametrize(
    "error",
    [
        StatusError(400),
        StatusError(401),
        ValueError("invalid evaluator configuration"),
        RuntimeError("content filter rejected the request"),
    ],
)
def test_non_transient_toolkit_errors_are_not_retryable(
    error: BaseException,
) -> None:
    assert not is_transient_toolkit_error(error)
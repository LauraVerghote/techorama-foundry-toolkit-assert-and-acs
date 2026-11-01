"""Compatibility exports for the standalone baseline package."""

from __future__ import annotations

import asyncio

from bank_manager.agents.baseline.agent import (
    AgentExecution,
    SYSTEM_PROMPT as SYSTEM_PROMPT_REALISTIC,
    _build_llm,
    _extract_text,
    _open_two_servers,
    capture_agent_executions,
    chat as chat_baseline,
    record_agent_execution,
    run_async as run_baseline_async,
)


def chat_cheaper(message: str) -> str:
    return asyncio.run(run_baseline_async(message, deployment="gpt-5.4-mini"))


__all__ = [
    "AgentExecution",
    "SYSTEM_PROMPT_REALISTIC",
    "_build_llm",
    "_extract_text",
    "_open_two_servers",
    "capture_agent_executions",
    "chat_baseline",
    "chat_cheaper",
    "record_agent_execution",
    "run_baseline_async",
]

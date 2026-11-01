"""Expose the package's ACS runtime through the Agent Framework response contract."""

from __future__ import annotations

import inspect
from typing import Any

from agent_framework import AgentResponse, Content, Message

from . import agent


class _AgentResponse(AgentResponse):
    def __init__(
        self,
        text: str,
        messages: list[Message],
        execution: agent.AgentExecution,
    ):
        super().__init__(messages=messages)
        self._final_text = text
        self.execution = execution

    @property
    def text(self) -> str:
        return self._final_text


def _messages(execution: agent.AgentExecution, final_text: str) -> list[Message]:
    messages = []
    for entry in execution.response_messages:
        contents = []
        for block in entry["content"]:
            if block["type"] == "text":
                contents.append(Content.from_text(block["text"]))
            elif block["type"] == "tool_call":
                call = block["tool_call"]
                contents.append(
                    Content.from_function_call(
                        call["id"],
                        call["function"]["name"],
                        arguments=call["function"]["arguments"],
                    )
                )
            elif block["type"] == "tool_result":
                contents.append(
                    Content.from_function_result(
                        entry["tool_call_id"],
                        result=block["tool_result"],
                    )
                )
            else:
                raise ValueError(
                    f"Unsupported ACS execution content: {block['type']!r}"
                )
        messages.append(Message(entry["role"], contents))

    if messages and messages[-1].role == "assistant" and all(
        content.type == "text" for content in messages[-1].contents
    ):
        messages[-1] = Message("assistant", [Content.from_text(final_text)])
    else:
        messages.append(Message("assistant", [Content.from_text(final_text)]))
    return messages


class ControlledAgent:
    """Production and evaluation interface backed by the package's ACS runtime."""

    def __init__(self, deployment: str | None = None):
        self.deployment = deployment
        self.default_options: dict[str, Any] = {"instructions": agent.SYSTEM_PROMPT}

    async def run(self, query: str) -> AgentResponse:
        try:
            with agent.capture_agent_executions() as executions:
                if self.deployment is None:
                    text = await agent.run_async(query)
                else:
                    text = await agent.run_async(query, deployment=self.deployment)
        except ExceptionGroup as error:
            def details(group: BaseExceptionGroup) -> list[str]:
                return [
                    detail
                    for child in group.exceptions
                    for detail in (
                        details(child)
                        if isinstance(child, BaseExceptionGroup)
                        else [f"{type(child).__name__}: {child}"]
                    )
                ]

            raise RuntimeError(
                "ACS agent execution failed: " + "; ".join(details(error))
            ) from error
        finally:
            for attribute in ("_foundry_credential", "_aad_credential"):
                credential = getattr(agent, attribute)
                setattr(agent, attribute, None)
                if credential is None:
                    continue
                closed = credential.close()
                if inspect.isawaitable(closed):
                    await closed

        if len(executions) != 1:
            raise RuntimeError(
                f"Expected one ACS execution, recorded {len(executions)}"
            )
        execution = executions[0]
        if not execution.tool_definitions:
            raise RuntimeError("ACS execution did not record tool schemas")
        self.default_options["tools"] = execution.tool_definitions
        return _AgentResponse(text, _messages(execution, text), execution)

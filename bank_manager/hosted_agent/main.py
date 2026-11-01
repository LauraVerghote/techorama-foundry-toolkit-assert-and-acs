"""Microsoft Foundry BYO hosted agent using the Responses protocol."""

from __future__ import annotations

import asyncio
import json
import os
from collections.abc import AsyncIterator

from azure.ai.agentserver.responses import (
    CreateResponse,
    ResponseContext,
    ResponseEventStream,
    ResponsesAgentServerHost,
)

from bank_manager.agents.assert_optimized_instructions_acs.agent import (
    AgentExecution,
    run_with_execution,
)

app = ResponsesAgentServerHost()


class _EvaluationTraceResponse:
    def __init__(
        self,
        context: ResponseContext,
        request: CreateResponse,
        execution: AgentExecution,
    ) -> None:
        self._context = context
        self._request = request
        self._execution = execution

    def __aiter__(self) -> AsyncIterator[object]:
        return self._generate()

    async def _generate(self) -> AsyncIterator[object]:
        stream = ResponseEventStream(
            response_id=self._context.response_id,
            request=self._request,
        )
        stream.response["metadata"] = {
            "bank_manager_system_prompt": self._execution.system_prompt,
        }
        yield stream.emit_created()
        yield stream.emit_in_progress()

        for message in self._execution.response_messages:
            if message["role"] == "assistant":
                for part in message["content"]:
                    if part["type"] != "tool_call":
                        continue
                    tool_call = part["tool_call"]
                    function = tool_call["function"]
                    for event in stream.output_item_function_call(
                        name=function["name"],
                        call_id=tool_call["id"],
                        arguments=json.dumps(function["arguments"], ensure_ascii=True),
                    ):
                        yield event
            elif message["role"] == "tool":
                for part in message["content"]:
                    if part["type"] != "tool_result":
                        continue
                    for event in stream.output_item_function_call_output(
                        call_id=message["tool_call_id"],
                        output=part["tool_result"],
                    ):
                        yield event

        for event in stream.output_item_message(self._execution.text):
            yield event
        yield stream.emit_completed()


@app.response_handler
async def handle_response(
    request: CreateResponse,
    context: ResponseContext,
    cancellation_signal: asyncio.Event,
) -> _EvaluationTraceResponse:
    """Run the locally validated prompt-plus-ACS agent after final promotion."""
    del cancellation_signal
    message = await context.get_input_text() or ""
    _, execution = await run_with_execution(
        message,
        deployment=os.environ.get("AZURE_AI_MODEL_DEPLOYMENT_NAME") or None,
    )
    return _EvaluationTraceResponse(context, request, execution)


def main() -> None:
    app.run()


if __name__ == "__main__":
    main()

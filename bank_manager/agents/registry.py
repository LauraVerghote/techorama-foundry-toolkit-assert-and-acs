"""Resolve Bank Manager agent revision packages by name."""

from __future__ import annotations

import importlib
import re
from dataclasses import dataclass


AGENT_REVISIONS = (
    "baseline",
    "assert_optimized_instructions",
    "assert_optimized_instructions_acs",
    "cheaper_model",
)
_REVISION_NAME = re.compile(r"^[a-z][a-z0-9_]*$")


@dataclass(frozen=True)
class AgentMetadata:
    name: str
    instructions: str
    tool_descriptions: dict[str, str]
    controls: str
    default_deployment: str | None


def get_agent_revision(name: str):
    """Read metadata from one explicitly named standalone agent package."""
    if not _REVISION_NAME.fullmatch(name):
        raise ValueError(
            "Bank Manager revision names must contain only lowercase letters, "
            "numbers, and underscores."
        )
    try:
        module = importlib.import_module(f"bank_manager.agents.{name}.agent")
    except ModuleNotFoundError as exc:
        raise ValueError(
            f"Unknown Bank Manager revision {name!r}. "
            "Create its package under bank_manager/agents or choose one of: "
            f"{', '.join(AGENT_REVISIONS)}"
        ) from exc
    if module.AGENT_NAME != name:
        raise ValueError(
            f"Agent folder {name!r} declares name {module.AGENT_NAME!r}."
        )
    return AgentMetadata(
        name=module.AGENT_NAME,
        instructions=module.SYSTEM_PROMPT,
        tool_descriptions=dict(module.TOOL_DESCRIPTIONS),
        controls=module.CONTROLS,
        default_deployment=module.DEFAULT_DEPLOYMENT,
    )

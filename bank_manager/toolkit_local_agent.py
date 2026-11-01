"""Compatibility adapter for revision-packaged Toolkit agents."""

import importlib

from bank_manager.agents.baseline.agent import TOOL_DESCRIPTIONS


def build_bank_manager_agent(
    module_path: str = "bank_manager.agents.baseline.agent",
):
    """Build one explicitly selected standalone agent module."""
    return importlib.import_module(module_path).create_agent()


__all__ = [
    "TOOL_DESCRIPTIONS",
    "build_bank_manager_agent",
]

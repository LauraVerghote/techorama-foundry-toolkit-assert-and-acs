"""Compatibility entry points for standalone Bank Manager packages."""

from __future__ import annotations

import asyncio

from bank_manager.agents.assert_optimized_instructions_acs.agent import (
    chat as chat_acs,
    chat as chat_prompted_acs,
    run_async as run_acs_async,
    run_async as run_prompted_acs_async,
)
from bank_manager.agents.assert_optimized_instructions.agent import (
    chat as chat_prompted,
    run_async as run_prompted_async,
)
from bank_manager.agents.cheaper_model.agent import (
    chat as chat_acs_alternate,
    chat as chat_prompted_acs_alternate,
)


__all__ = [
    "chat_acs",
    "chat_acs_alternate",
    "chat_prompted",
    "chat_prompted_acs",
    "chat_prompted_acs_alternate",
    "run_acs_async",
    "run_prompted_async",
    "run_prompted_acs_async",
]

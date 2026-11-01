"""Shared Entra/AAD auth for a key-auth-disabled Foundry endpoint.

Activated when ``ASSERT_AZURE_USE_AAD=1``. Uses ``DefaultAzureCredential`` so
local Azure CLI credentials and Azure managed identities share one code path.
"""
from __future__ import annotations

import functools
import os

SCOPE = "https://cognitiveservices.azure.com/.default"


def use_aad() -> bool:
    return os.environ.get("ASSERT_AZURE_USE_AAD", "").strip().lower() in ("1", "true", "yes")


@functools.lru_cache(maxsize=1)
def _credential():
    from azure.identity import DefaultAzureCredential
    return DefaultAzureCredential()


@functools.lru_cache(maxsize=1)
def get_provider():
    """0-arg callable returning a cached, auto-refreshed bearer token."""
    from azure.identity import get_bearer_token_provider
    return get_bearer_token_provider(_credential(), SCOPE)


def get_token() -> str:
    return _credential().get_token(SCOPE).token

"""Retry classification for transient Foundry Toolkit evaluation failures."""

from __future__ import annotations

from collections.abc import Iterator


_TRANSIENT_STATUS_CODES = {408, 409, 429}
_TRANSIENT_ERROR_NAMES = {
    "APIConnectionError",
    "APITimeoutError",
    "ConnectError",
    "ConnectTimeout",
    "ConnectionError",
    "PoolTimeout",
    "ReadError",
    "ReadTimeout",
    "ServiceRequestError",
    "ServiceResponseError",
    "TimeoutError",
}
_TRANSIENT_MESSAGE_MARKERS = (
    "api connection error",
    "apiconnectionerror",
    "api timeout error",
    "apitimeouterror",
    "bad gateway",
    "connection error",
    "connection reset",
    "connection timed out",
    "failed to invoke the azure cli",
    "gateway timeout",
    "internal server error",
    "rate limit",
    "request timeout",
    "service unavailable",
    "temporarily unavailable",
    "too many requests",
)


def _error_chain(error: BaseException) -> Iterator[BaseException]:
    """Yield an exception and its explicit or implicit causes once each."""
    seen: set[int] = set()
    current: BaseException | None = error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        yield current
        current = current.__cause__ or current.__context__


def is_transient_toolkit_error(error: BaseException | None) -> bool:
    """Return whether an evaluator failure is safe to retry."""
    if error is None:
        return False

    for current in _error_chain(error):
        status_code = getattr(current, "status_code", None)
        if status_code in _TRANSIENT_STATUS_CODES or (
            isinstance(status_code, int) and 500 <= status_code <= 599
        ):
            return True

        if type(current).__name__ in _TRANSIENT_ERROR_NAMES:
            return True

        message = str(current).lower()
        if any(marker in message for marker in _TRANSIENT_MESSAGE_MARKERS):
            return True

    return False
"""Classification of LLM API failures into transient (retry) vs. permanent.

Transient: connection drops, timeouts, rate limits (429), server errors (5xx),
empty provider responses. Permanent: bad requests, authentication, insufficient
balance, unknown model, … — retrying those only wastes time and money.
"""

from __future__ import annotations

import asyncio

import openai


class EmptyResponseError(RuntimeError):
    """The provider returned a response without choices or message content."""


# HTTP status codes worth retrying (plus every 5xx server error).
TRANSIENT_STATUS_CODES = frozenset({408, 409, 425, 429})


def _is_transient_status(status: int) -> bool:
    return status in TRANSIENT_STATUS_CODES or status >= 500


# Message fragments of transient failures, for exceptions that carry no type
# or status information (e.g. errors re-raised by wrappers).
_TRANSIENT_INDICATORS = (
    "rate limit",
    "timeout",
    "timed out",
    "connection",
    "disconnected",
    "network",
    "service unavailable",
    "temporarily unavailable",
    "gateway",
    "503",
    "502",
    "504",
    "429",
    "overloaded",
)


def _status_code(error: BaseException) -> int | None:
    status = getattr(error, "status_code", None)
    return status if isinstance(status, int) else None


def is_transient_error(error: BaseException) -> bool:
    """Return True if ``error`` is a temporary LLM transport/server failure."""
    if isinstance(error, (openai.APITimeoutError, openai.APIConnectionError)):
        return True
    if isinstance(error, (EmptyResponseError, asyncio.TimeoutError, TimeoutError, ConnectionError)):
        return True
    if isinstance(error, openai.APIStatusError):
        # The HTTP status is authoritative: 4xx other than 408/409/425/429
        # (auth, bad request, insufficient balance, unknown model) never heal.
        return _is_transient_status(error.status_code)
    try:
        import httpx
        if isinstance(error, httpx.TransportError):
            return True
    except ImportError:  # pragma: no cover - httpx ships with openai
        pass

    # Wrappers (e.g. Instructor) may re-raise; inspect the explicit cause.
    cause = error.__cause__
    if cause is not None and cause is not error:
        if isinstance(cause, (openai.APIError, EmptyResponseError)):
            return is_transient_error(cause)

    status = _status_code(error)
    if status is not None:
        return _is_transient_status(status)

    text = str(error).lower()
    return any(indicator in text for indicator in _TRANSIENT_INDICATORS)


def retry_after_seconds(error: BaseException) -> float | None:
    """Return the server-requested delay (Retry-After header), if any."""
    response = getattr(error, "response", None)
    headers = getattr(response, "headers", None)
    if not headers:
        return None
    value = headers.get("retry-after")
    if value is None:
        return None
    try:
        return max(0.0, float(value))
    except (TypeError, ValueError):
        return None


def backoff_delay(attempt: int, error: BaseException | None = None,
                  base: float = 1.0, cap: float = 60.0) -> float:
    """Bounded exponential backoff: base * 2**attempt, capped, Retry-After aware."""
    delay = min(cap, base * (2 ** attempt))
    if error is not None:
        requested = retry_after_seconds(error)
        if requested is not None:
            delay = min(cap, max(delay, requested))
    return delay

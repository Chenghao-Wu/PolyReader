"""Shared transport-error classification.

Imported by both extraction_loop (for retry decisions on errors raised by
extract_entity) and guided_extraction / tools (for deciding whether to
re-raise an LLM call failure).
"""

from __future__ import annotations

from polyreader.llm.errors import is_transient_error


def is_transport_error(error: BaseException) -> bool:
    """Return True if `error` is a transient LLM transport failure.

    Uses the exception type and HTTP status first (connection errors,
    timeouts, 429 and 5xx are transient; 400/401/402/403/404/422 are not),
    then falls back to matching the message for untyped errors.
    """
    return is_transient_error(error)

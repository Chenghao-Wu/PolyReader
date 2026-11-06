"""Token counting via tiktoken with fallback."""

import tiktoken


def _get_encoding(model: str = None):
    try:
        return tiktoken.encoding_for_model(model or "gpt-4o")
    except (KeyError, Exception):
        try:
            return tiktoken.get_encoding("cl100k_base")
        except Exception:
            return None


def count_tokens(text: str, model: str = None) -> int:
    """Count tokens using tiktoken, fallback to len(text)//4."""
    if not text:
        return 0
    enc = _get_encoding(model)
    if enc is None:
        return len(text) // 4
    return len(enc.encode(text))

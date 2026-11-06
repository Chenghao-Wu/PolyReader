"""JSON extraction from LLM output, cleaning."""

import json
import logging
import re

logger = logging.getLogger(__name__)


def get_json_content(response: str) -> str:
    """Strip markdown code fences and return raw JSON string."""
    # Match ```json or ``` (case-insensitive), non-greedy content, closing ```
    match = re.search(r"```(?:json)?\s*\r?\n([\s\S]*?)```", response, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return response.strip()


def _find_balanced_json(text: str) -> list[str]:
    """Find balanced JSON object/array substrings scanning left to right.

    Uses a bracket-counting parser that respects string literals,
    so nested braces inside strings or nested objects are handled correctly.
    Returns candidates in left-to-right order (outermost structures first).
    """
    candidates = []
    openers = {"{": "}", "[": "]"}
    i = 0
    while i < len(text):
        if text[i] in openers:
            stack = []
            in_string = False
            escape = False
            for j in range(i, len(text)):
                ch = text[j]
                if escape:
                    escape = False
                elif ch == "\\" and in_string:
                    escape = True
                elif ch == '"':
                    in_string = not in_string
                elif not in_string:
                    if ch in openers:
                        stack.append(openers[ch])
                    elif stack and ch == stack[-1]:
                        stack.pop()
                        if not stack:
                            candidates.append(text[i : j + 1])
                            i = j  # skip past this candidate
                            break
                    elif ch in openers.values():
                        break  # mismatched bracket — abandon
            # If loop ends without break, no balanced match from position i
        i += 1
    return candidates


def extract_json(content: str, *, quiet: bool = False) -> dict | list:
    """Extract and parse JSON from LLM output.

    Handles:
    - ```json fences
    - Python None -> JSON null
    - Trailing commas
    - Whitespace normalization
    """
    try:
        # Already parsed — return as-is
        if isinstance(content, (dict, list)):
            return content

        # Strip json fences
        json_content = get_json_content(content)
        if not json_content:
            json_content = content.strip()

        # Clean Python literals -> JSON (word-boundary to avoid corrupting data)
        json_content = re.sub(r'\bNone\b', 'null', json_content)
        json_content = re.sub(r'\bTrue\b', 'true', json_content)
        json_content = re.sub(r'\bFalse\b', 'false', json_content)

        # Try direct parse first
        try:
            return json.loads(json_content)
        except json.JSONDecodeError:
            pass

        # Remove trailing commas
        json_content = re.sub(r",\s*([}\]])", r"\1", json_content)

        # Normalize whitespace (but preserve strings)
        try:
            return json.loads(json_content)
        except json.JSONDecodeError:
            pass

        # Last resort: find balanced JSON blocks using brace/bracket counting
        for candidate in _find_balanced_json(json_content):
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                continue

        if quiet:
            logger.debug(f"Failed to extract JSON from: {content[:200]}...")
        else:
            logger.error(f"Failed to extract JSON from: {content[:200]}...")
        return {}

    except Exception as e:
        if quiet:
            logger.debug(f"Unexpected error extracting JSON: {e}")
        else:
            logger.error(f"Unexpected error extracting JSON: {e}")
        return {}


def write_json_atomic(path, data, *, indent: int = 2, ensure_ascii: bool = False) -> None:
    """Write JSON to ``path`` atomically (temp file in the same dir + rename).

    An interrupted write leaves either the previous file or the new one,
    never a truncated file.
    """
    import os
    import tempfile
    from pathlib import Path

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=indent, ensure_ascii=ensure_ascii)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise

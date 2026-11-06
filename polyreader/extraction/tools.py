"""Shared tool definitions and dispatch for document navigation.

Both entity_discovery and guided_extraction_agent use the same 3 tools
(get_page_content, get_page_image, get_document_structure) with identical
dispatch logic. This module provides a single source of truth.
"""

from __future__ import annotations

import json
import logging

from polyreader.llm.client import LLMClient
from polyreader.llm.providers import DEEPSEEK, deepseek_supports_vision, detect_provider
from polyreader.utils.json_utils import extract_json

from .config import ExtractionConfig
from .document_set import DocumentSet, UnknownDocumentError
from ._transport import is_transport_error

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tool definitions (OpenAI function-calling schema)
# ---------------------------------------------------------------------------

NAVIGATION_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_page_content",
            "description": (
                "Retrieve the parsed text content for one or more pages. "
                "Use page ranges from the document structure tree. "
                "This is your PRIMARY tool — prioritize reading: "
                "Abstract (polymer identity/abbreviations), "
                "Experimental (synthesis conditions), "
                "Results/Tables (all characterization: Mn, Mw, PDI, thermal, structural, mechanical, rheological)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "doc": {
                        "type": "string",
                        "description": (
                            "Document label. Leave empty for the primary document."
                        ),
                    },
                    "pages": {
                        "type": "string",
                        "description": 'Page range string, e.g. "1", "1-3", "1,3,5".',
                    },
                },
                "required": ["pages"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_page_image",
            "description": (
                "Retrieve a specific extracted image from the document. "
                "Image paths appear in page text as ![](img_path). "
                "Use this to view figures, tables, and diagrams that need visual inspection."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "doc": {
                        "type": "string",
                        "description": (
                            "Document label. Leave empty for the primary document."
                        ),
                    },
                    "img_path": {
                        "type": "string",
                        "description": (
                            "Image filename from page text, e.g. '_page_16_Picture_1.jpeg'. "
                            "Found in page content as ![](img_path)."
                        ),
                    },
                },
                "required": ["img_path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_document_structure",
            "description": (
                "Return the document's section tree (titles, summaries, page ranges). "
                "Use this to understand the paper structure and locate relevant sections."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "doc": {
                        "type": "string",
                        "description": (
                            "Document label. Leave empty for the primary document."
                        ),
                    },
                },
            },
        },
    },
]


TOOL_NAMES = tuple(t["function"]["name"] for t in NAVIGATION_TOOLS)


def parse_tool_arguments(raw) -> tuple[dict, str | None]:
    """Parse a tool call's JSON arguments.

    Returns ``(arguments, error)``. On malformed input ``arguments`` is
    empty and ``error`` describes the problem, so the caller can report it
    back to the LLM instead of aborting the conversation.
    """
    if isinstance(raw, dict):
        return raw, None
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        return {}, None
    if not isinstance(raw, str):
        return {}, f"Tool arguments must be a JSON object, got {type(raw).__name__}"
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as e:
        # Lenient fallback: fences, trailing commas, extra text around the object.
        parsed = extract_json(raw, quiet=True)
        if not isinstance(parsed, dict) or not parsed:
            return {}, f"Invalid JSON in tool arguments ({e.msg} at position {e.pos})"
    if not isinstance(parsed, dict):
        return {}, f"Tool arguments must be a JSON object, got {type(parsed).__name__}"
    return parsed, None


def tool_error_result(message: str) -> str:
    """JSON tool result reporting an error back to the LLM."""
    return json.dumps({"success": False, "error": message}, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Tool call summary helper (used by agent modules)
# ---------------------------------------------------------------------------


def build_navigation_tools(doc_set: DocumentSet) -> list[dict]:
    """Build navigation tool list, making ``doc`` required for multi-doc sets.

    When the DocumentSet contains multiple documents, injects the actual
    available labels into the ``doc`` parameter description and adds ``"doc"``
    to the ``required`` list for every tool. For single-doc sets the tools
    are returned unchanged.
    """
    import copy

    if not doc_set.has_multiple_documents():
        return NAVIGATION_TOOLS

    labels = doc_set.labels()
    labels_str = ", ".join(f'"{l}"' for l in labels)
    doc_description = (
        f"Which document to read. Available: {labels_str}. You MUST specify one."
    )

    tools = copy.deepcopy(NAVIGATION_TOOLS)
    for tool in tools:
        params = tool["function"]["parameters"]
        if "doc" in params.get("properties", {}):
            params["properties"]["doc"]["description"] = doc_description
            required = params.get("required", [])
            if "doc" not in required:
                required.append("doc")
            params["required"] = required
    return tools


def _summarize_tool_args(name: str, arguments: dict) -> str:
    """Render tool args as a ``k=v, k=v`` string for logging.

    Omits empty values. For ``get_page_content`` with no explicit ``doc``, leaves
    the args empty rather than inventing a ``(primary)`` marker — the caller's
    context already tells you which document set is in play.
    """
    return ", ".join(f"{k}={v}" for k, v in arguments.items() if v)


def summarize_tool_calls(tool_calls: list) -> str:
    """Produce a concise one-line summary of tool calls in an iteration.

    Each tool call is represented as ``name(key=val, ...)``.  Designed for
    structured log lines in entity_discovery and guided_extraction_agent.

    Args:
        tool_calls: List of tool-call objects with ``.name`` and ``.arguments``
            (JSON string) attributes.

    Returns:
        Semicolon-separated summary string, e.g.
        ``content(pages=1-3); image(img_path=fig1.png)``
    """
    parts: list[str] = []
    for tc in tool_calls:
        try:
            args = json.loads(tc.arguments) if isinstance(tc.arguments, str) else tc.arguments
        except (json.JSONDecodeError, TypeError):
            args = {}

        short_name = tc.name.replace("get_page_", "").replace("get_document_", "")
        kv = _summarize_tool_args(tc.name, args)
        if not kv and tc.name == "get_page_content":
            kv = "doc=(primary)"
        parts.append(f"{short_name}({kv})")
    return "; ".join(parts) if parts else "(none)"


def _summarize_tool_result(
    name: str,
    arguments: dict,
    result_str: str,
    image_data_url: str | None,
) -> str:
    """Render a short description of a tool's return value for logging.

    Returns the right-hand side of the arrow in lines like
    ``get_page_content(pages=1-3) → text, ~456 tokens``.
    """
    try:
        result = json.loads(result_str)
    except (json.JSONDecodeError, TypeError):
        return "unparseable result"

    if isinstance(result, dict) and "error" in result:
        return f"error: {result['error']}"

    if name == "get_page_content":
        tokens = result.get("approximate_tokens", 0) if isinstance(result, dict) else 0
        return f"text, ~{tokens} tokens"

    if name == "get_page_image":
        if image_data_url is not None:
            return "image returned"
        if isinstance(result, dict) and result.get("success"):
            return "image available (vision off)"
        return "image unavailable"

    if name == "get_document_structure":
        if isinstance(result, list):
            return f"{len(result)} sections"
        return "structure returned"

    return "ok"


# ---------------------------------------------------------------------------
# Tool dispatch
# ---------------------------------------------------------------------------


# Map file suffixes to IANA image MIME subtypes.
_IMAGE_MIME_SUBTYPES = {
    "jpg": "jpeg",
    "jpeg": "jpeg",
    "png": "png",
    "gif": "gif",
    "webp": "webp",
    "bmp": "bmp",
    "tif": "tiff",
    "tiff": "tiff",
}


def _image_data_url(b64: str, fmt: str | None) -> str:
    """Build a correctly-typed data URL for a base64-encoded image."""
    subtype = _IMAGE_MIME_SUBTYPES.get((fmt or "").lower(), "png")
    return f"data:image/{subtype};base64,{b64}"


def dispatch_tool(
    doc_set: DocumentSet,
    name: str,
    arguments: dict,
    *,
    vision_enabled: bool = False,
    log_context: str = "",
) -> tuple[str, str | None]:
    """Route a tool call to the corresponding PaperIndex method.

    Uses the ``doc`` argument to resolve the target document within the set.

    Args:
        doc_set: DocumentSet to resolve against.
        name: Tool name (``get_page_content``/``get_page_image``/``get_document_structure``).
        arguments: Parsed tool-call arguments.
        vision_enabled: If True and the tool is ``get_page_image``, return a
            data URL for the image separately from the result JSON.
        log_context: Optional short string identifying the caller (e.g.
            ``"entity=poly_123 iter=3"`` or ``"stage=discovery iter=2"``).
            Rendered as a ``[…] `` prefix on the emitted INFO log line. Omit
            for no prefix.

    Returns:
        ``(result_json, image_data_url | None)``. The image URL is only set
        for ``get_page_image`` calls when ``vision_enabled`` is True and the
        image was found.
    """
    if not isinstance(arguments, dict):
        arguments = {}
    doc_label = arguments.get("doc") or arguments.get("tree")
    image_data_url: str | None = None

    # In multi-document sets an unknown label for a page/image read is an
    # error reported back to the LLM, never a silent read of the primary
    # document. (get_document_structure returns every document anyway.)
    strict = doc_set.has_multiple_documents() and name != "get_document_structure"
    try:
        index = doc_set.resolve(doc_label, strict=strict)
    except UnknownDocumentError as e:
        index = None
        result = {"success": False, "error": str(e), "available_documents": e.available}

    try:
        result_obj = _run_tool(doc_set, index, name, arguments, vision_enabled) if index is not None else None
    except Exception as e:  # a failing tool must not end the conversation
        logger.warning(f"Tool {name} failed: {type(e).__name__}: {e}")
        result_obj = ({"success": False, "error": f"Tool {name} failed: {type(e).__name__}: {e}"}, None)
    if result_obj is not None:
        result, image_data_url = result_obj

    result_json = json.dumps(result, ensure_ascii=False)

    prefix = f"[{log_context}] " if log_context else ""
    args_str = _summarize_tool_args(name, arguments)
    result_summary = _summarize_tool_result(name, arguments, result_json, image_data_url)
    logger.info(f"{prefix}{name}({args_str}) → {result_summary}")

    return result_json, image_data_url


def _run_tool(doc_set, index, name: str, arguments: dict, vision_enabled: bool):
    """Execute one navigation tool; returns ``(result, image_data_url)``."""
    image_data_url: str | None = None

    if name == "get_page_content":
        pages_requested = arguments.get("pages", "")
        result = index.get_page_content(pages=pages_requested)
    elif name == "get_page_image":
        img_path = arguments.get("img_path", "unknown")
        result = index.get_page_image(img_path=img_path)
        if result.get("success") and "image_base64" in result:
            raw_b64 = result["image_base64"]
            img_fmt = result.get("format")
            if vision_enabled:
                image_data_url = _image_data_url(raw_b64, img_fmt)
                result = {
                    "success": True,
                    "img_path": result["img_path"],
                    "image_rendered": True,
                    "note": "Image delivered as visual context below.",
                }
            else:
                result = {
                    "success": True,
                    "img_path": result["img_path"],
                    "image_rendered": False,
                    "image_bytes_length": len(raw_b64),
                    "note": (
                        "Vision not available for this model. The image exists "
                        "but cannot be shown. Use get_page_content to read the "
                        "surrounding text instead."
                    ),
                }
    elif name == "get_document_structure":
        if doc_set.has_multiple_documents():
            result = doc_set.get_combined_structure()
        else:
            result = index.get_document_structure()
    else:
        logger.warning(f"Unknown tool: {name}")
        result = {"error": f"Unknown tool: {name}", "available_tools": list(TOOL_NAMES)}

    return result, image_data_url


# ---------------------------------------------------------------------------
# Document structure formatting
# ---------------------------------------------------------------------------


def format_doc_structure_json(doc_set: DocumentSet) -> str:
    """Build a JSON string of the document structure for embedding in prompts.

    For multi-doc sets, returns per-document sections with labels and page counts.
    For single-doc sets, returns the primary document's structure.
    """
    if doc_set.has_multiple_documents():
        parts = []
        for label, idx in doc_set:
            struct = idx.get_document_structure()
            struct_json = json.dumps(struct, indent=2, ensure_ascii=False)
            page_count = len(idx.page_texts) if idx.page_texts else 0
            parts.append(
                f'### Document "{label}": {idx.doc_name} ({page_count} pages)\n{struct_json}'
            )
        return "\n\n".join(parts)
    else:
        structure = doc_set.primary.get_document_structure()
        return json.dumps(structure, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def supports_vision(config: ExtractionConfig) -> bool:
    """Check whether vision is enabled for this run.

    With OpenRouter as the gateway, the user is responsible for picking a
    vision-capable model (e.g. google/gemini-2.5-pro, anthropic/claude-sonnet-4,
    openai/gpt-4o). We honor config.vision_enabled as the source of truth,
    except on the DeepSeek API where only deepseek-flash accepts images.
    """
    if not config.vision_enabled:
        return False
    if detect_provider(config.base_url) == DEEPSEEK and not deepseek_supports_vision(config.model):
        logger.warning(
            f"Vision disabled: DeepSeek model '{config.model}' does not accept images "
            "(use deepseek-flash)"
        )
        return False
    return True


async def wrap_up_conversation(
    llm: LLMClient, messages: list[dict], tools: list[dict],
) -> str:
    """Force LLM to produce a final answer when max iterations are reached.

    Appends a wrap-up instruction and calls with tool_choice="none" so the
    LLM synthesizes everything gathered so far.
    """
    messages_copy = messages + [{
        "role": "user",
        "content": (
            "You are out of iterations. Based on ALL the information you have "
            "gathered, provide your final answer NOW. Do NOT make any more tool calls. "
            "Synthesize everything into the required JSON output format."
        ),
    }]
    try:
        result = await llm.chat_with_tools(
            messages=messages_copy,
            tools=tools,
            tool_choice="none",
        )
        return result.content
    except Exception as e:
        if is_transport_error(e):
            logger.warning(f"  Wrap-up transport error: {e}")
            raise
        logger.error(f"Wrap-up failed: {e}")
        return ""

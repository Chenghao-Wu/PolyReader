"""Per-entity guided extraction agent — guided_navigation style.

Extracts all data for a single polymer entity using tool-calling loop.
Standalone functions following entity_discovery.py pattern — no class inheritance.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from polyreader.llm.client import LLMClient, strip_tool_messages
from polyreader.utils.json_utils import extract_json

from .config import ExtractionConfig
from .document_set import DocumentSet
from .prompts import (
    build_guided_extraction_prompt,
    build_guided_extraction_task,
)
from polyreader.schema.polymer import PolymerSample
from .tools import (
    build_navigation_tools, dispatch_tool, parse_tool_arguments, summarize_tool_calls,
    supports_vision, tool_error_result, wrap_up_conversation,
)
from ._transport import is_transport_error

logger = logging.getLogger(__name__)



# ---------------------------------------------------------------------------
# Answer parsing
# ---------------------------------------------------------------------------


def parse_extraction_answer(answer: str, entity_id: str) -> dict[str, Any]:
    """Parse extraction result from the LLM's final answer.

    Returns dict suitable for PolymerSample.model_validate() or {"skipped": True}.
    Backfills missing fields with defaults.
    """
    parsed = extract_json(answer, quiet=True)

    if not isinstance(parsed, dict):
        return {"sample_id": "", "polymer_id": entity_id, "raw": answer}

    # Check if entity was triaged out
    if parsed.get("skipped"):
        return {"skipped": True, "reason": parsed.get("reason", "Entity triaged out")}

    # Ensure minimal identity fields; CoercingBaseModel handles the rest
    parsed.setdefault("abbreviation", entity_id)

    return parsed


# ---------------------------------------------------------------------------
# Structured output extraction (Instructor)
# ---------------------------------------------------------------------------


async def _extract_with_structured_output(
    messages: list[dict],
    llm: LLMClient,
    entity_id: str,
    is_skipped: bool = False,
) -> PolymerSample | dict[str, Any]:
    """Extract entity using Instructor structured output.

    Uses the conversation history to produce a validated PolymerSample.
    Instructor handles validation and automatic retry on errors.

    Args:
        messages: Conversation history from tool-calling phase.
        llm: LLM client with chat_structured support.
        entity_id: The entity being extracted.
        is_skipped: If True, the entity was triaged out during tool-calling.

    Returns:
        Validated PolymerSample, or {"skipped": True} dict for triaged entities.
    """
    if is_skipped:
        return {"skipped": True, "reason": "Entity triaged out during extraction"}

    # Add final prompt for structured extraction
    extraction_messages = messages + [
        {
            "role": "user",
            "content": (
                "Based on all the research you have done, provide the final extraction "
                f"for polymer entity '{entity_id}'. Return a complete, validated data structure."
            ),
        }
    ]

    # Structured output in thinking mode is requested without tools, so the
    # tool-protocol history is flattened to plain messages.
    if llm._is_reasoner_model() or llm._uses_thinking_mode():
        extraction_messages = strip_tool_messages(extraction_messages)

    try:
        sample = await llm.chat_structured(
            response_model=PolymerSample,
            messages=extraction_messages,
            max_retries=3,  # Instructor auto-retries on validation errors
        )
        return sample
    except Exception as e:
        if is_transport_error(e):
            logger.warning(
                f"  [guided_extraction:{entity_id}] structured-output transport error: {e}"
            )
            raise
        logger.error(f"Structured extraction failed for '{entity_id}': {e}")
        # Fallback: return error dict for retry logic in extraction_loop
        return {"sample_id": "", "polymer_id": entity_id, "error": str(e)}


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


async def extract_entity(
    doc_set: DocumentSet,
    llm: LLMClient,
    entity_id: str,
    entity_info: dict[str, Any],
    entity_map: dict[str, Any],
    config: ExtractionConfig,
    base_samples_context: dict[str, Any] | None = None,
    error_history: list[dict[str, Any]] | None = None,
    use_structured_output: bool = True,
    target_sample_id: str | None = None,
    is_multi_sample: bool = False,
) -> dict[str, Any]:
    """Extract all data for one polymer entity in a single conversation.

    Follows entity_discovery.py:discover_entities() loop structure. Builds a
    conversation with the LLM using tool calls until a final answer is produced.

    Args:
        doc_set: DocumentSet with one or more indexed documents.
        llm: LLM client for API calls.
        entity_id: The abbreviation/name of the entity to extract.
        entity_info: Entity metadata from entity_map.
        entity_map: Full entity map for context.
        config: Extraction configuration.
        base_samples_context: For derived entities, extracted data from base polymers.
        error_history: Previous validation errors for retry context.
        use_structured_output: If True, use Instructor for validated output.

    Returns:
        Dict suitable for PolymerSample.model_validate(), or {"skipped": True}.
    """
    doc_set.ensure_pages_loaded()

    system_prompt = build_guided_extraction_prompt(
        doc_set=doc_set,
        entity_id=entity_id,
        entity_info=entity_info,
        entity_map=entity_map,
        base_samples_context=base_samples_context,
        error_history=error_history,
        target_sample_id=target_sample_id,
        is_multi_sample=is_multi_sample,
    )
    task_prompt = build_guided_extraction_task(
        entity_id,
        target_sample_id=target_sample_id,
    )

    messages: list[dict] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": task_prompt},
    ]

    nav_tools = build_navigation_tools(doc_set)
    tools = nav_tools if config.vision_enabled else [
        t for t in nav_tools if t["function"]["name"] != "get_page_image"
    ]

    use_vision = supports_vision(config)

    final_answer = None
    is_skipped = False
    max_iterations = config.guided_max_iterations if hasattr(config, 'guided_max_iterations') else 12

    logger.info(
        f"  [guided_extraction:{entity_id}] starting "
        f"(docs={doc_set.labels()}, max_iter={max_iterations}, "
        f"vision={use_vision})"
    )

    for iteration in range(max_iterations):
        logger.info(f"  [guided_extraction:{entity_id}] iteration {iteration + 1}/{max_iterations}")

        try:
            result = await llm.chat_with_tools(
                messages=messages,
                tools=tools,
                tool_choice="auto",
            )
        except Exception as e:
            if is_transport_error(e):
                logger.warning(
                    f"  [guided_extraction:{entity_id}] transport error at iter {iteration + 1}: {e}"
                )
                raise
            logger.error(f"Guided extraction LLM call failed: {e}")
            break

        messages.append(result.raw_message_dict)

        # No tool calls = final answer
        if not result.tool_calls:
            final_answer = result.content
            # Check if entity was triaged out from the answer
            if final_answer:
                parsed = extract_json(final_answer, quiet=True)
                if isinstance(parsed, dict) and parsed.get("skipped"):
                    is_skipped = True
            break

        # Dispatch tool calls
        pending_images: list[tuple[str, str]] = []
        for tc in result.tool_calls:
            args, arg_error = parse_tool_arguments(tc.arguments)
            if arg_error:
                logger.warning(f"  [guided_extraction:{entity_id}] {tc.name}: {arg_error}")
                result_str = tool_error_result(f"{arg_error}. Call the tool again with valid JSON arguments.")
                image_data_url = None
            else:
                result_str, image_data_url = dispatch_tool(
                    doc_set, tc.name, args,
                    vision_enabled=use_vision,
                    log_context=f"entity={entity_id} iter={iteration + 1}",
                )
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result_str,
                }
            )
            if image_data_url is not None:
                img_path = args.get("img_path", "unknown")
                pending_images.append((img_path, image_data_url))

        logger.info(
            f"  [guided_extraction:{entity_id}] tools: "
            f"{summarize_tool_calls(result.tool_calls)}"
        )

        # Inject image user message after all tool responses
        if pending_images:
            parts: list[dict] = [{"type": "text", "text": "Here are the extracted images you requested:"}]
            for img_path, data_url in pending_images:
                parts.append({"type": "text", "text": f"Image {img_path}:"})
                parts.append({"type": "image_url", "image_url": {"url": data_url}})
            messages.append({"role": "user", "content": parts})
    else:
        # Max iterations — force final answer
        final_answer = await wrap_up_conversation(llm, messages, tools)

    logger.info(
        f"  [guided_extraction:{entity_id}] completed after "
        f"{iteration + 1} iteration(s)"
    )

    if not final_answer and not is_skipped:
        return {"sample_id": "", "polymer_id": entity_id, "error": "No answer produced"}

    # Use structured output with Instructor for validated extraction
    if use_structured_output and not is_skipped:
        result = await _extract_with_structured_output(
            messages=messages,
            llm=llm,
            entity_id=entity_id,
            is_skipped=is_skipped,
        )
        if isinstance(result, PolymerSample):
            return result.model_dump()
        return result

    # Fallback to JSON parsing for skipped entities or when structured output is disabled
    return parse_extraction_answer(final_answer, entity_id)

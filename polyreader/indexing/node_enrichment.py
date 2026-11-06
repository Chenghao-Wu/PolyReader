"""Generate summaries and doc descriptions for tree nodes."""

import asyncio
import json
import logging

from polyreader.llm.client import LLMClient
from polyreader.llm.token_counter import count_tokens
from .prompts import SUMMARIZE_NODE_PROMPT, GENERATE_DOC_DESCRIPTION_PROMPT
from polyreader.utils.tree_utils import structure_to_list, extract_section_text

logger = logging.getLogger(__name__)


async def _generate_summary(
    node: dict,
    llm: LLMClient,
    page_texts: list[str],
    summary_token_threshold: int = 200,
    use_prefix: bool = False,
) -> str:
    """Generate summary for a single node from its page range.

    Uses section-specific text extraction (via markdown header parsing) to
    avoid polluting summaries with content from adjacent sections.

    Args:
        node: Tree node dict.
        llm: LLM client.
        page_texts: Per-page text (1-indexed via slicing).
        summary_token_threshold: Below this token count, use raw text as summary.
        use_prefix: If True, use prefix_text (for parent nodes) instead of section_body.
    """
    sp = node.get("start_index")
    ep = node.get("end_index")
    if sp is None or ep is None or not page_texts:
        return ""

    title = node.get("title", "")

    # Try section-specific extraction first
    section_body, prefix_text = extract_section_text(node, page_texts)
    text = (prefix_text if use_prefix else section_body) or ""

    # Fall back to full page range if extraction returned nothing useful
    if not text.strip():
        text = "\n\n".join(page_texts[sp - 1 : ep])

    tokens = count_tokens(text)
    if tokens < summary_token_threshold:
        return text

    prompt = SUMMARIZE_NODE_PROMPT.format(title=title, text=text)
    return await llm.chat_simple(prompt)


async def enrich_nodes(
    tree: list[dict],
    llm: LLMClient,
    page_texts: list[str],
    summary_token_threshold: int = 200,
    max_concurrent: int = 3,
):
    """Generate summaries for all nodes using page_texts directly.

    Leaf nodes get a "summary" field; parent nodes get "prefix_summary" instead.

    Args:
        tree: Hierarchical tree structure
        llm: LLM client for summary generation
        page_texts: Per-page text from parser (1-indexed via slicing)
        summary_token_threshold: Below this token count, use text as summary directly
        max_concurrent: Max concurrent API calls
    """
    nodes = structure_to_list(tree)
    semaphore = asyncio.Semaphore(max_concurrent)

    async def summarize_with_limit(node, use_prefix=False):
        async with semaphore:
            return await _generate_summary(
                node, llm, page_texts, summary_token_threshold, use_prefix=use_prefix,
            )

    # Build tasks: leaf nodes get section_body, parent nodes get prefix_text
    task_specs = []  # (node, asyncio.Task, is_parent)
    coros = []
    for node in nodes:
        is_parent = bool(node.get("nodes"))
        coros.append(summarize_with_limit(node, use_prefix=is_parent))
        task_specs.append((node, is_parent))

    summaries = await asyncio.gather(*coros, return_exceptions=True)

    for (node, is_parent), summary in zip(task_specs, summaries):
        if isinstance(summary, Exception):
            logger.error(f"Summary failed for {node.get('title', '')}: {summary}")
            continue

        if is_parent:
            node["prefix_summary"] = summary
            node.pop("summary", None)
        else:
            node["summary"] = summary

    _reorder_node_keys(tree)
    return tree


_NODE_KEY_ORDER = ["title", "start_index", "end_index", "node_id", "summary", "prefix_summary", "nodes"]


def _reorder_node_keys(tree: list[dict]):
    """Reorder node dict keys for clean JSON output and strip stale fields.

    Ensures summary/prefix_summary appear before nodes, and that parent nodes
    only have prefix_summary (not summary) and leaf nodes only have summary.
    """
    for node in structure_to_list(tree):
        is_parent = bool(node.get("nodes"))
        if is_parent:
            node.pop("summary", None)
        else:
            node.pop("prefix_summary", None)

        ordered = {k: node[k] for k in _NODE_KEY_ORDER if k in node}
        for k in node:
            if k not in ordered:
                ordered[k] = node[k]
        node.clear()
        node.update(ordered)


def _clean_structure_for_description(tree: list[dict]) -> list[dict]:
    """Build a minimal structure with only title, node_id, summary."""
    result = []
    for node in (tree if isinstance(tree, list) else [tree]):
        clean = {"title": node.get("title", "")}
        if node.get("node_id"):
            clean["node_id"] = node["node_id"]
        if node.get("summary"):
            clean["summary"] = node["summary"]
        if node.get("prefix_summary"):
            clean["prefix_summary"] = node["prefix_summary"]
        if node.get("nodes"):
            clean["nodes"] = _clean_structure_for_description(node["nodes"])
        result.append(clean)
    return result


async def generate_doc_description(tree: list[dict], llm: LLMClient) -> str:
    """Generate a one-sentence description of the document from its tree structure.

    Args:
        tree: Hierarchical tree with summaries
        llm: LLM client

    Returns:
        One-sentence description string
    """
    clean = _clean_structure_for_description(tree)
    structure_str = json.dumps(clean, indent=2, ensure_ascii=False)
    prompt = GENERATE_DOC_DESCRIPTION_PROMPT.format(structure=structure_str)
    return await llm.chat_simple(prompt)

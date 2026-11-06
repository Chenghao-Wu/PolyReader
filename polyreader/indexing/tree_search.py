"""LLM reasoning-based node selection (vectorless retrieval)."""

import json
import logging
import re
from dataclasses import dataclass, field

from polyreader.llm.client import LLMClient
from .prompts import TREE_SEARCH_PROMPT, TREE_SEARCH_RETRY_PROMPT
from polyreader.utils.tree_utils import remove_fields, create_node_mapping, keyword_search_nodes
from polyreader.utils.constants import tokenize_query
from polyreader.utils.json_utils import extract_json

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Result from tree search."""
    thinking: str = ""
    node_ids: list[str] = field(default_factory=list)
    raw_response: str = ""


@dataclass
class VerifiedSearchResult(SearchResult):
    """SearchResult with verification metadata."""
    confidence: float = 1.0
    verified_node_ids: list[str] = field(default_factory=list)
    dropped_node_ids: list[str] = field(default_factory=list)


def _parse_search_response(response: str) -> SearchResult:
    """Parse LLM response into a SearchResult."""
    parsed = extract_json(response)
    thinking = ""
    node_ids = []
    if isinstance(parsed, dict):
        thinking = parsed.get("thinking", "")
        node_ids = parsed.get("node_list", [])
    elif isinstance(parsed, list):
        node_ids = parsed
    return SearchResult(
        thinking=thinking,
        node_ids=[str(nid) for nid in node_ids],
        raw_response=response,
    )


def verify_search_result(
    result: SearchResult,
    tree: list[dict],
    query: str,
    node_map: dict = None,
) -> VerifiedSearchResult:
    """Verify that search results are valid (deterministic, no LLM call).

    Checks:
    1. Each node_id actually exists in the tree
    2. Each node has valid page ranges (start_index, end_index)
    3. Node title/summary has some keyword overlap with query

    Returns:
        VerifiedSearchResult with confidence score and verified/dropped lists
    """
    if node_map is None:
        node_map = create_node_mapping(tree)

    if not result.node_ids:
        return VerifiedSearchResult(
            thinking=result.thinking,
            node_ids=[],
            raw_response=result.raw_response,
            confidence=0.0,
            verified_node_ids=[],
            dropped_node_ids=[],
        )

    query_keywords = tokenize_query(query)

    verified = []
    dropped = []

    for nid in result.node_ids:
        node = node_map.get(nid)

        # Check 1: node exists
        if node is None:
            dropped.append(nid)
            continue

        # Check 2: valid page range
        sp = node.get("start_index")
        ep = node.get("end_index")
        if sp is None or ep is None or sp < 1 or ep < sp:
            dropped.append(nid)
            continue

        # Check 3: keyword overlap (lenient — at least some connection)
        if query_keywords:
            title = node.get("title", "").lower()
            summary = (node.get("summary") or node.get("prefix_summary", "")).lower()
            node_words = set(re.findall(r'[a-z0-9]+', title + " " + summary))
            has_overlap = bool(query_keywords & node_words)
            is_parent = bool(node.get("nodes"))
            if not has_overlap and not is_parent:
                # Still keep it but note the weak match — don't drop purely
                # on keyword mismatch as LLM may reason about implicit relevance
                pass

        verified.append(nid)

    total = len(result.node_ids)
    confidence = len(verified) / total if total > 0 else 0.0

    return VerifiedSearchResult(
        thinking=result.thinking,
        node_ids=verified,
        raw_response=result.raw_response,
        confidence=confidence,
        verified_node_ids=verified,
        dropped_node_ids=dropped,
    )


async def search_tree(
    tree: list[dict],
    query: str,
    llm: LLMClient,
) -> SearchResult:
    """Use LLM reasoning to find nodes relevant to a query.

    Sends the tree structure (titles + summaries, no text) to LLM.
    LLM reasons about which nodes contain relevant information.

    Returns:
        SearchResult with thinking process and selected node IDs
    """
    # Remove text fields to save tokens
    tree_for_search = remove_fields(tree, fields=["text"])
    tree_json = json.dumps(tree_for_search, indent=2, ensure_ascii=False)

    prompt = TREE_SEARCH_PROMPT.format(
        query=query,
        tree_json=tree_json,
    )

    response = await llm.chat_simple(prompt)
    return _parse_search_response(response)


async def search_tree_with_fallback(
    tree: list[dict],
    query: str,
    llm: LLMClient,
    node_map: dict = None,
    confidence_threshold: float = 0.5,
    max_retries: int = 2,
) -> VerifiedSearchResult:
    """Search tree with verification and fallback mechanisms.

    1. Run search_tree() -> verify results
    2. If confidence < threshold: retry with TREE_SEARCH_RETRY_PROMPT
    3. If all retries fail: fall back to keyword_search_nodes()

    Args:
        tree: Paper tree structure
        query: Search query
        llm: LLM client
        node_map: Optional pre-built node mapping
        confidence_threshold: Minimum confidence to accept results
        max_retries: Maximum retry attempts before keyword fallback

    Returns:
        VerifiedSearchResult with confidence and verification metadata
    """
    if node_map is None:
        node_map = create_node_mapping(tree)

    # First attempt
    result = await search_tree(tree, query, llm)
    verified = verify_search_result(result, tree, query, node_map=node_map)

    if verified.confidence >= confidence_threshold and verified.node_ids:
        logger.info(
            f"Search verified: {len(verified.verified_node_ids)} valid, "
            f"{len(verified.dropped_node_ids)} dropped, "
            f"confidence={verified.confidence:.2f}"
        )
        return verified

    # Retry with feedback about what failed
    tree_for_search = remove_fields(tree, fields=["text"])
    tree_json = json.dumps(tree_for_search, indent=2, ensure_ascii=False)

    for retry in range(max_retries):
        issues = []
        if verified.dropped_node_ids:
            issues.append(
                f"Invalid node_ids (not found in tree): {verified.dropped_node_ids}"
            )
        if not verified.node_ids:
            issues.append("No valid nodes were selected. Please look more carefully.")

        prompt = TREE_SEARCH_RETRY_PROMPT.format(
            query=query,
            tree_json=tree_json,
            issues="\n".join(issues) if issues else "No results returned.",
        )

        response = await llm.chat_simple(prompt)
        retry_result = _parse_search_response(response)
        verified = verify_search_result(retry_result, tree, query, node_map=node_map)

        if verified.confidence >= confidence_threshold and verified.node_ids:
            logger.info(
                f"Search verified on retry {retry + 1}: "
                f"{len(verified.verified_node_ids)} valid, "
                f"confidence={verified.confidence:.2f}"
            )
            return verified

    # Keyword fallback
    logger.warning(
        f"Search failed after {max_retries} retries. "
        f"Falling back to keyword search."
    )
    keyword_ids = keyword_search_nodes(tree, query, top_k=5)

    return VerifiedSearchResult(
        thinking=f"Keyword fallback after {max_retries} failed LLM searches.",
        node_ids=keyword_ids,
        raw_response="",
        confidence=0.3 if keyword_ids else 0.0,
        verified_node_ids=keyword_ids,
        dropped_node_ids=verified.dropped_node_ids,
    )

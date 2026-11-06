"""Page texts -> hierarchical section tree via LLM structure extraction."""

import asyncio
import math
import logging

from polyreader.llm.client import LLMClient
from polyreader.llm.token_counter import count_tokens
from .prompts import (
    EXTRACT_STRUCTURE_INIT_PROMPT,
    EXTRACT_STRUCTURE_CONTINUE_PROMPT,
    FIX_SECTION_INDEX_PROMPT,
    CHECK_APPEAR_START_PROMPT,
)
from polyreader.utils.json_utils import extract_json
from polyreader.utils.constants import normalize_text as _normalize_text

logger = logging.getLogger(__name__)

# Approximate token overhead added by <page_N>...</page_N> wrapper tags
_PAGE_TAG_TOKEN_OVERHEAD = 10

# Number of leading characters on page 1 to scan for "abstract"
_ABSTRACT_SCAN_CHARS = 500


# ---------------------------------------------------------------------------
# LLM-based structure extraction
# ---------------------------------------------------------------------------

def _wrap_pages_with_tags(
    page_texts: list[str],
    page_tokens: list[int] | None = None,
) -> list[tuple[str, int]]:
    """Wrap page texts with <page_N> tags. Returns (tagged_text, token_count).

    If *page_tokens* is provided the tag overhead is estimated rather than
    re-counting every page.
    """
    result = []
    for i, text in enumerate(page_texts):
        tagged = f"<page_{i + 1}>\n{text}\n</page_{i + 1}>\n\n"
        tokens = (
            page_tokens[i] + _PAGE_TAG_TOKEN_OVERHEAD
            if page_tokens is not None
            else count_tokens(tagged)
        )
        result.append((tagged, tokens))
    return result


def _group_pages(
    tagged_pages: list[tuple[str, int]],
    max_tokens: int = 20000,
    overlap_pages: int = 1,
) -> list[str]:
    """Group tagged pages into token-limited chunks with overlap.

    Args:
        tagged_pages: List of (tagged_text, token_count) tuples.
        max_tokens: Maximum tokens per group.
        overlap_pages: Number of pages from the end of the previous group
            to include at the start of the next group.
    """
    total = sum(t for _, t in tagged_pages)
    if total <= max_tokens:
        return ["".join(text for text, _ in tagged_pages)]

    groups = []
    current = []       # list of (text, tokens)
    current_tokens = 0
    expected_parts = math.ceil(total / max_tokens)
    avg_tokens = math.ceil(((total / expected_parts) + max_tokens) / 2)

    for text, tokens in tagged_pages:
        if current_tokens + tokens > avg_tokens and current:
            groups.append("".join(t for t, _ in current))
            # Start next group with overlap from previous group
            overlap = current[-overlap_pages:] if overlap_pages > 0 else []
            current = list(overlap)
            current_tokens = sum(tk for _, tk in overlap)
        current.append((text, tokens))
        current_tokens += tokens

    if current:
        groups.append("".join(t for t, _ in current))
    return groups


async def _llm_extract_structure(
    page_texts: list[str],
    llm: LLMClient,
    page_tokens: list[int] | None = None,
) -> list[dict]:
    """Use LLM to extract section structure from page texts."""
    tagged = _wrap_pages_with_tags(page_texts, page_tokens)
    groups = _group_pages(tagged)

    # First group
    prompt = EXTRACT_STRUCTURE_INIT_PROMPT.format(text=groups[0])
    result = await llm.chat_simple(prompt)
    structure = extract_json(result)
    if not isinstance(structure, list):
        structure = []

    # Subsequent groups
    for group_text in groups[1:]:
        prompt = EXTRACT_STRUCTURE_CONTINUE_PROMPT.format(
            text=group_text,
            prev_structure=str(structure),
        )
        result = await llm.chat_simple(prompt)
        additional = extract_json(result)
        if isinstance(additional, list):
            structure.extend(additional)

    # Coerce page_num to int — LLM sometimes returns JSON strings like "5"
    for item in structure:
        if "page_num" in item and item["page_num"] is not None:
            item["page_num"] = int(item["page_num"])

    return structure


# ---------------------------------------------------------------------------
# Verification & fix-up
# ---------------------------------------------------------------------------

def _verify_structure(
    structure: list[dict], page_texts: list[str]
) -> tuple[float, list[dict], list[dict]]:
    """Verify each entry by checking if its title appears on the claimed page.

    Uses case-insensitive, whitespace-normalized substring matching.
    Also checks adjacent pages (+-1) for boundary cases.

    Returns (accuracy, correct_items, incorrect_items).
    """
    if not structure or not page_texts:
        return 0.0, [], list(structure)

    correct = []
    incorrect = []

    for item in structure:
        title = item.get("title", "")
        page_num = item.get("page_num")

        if not title or page_num is None:
            incorrect.append(item)
            continue

        # Check if title appears on the claimed page (1-indexed)
        page_idx = page_num - 1
        if page_idx < 0 or page_idx >= len(page_texts):
            incorrect.append(item)
            continue

        title_norm = _normalize_text(title)

        found = False
        for offset in [0, -1, 1]:
            check_idx = page_idx + offset
            if 0 <= check_idx < len(page_texts):
                check_text = _normalize_text(page_texts[check_idx])
                if title_norm in check_text:
                    found = True
                    # If found on adjacent page, fix the page_num
                    if offset != 0:
                        item["page_num"] = check_idx + 1
                        logger.info(
                            f"Title '{title}' found on page {check_idx + 1} "
                            f"instead of {page_num}, auto-corrected"
                        )
                    break

        if found:
            correct.append(item)
        else:
            incorrect.append(item)
            logger.warning(
                f"Title '{title}' NOT found on page {page_num} (or adjacent)"
            )

    accuracy = len(correct) / len(structure) if structure else 0.0
    logger.info(
        f"Verification: {len(correct)}/{len(structure)} correct "
        f"({accuracy:.0%})"
    )
    return accuracy, correct, incorrect


async def _fix_incorrect_entries(
    incorrect: list[dict],
    correct: list[dict],
    structure: list[dict],
    page_texts: list[str],
    llm: LLMClient,
    max_concurrent: int = 5,
) -> list[dict]:
    """Re-extract page numbers for incorrect entries using narrowed context.

    Sends only pages between surrounding correct entries to the LLM.
    Runs calls concurrently with a semaphore bound.
    """
    if not incorrect:
        return structure

    # Build a sorted list of correct page_nums for context narrowing
    correct_pages = sorted(
        set(item.get("page_num", 1) for item in correct if item.get("page_num"))
    )

    semaphore = asyncio.Semaphore(max_concurrent)

    async def fix_one(item: dict) -> bool:
        """Fix a single incorrect entry. Returns True if fixed."""
        title = item.get("title", "")
        orig_page = item.get("page_num", 1)

        # Narrow page range to between surrounding correct entries
        prev_page = 1
        next_page = len(page_texts)
        for cp in correct_pages:
            if cp < orig_page:
                prev_page = cp
            elif cp > orig_page:
                next_page = cp
                break

        start_idx = max(0, prev_page - 1)
        end_idx = min(len(page_texts), next_page)
        context_pages = page_texts[start_idx:end_idx]
        if not context_pages:
            return False

        tagged_text = "".join(
            f"<page_{start_idx + i + 1}>\n{text}\n</page_{start_idx + i + 1}>\n\n"
            for i, text in enumerate(context_pages)
        )

        prompt = FIX_SECTION_INDEX_PROMPT.format(title=title, text=tagged_text)
        try:
            async with semaphore:
                result = await llm.chat_simple(prompt)
            parsed = extract_json(result)
            if not isinstance(parsed, dict) or "page_num" not in parsed:
                return False
            try:
                new_page = int(parsed["page_num"])
            except (TypeError, ValueError):
                return False
            if 1 <= new_page <= len(page_texts):
                logger.info(f"Fixed '{title}': page {orig_page} -> {new_page}")
                item["page_num"] = new_page
                return True
        except Exception as e:
            logger.warning(f"Failed to fix entry '{title}': {e}")
        return False

    results = await asyncio.gather(
        *(fix_one(item) for item in incorrect), return_exceptions=True
    )
    fixed_count = sum(
        1 for r in results if isinstance(r, bool) and r
    )
    for item, r in zip(incorrect, results):
        if isinstance(r, Exception):
            logger.error(f"Fix-up failed for '{item.get('title', '')}': {r}")

    logger.info(f"Fix-up: corrected {fixed_count}/{len(incorrect)} entries")
    return structure


async def _check_appear_start(
    structure: list[dict], page_texts: list[str], llm: LLMClient,
    max_concurrent: int = 5,
):
    """Check if each section title appears at the start of its page (LLM).

    Sets ``appear_start`` ('yes' or 'no') on each item so
    ``_structure_to_tree`` can decide page boundary behaviour.
    """
    semaphore = asyncio.Semaphore(max_concurrent)

    async def check_one(item: dict):
        title = item.get("title", "")
        page_num = item.get("page_num")
        page_idx = (page_num or 0) - 1

        if not title or page_num is None or not (0 <= page_idx < len(page_texts)):
            item["appear_start"] = "no"
            return

        prompt = CHECK_APPEAR_START_PROMPT.format(
            title=title, page_text=page_texts[page_idx],
        )
        try:
            async with semaphore:
                result = await llm.chat_simple(prompt)
            parsed = extract_json(result)
            if isinstance(parsed, dict):
                answer = str(parsed.get("start_begin", "no")).lower().strip()
                item["appear_start"] = "yes" if answer == "yes" else "no"
            else:
                item["appear_start"] = "no"
        except Exception as e:
            logger.warning(f"appear_start check failed for '{title}': {e}")
            item["appear_start"] = "no"

    await asyncio.gather(*(check_one(item) for item in structure))


def _detect_abstract(
    structure: list[dict], page_texts: list[str]
) -> list[dict]:
    """Insert a dedicated Abstract node when the abstract lives on page 1.

    If the first structure entry starts on page 1 and "abstract" appears in
    the leading text, inserts a structure-"0" node so the abstract gets its
    own leaf. Skips if an abstract entry already exists.
    """
    if not structure or not page_texts:
        return structure

    if any("abstract" in item.get("title", "").lower() for item in structure):
        return structure

    if structure[0].get("page_num", 1) != 1:
        return structure

    page1_lower = page_texts[0].lower()
    if "abstract" in page1_lower[:_ABSTRACT_SCAN_CHARS]:
        abstract_node = {
            "structure": "0",
            "title": "Abstract",
            "page_num": 1,
        }
        structure.insert(0, abstract_node)
        logger.info("Inserted Abstract node (page 1)")

    return structure


async def _fix_with_retries(
    structure: list[dict],
    page_texts: list[str],
    llm: LLMClient,
    max_attempts: int = 3,
) -> list[dict]:
    """Verify structure and fix incorrect entries, retrying up to max_attempts."""
    for attempt in range(max_attempts):
        accuracy, correct, incorrect = _verify_structure(structure, page_texts)
        if not incorrect:
            break
        structure = await _fix_incorrect_entries(
            incorrect, correct, structure, page_texts, llm
        )
        logger.info(f"Fix attempt {attempt + 1}/{max_attempts}: {len(incorrect)} entries to fix")
    return structure


def _structure_to_tree(structure: list[dict], total_pages: int) -> list[dict]:
    """Convert flat LLM structure list to hierarchical tree.

    Computes both start_index and end_index (inclusive) on the flat list
    before building the tree hierarchy.
    """
    if not structure:
        return []

    # Compute start_index and end_index on flat list.
    # appear_start == 'yes' → clean break, end on prev page
    # appear_start == 'no'  → include the shared page in current section
    for i, item in enumerate(structure):
        item["start_index"] = item.get("page_num")
        if i + 1 < len(structure):
            nxt = structure[i + 1]
            next_page = nxt.get("page_num")
            boundary = -1 if nxt.get("appear_start") == "yes" else 0
            item["end_index"] = max(item["start_index"], next_page + boundary)
        else:
            item["end_index"] = total_pages

    # Build parent-child via structure index
    nodes = {}
    roots = []

    for item in structure:
        struct_idx = item.get("structure", "")
        node = {
            "title": item.get("title", ""),
            "start_index": item.get("start_index"),
            "end_index": item.get("end_index"),
            "node_id": None,
            "nodes": [],
        }
        nodes[struct_idx] = node

        # Find parent
        parts = str(struct_idx).split(".")
        parent_idx = ".".join(parts[:-1]) if len(parts) > 1 else None

        if parent_idx and parent_idx in nodes:
            nodes[parent_idx]["nodes"].append(node)
        else:
            roots.append(node)

    return roots


# ---------------------------------------------------------------------------
# Large node splitting (PageIndex-style)
# ---------------------------------------------------------------------------

async def _split_one_node(
    node: dict,
    effective_ep: int,
    page_texts: list[str],
    llm: LLMClient,
    semaphore: asyncio.Semaphore,
):
    """Split a single large node into sub-nodes via LLM extraction.

    Mutates ``node`` in place — sets node["nodes"] and adjusts end_index.
    """
    sp = node.get("start_index", 1)
    ep = node.get("end_index", len(page_texts))
    node_pages = page_texts[sp - 1 : effective_ep]  # trimmed for LLM

    async with semaphore:
        sub_structure = await _llm_extract_structure(node_pages, llm)

    logger.info(
        f"Sub-extraction for '{node.get('title', '')}' "
        f"(pages {sp}-{effective_ep}, trimmed from {ep}) found "
        f"{len(sub_structure)} sub-sections: "
        f"{[s.get('title', '') for s in sub_structure]}"
    )

    if not isinstance(sub_structure, list) or len(sub_structure) <= 1:
        return

    # Adjust page numbers to global (offset by sp-1)
    for item in sub_structure:
        if item.get("page_num"):
            item["page_num"] += sp - 1

    # Verify and fix sub-extracted structure
    _, sub_correct, sub_incorrect = _verify_structure(
        sub_structure, page_texts
    )
    if sub_incorrect:
        async with semaphore:
            sub_structure = await _fix_incorrect_entries(
                sub_incorrect, sub_correct, sub_structure,
                page_texts, llm,
            )

    # Drop first child if it duplicates the parent heading
    parent_title = _normalize_text(node.get("title", ""))
    first_title = _normalize_text(sub_structure[0].get("title", ""))
    if parent_title and parent_title == first_title:
        logger.info(
            f"Dropping duplicate first child '{sub_structure[0].get('title', '')}'"
        )
        sub_structure = sub_structure[1:]

    if sub_structure:
        async with semaphore:
            await _check_appear_start(sub_structure, page_texts, llm)
        sub_tree = _structure_to_tree(sub_structure, effective_ep)
        if sub_tree:
            node["nodes"] = sub_tree
            # Adjust parent end_index to just before first child
            node["end_index"] = max(
                node["start_index"],
                sub_tree[0]["start_index"] - 1,
            )
            logger.info(
                f"Split large node '{node.get('title', '')}' "
                f"into {len(sub_tree)} sub-nodes"
            )


async def _split_large_nodes(
    tree: list[dict],
    page_texts: list[str],
    max_tokens: int,
    llm: LLMClient,
    max_page_num: int = 5,
    page_tokens: list[int] | None = None,
    max_concurrent: int = 3,
):
    """Recursively split leaf nodes exceeding both max_tokens AND max_page_num.

    Siblings are split concurrently (bounded by semaphore), then children
    are processed recursively.
    """
    semaphore = asyncio.Semaphore(max_concurrent)

    to_split = []
    for i, node in enumerate(tree):
        sp = node.get("start_index", 1)
        ep = node.get("end_index", len(page_texts))

        # Compute effective_ep: trim shared boundary page
        effective_ep = ep
        if i + 1 < len(tree):
            next_sp = tree[i + 1].get("start_index", ep + 1)
            if ep >= next_sp:
                effective_ep = next_sp - 1

        page_count = ep - sp + 1
        page_range = range(sp - 1, ep)
        tokens = (
            sum(page_tokens[j] for j in page_range)
            if page_tokens is not None
            else sum(count_tokens(page_texts[j]) for j in page_range)
        )

        logger.debug(
            f"Evaluating node '{node.get('title', '')}': "
            f"pages={sp}-{ep} ({page_count} pages) (max page num {max_page_num}), "
            f"tokens={tokens} (max_tokens {max_tokens}), "
            f"exceeds_thresholds={page_count > max_page_num and tokens > max_tokens}"
        )

        if page_count > max_page_num and tokens > max_tokens:
            to_split.append((node, effective_ep))

    if to_split:
        await asyncio.gather(*(
            _split_one_node(node, eff_ep, page_texts, llm, semaphore)
            for node, eff_ep in to_split
        ))

    # Recurse into children (including newly created sub-nodes)
    child_tasks = []
    for node in tree:
        if node.get("nodes"):
            child_tasks.append(
                _split_large_nodes(
                    node["nodes"], page_texts, max_tokens, llm, max_page_num,
                    page_tokens=page_tokens, max_concurrent=max_concurrent,
                )
            )
    if child_tasks:
        await asyncio.gather(*child_tasks)


# ---------------------------------------------------------------------------
# Validation diagnostics
# ---------------------------------------------------------------------------

def _format_tree(nodes: list[dict], indent: int = 0) -> list[str]:
    """Return indented lines showing title and page range for each node."""
    lines = []
    for n in nodes:
        lines.append(
            f"{'  ' * indent}{n['title']} [pp.{n['start_index']}-{n['end_index']}]"
        )
        lines.extend(_format_tree(n.get("nodes", []), indent + 1))
    return lines


def _validate_tree_invariants(nodes: list[dict], parent: dict | None = None):
    """Log warnings for tree invariant violations (diagnostic only)."""
    seen = set()
    for node in nodes:
        title = node.get("title", "")
        sp = node.get("start_index")
        ep = node.get("end_index")

        if sp is not None and ep is not None and sp > ep:
            logger.warning(
                f"Inverted range: '{title}' has start_index={sp} > end_index={ep}"
            )

        if parent is not None:
            parent_sp = parent.get("start_index")
            if sp is not None and parent_sp is not None and sp < parent_sp:
                logger.warning(
                    f"Child before parent: '{title}' start_index={sp} < "
                    f"parent '{parent.get('title', '')}' start_index={parent_sp}"
                )

        key = (_normalize_text(title), sp)
        if key in seen:
            logger.warning(f"Duplicate sibling: '{title}' at start_index={sp}")
        seen.add(key)

        if node.get("nodes"):
            _validate_tree_invariants(node["nodes"], parent=node)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

async def build_tree(
    page_texts: list[str],
    llm: LLMClient,
    max_token_num: int = 20000,
    max_page_num: int = 5,
) -> list[dict]:
    """Build hierarchical section tree from page texts via LLM extraction.

    Pipeline: extract -> verify/fix -> check_appear_start -> detect_abstract
              -> to_tree -> split_large -> validate

    Returns tree WITHOUT text on nodes (text assignment handled separately).
    """
    logger.info(
        f"Parameters: max_token_num={max_token_num}, max_page_num={max_page_num}, "
        f"total_pages={len(page_texts)}"
    )

    page_tokens = [count_tokens(text) for text in page_texts]
    logger.info(f"Total tokens: {sum(page_tokens)}")

    structure = await _llm_extract_structure(page_texts, llm, page_tokens)
    structure = await _fix_with_retries(structure, page_texts, llm, max_attempts=3)
    await _check_appear_start(structure, page_texts, llm)
    structure = _detect_abstract(structure, page_texts)

    tree = _structure_to_tree(structure, len(page_texts))
    logger.info("Tree structure:\n%s", "\n".join(_format_tree(tree)))

    await _split_large_nodes(
        tree, page_texts, max_token_num, llm, max_page_num,
        page_tokens=page_tokens,
    )
    _validate_tree_invariants(tree)

    return tree

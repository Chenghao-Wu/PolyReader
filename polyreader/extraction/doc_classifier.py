"""LLM-based auto-detection of main vs. supplementary documents.

Classifies each PaperIndex as "main" (research paper) or "supplementary"
(supporting information) by examining root section titles, first-page text,
and filename.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from enum import Enum

from polyreader.indexing import PaperIndex
from polyreader.llm import LLMClient
from polyreader.utils.json_utils import extract_json

logger = logging.getLogger(__name__)


class DocumentRole(Enum):
    MAIN = "main"
    SUPPLEMENTARY = "supplementary"


@dataclass
class ClassificationResult:
    role: DocumentRole
    confidence: str  # "high" or "low"
    reasoning: str  # one-line LLM explanation


CLASSIFY_PROMPT = """\
Classify this scientific document as either "main" (a research paper) or "supplementary" (supporting information / SI).

Document name: {doc_name}

Section titles:
{section_titles}

First page text:
{first_page_text}

Return JSON:
{{"doc_type": "main" or "supplementary", "confidence": "high" or "low", "reasoning": "<one sentence>"}}

Directly return the JSON. Do not output anything else."""


def _build_section_titles(tree: list[dict]) -> str:
    """Extract root-level section titles with page ranges from the tree."""
    lines = []
    for node in tree:
        title = node.get("title", "Untitled")
        start = node.get("start_index", "?")
        end = node.get("end_index", "?")
        lines.append(f"- {title} (pages {start}-{end})")
    return "\n".join(lines) if lines else "(no sections found)"


async def classify_document(index: PaperIndex, llm: LLMClient) -> ClassificationResult:
    """Classify a single PaperIndex as main or supplementary.

    Uses a single LLM call examining root section titles, first-page text,
    and the document filename.
    """
    section_titles = _build_section_titles(index.tree)

    # Get first page text, truncated to ~2000 chars
    index._lazy_load_pages()
    if index.page_texts:
        first_page = index.page_texts[0][:2000]
    else:
        first_page = "(no page text available)"

    prompt = CLASSIFY_PROMPT.format(
        doc_name=index.doc_name or "(unknown)",
        section_titles=section_titles,
        first_page_text=first_page,
    )

    response = await llm.chat_simple(prompt)
    parsed = extract_json(response, quiet=True)

    doc_type = parsed.get("doc_type", "main").lower().strip()
    confidence = parsed.get("confidence", "low").lower().strip()
    reasoning = parsed.get("reasoning", "")

    if doc_type == "supplementary":
        role = DocumentRole.SUPPLEMENTARY
    else:
        role = DocumentRole.MAIN

    if confidence not in ("high", "low"):
        confidence = "low"

    result = ClassificationResult(role=role, confidence=confidence, reasoning=reasoning)
    logger.info(f"  [doc_classifier] {index.doc_name}: {result.role.value} ({result.confidence}) — {result.reasoning}")
    return result


async def sort_documents(
    indices: list[PaperIndex], llm: LLMClient
) -> tuple[PaperIndex, list[PaperIndex], list[ClassificationResult]]:
    """Classify all documents and sort into main + supplementary.

    Returns:
        (main_index, si_list, classifications) where classifications
        are in the same order as the input indices.
    """
    if len(indices) == 1:
        result = ClassificationResult(
            role=DocumentRole.MAIN, confidence="high", reasoning="Only one document provided."
        )
        return indices[0], [], [result]

    classifications = await asyncio.gather(
        *(classify_document(idx, llm) for idx in indices)
    )

    # Find main documents
    main_candidates = [
        (i, idx, cls)
        for i, (idx, cls) in enumerate(zip(indices, classifications))
        if cls.role == DocumentRole.MAIN
    ]

    if len(main_candidates) == 0:
        logger.warning("  [doc_classifier] No document classified as main — falling back to first document.")
        main_idx = 0
    elif len(main_candidates) == 1:
        main_idx = main_candidates[0][0]
    else:
        # Multiple mains — pick highest confidence, then first positionally
        logger.warning(
            f"  [doc_classifier] {len(main_candidates)} documents classified as main — "
            "picking highest-confidence one."
        )
        high_conf = [c for c in main_candidates if c[2].confidence == "high"]
        if high_conf:
            main_idx = high_conf[0][0]
        else:
            main_idx = main_candidates[0][0]

    main_index = indices[main_idx]
    si_list = [idx for i, idx in enumerate(indices) if i != main_idx]

    return main_index, si_list, classifications

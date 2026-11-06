"""Parse bibliography/citations from the References section."""

import re
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class Reference:
    """A single bibliography entry."""
    number: int = 0
    text: str = ""
    authors: str = ""
    title: str = ""
    journal: str = ""
    year: str = ""
    doi: str = ""


def parse_references(references_text: str) -> list[Reference]:
    """Parse numbered references from the References section.

    Handles common formats:
    - [1] Author, Title, Journal, Year
    - (1) Author, Title, Journal, Year
    - 1. Author, Title, Journal, Year
    """
    if not references_text:
        return []

    # Find all reference starts with their positions
    ref_pattern = re.compile(
        r"(?:^|\n)\s*(?:\[(\d+)\]|\((\d+)\)|(\d+)\.)\s+",
        re.MULTILINE,
    )

    matches = list(ref_pattern.finditer(references_text))
    if not matches:
        return []

    refs = []
    for i, match in enumerate(matches):
        # Get reference number from whichever group matched
        num = int(next(g for g in match.groups() if g is not None))

        # Text runs from end of this match to start of next match (or end)
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(references_text)
        text = references_text[start:end].strip()

        ref = Reference(number=num, text=text)

        # Try to extract DOI
        doi_match = re.search(
            r"(?:doi:|DOI:|https?://doi\.org/)(10\.\S+)",
            text,
        )
        if doi_match:
            ref.doi = doi_match.group(1).rstrip(".")

        refs.append(ref)

    return refs


def find_references_node(tree: list[dict]) -> dict | None:
    """Find the References/Bibliography node in the tree."""
    from polyreader.utils.tree_utils import structure_to_list

    for node in structure_to_list(tree):
        title = node.get("title", "").lower()
        if any(keyword in title for keyword in ["references", "bibliography", "works cited"]):
            return node
    return None

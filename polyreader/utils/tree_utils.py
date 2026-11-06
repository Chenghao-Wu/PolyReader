"""Tree traversal, flattening, printing, node lookup utilities.

Tree helpers adapted from PageIndex by VectifyAI (MIT License; see THIRD_PARTY_NOTICES).
"""

import copy
import json
import re

from .constants import normalize_text, tokenize_query


def print_tree(tree, indent=0, max_title_len=50):
    """Print a tree structure as an indented table of contents with page ranges."""
    nodes = tree if isinstance(tree, list) else [tree]
    for node in nodes:
        title = node.get("title", "")
        if len(title) > max_title_len:
            title = title[:max_title_len] + "..."
        sp = node.get("start_index", "?")
        ep = node.get("end_index", "?")
        prefix = "  " * indent
        summary = node.get("summary") or node.get("prefix_summary", "")
        summary_text = f": {summary[:80]}..." if summary and len(summary) > 80 else f": {summary}" if summary else ""
        print(f"{prefix}{title} (pp.{sp}-{ep}){summary_text}")
        if node.get("nodes"):
            print_tree(node["nodes"], indent + 1, max_title_len)


def print_toc(tree, indent=0):
    """Print tree as plain TOC."""
    nodes = tree if isinstance(tree, list) else [tree]
    for node in nodes:
        print("  " * indent + node.get("title", ""))
        if node.get("nodes"):
            print_toc(node["nodes"], indent + 1)


def get_nodes(structure):
    """Flatten tree to list of nodes (without their children)."""
    if isinstance(structure, dict):
        node = copy.deepcopy(structure)
        node.pop("nodes", None)
        result = [node]
        for child in structure.get("nodes", []):
            result.extend(get_nodes(child))
        return result
    elif isinstance(structure, list):
        result = []
        for item in structure:
            result.extend(get_nodes(item))
        return result
    return []


def get_leaf_nodes(structure):
    """Get only leaf nodes (no children)."""
    if isinstance(structure, dict):
        if not structure.get("nodes"):
            node = copy.deepcopy(structure)
            node.pop("nodes", None)
            return [node]
        result = []
        for child in structure.get("nodes", []):
            result.extend(get_leaf_nodes(child))
        return result
    elif isinstance(structure, list):
        result = []
        for item in structure:
            result.extend(get_leaf_nodes(item))
        return result
    return []


def structure_to_list(structure):
    """Flatten tree to list keeping node references (not copies)."""
    if isinstance(structure, dict):
        result = [structure]
        for child in structure.get("nodes", []):
            result.extend(structure_to_list(child))
        return result
    elif isinstance(structure, list):
        result = []
        for item in structure:
            result.extend(structure_to_list(item))
        return result
    return []


def find_node_by_id(structure, node_id):
    """Find a node by its node_id."""
    for node in structure_to_list(structure):
        if node.get("node_id") == node_id:
            return node
    return None


def create_node_mapping(tree):
    """Create dict mapping node_id -> node."""
    return {
        node["node_id"]: node
        for node in structure_to_list(tree)
        if "node_id" in node
    }


def write_node_id(data, node_id=0):
    """Assign sequential zero-padded node IDs to all nodes."""
    if isinstance(data, dict):
        data["node_id"] = str(node_id).zfill(4)
        node_id += 1
        for child in data.get("nodes", []):
            node_id = write_node_id(child, node_id)
    elif isinstance(data, list):
        for item in data:
            node_id = write_node_id(item, node_id)
    return node_id


def format_structure(structure, order=None):
    """Reorder keys in tree nodes according to specified order."""
    if not order:
        return structure
    if isinstance(structure, dict):
        if "nodes" in structure:
            structure["nodes"] = format_structure(structure["nodes"], order)
        if not structure.get("nodes"):
            structure.pop("nodes", None)
        return {key: structure[key] for key in order if key in structure}
    elif isinstance(structure, list):
        return [format_structure(item, order) for item in structure]
    return structure


def remove_fields(data, fields=None):
    """Remove specified fields from all nodes in the tree."""
    if fields is None:
        fields = ["text"]
    if isinstance(data, dict):
        return {
            k: remove_fields(v, fields)
            for k, v in data.items()
            if k not in fields
        }
    elif isinstance(data, list):
        return [remove_fields(item, fields) for item in data]
    return data


def add_node_text(tree, page_texts: list[str]):
    """Assign text to each node from its page range."""
    for node in structure_to_list(tree):
        sp = node.get("start_index")
        ep = node.get("end_index")
        if sp is not None and ep is not None:
            node["text"] = "\n\n".join(page_texts[sp - 1 : ep])


def remove_node_text(tree):
    """Recursively remove text field from all nodes."""
    nodes = tree if isinstance(tree, list) else [tree]
    for node in nodes:
        node.pop("text", None)
        if node.get("nodes"):
            remove_node_text(node["nodes"])


def keyword_search_nodes(tree, query: str, top_k: int = 5) -> list[str]:
    """Find nodes whose titles or summaries match query keywords.

    Tokenizes the query, scores each node by keyword overlap with its
    title and summary, and returns the top-k node_ids sorted by score.

    Args:
        tree: Paper tree structure
        query: Search query string
        top_k: Maximum number of node IDs to return

    Returns:
        List of node_id strings, best matches first
    """
    keywords = tokenize_query(query)
    if not keywords:
        return []

    scored = []
    for node in structure_to_list(tree):
        nid = node.get("node_id")
        if not nid:
            continue

        title = node.get("title", "").lower()
        summary = (node.get("summary") or node.get("prefix_summary", "")).lower()
        combined = set(re.findall(r'[a-z0-9]+', title + " " + summary))

        overlap = len(keywords & combined)
        if overlap > 0:
            # Weight title matches higher
            title_words = set(re.findall(r'[a-z0-9]+', title))
            title_overlap = len(keywords & title_words)
            score = title_overlap * 2.0 + overlap
            scored.append((nid, score))

    scored.sort(key=lambda x: x[1], reverse=True)
    return [nid for nid, _ in scored[:top_k]]


def extract_section_text(
    node: dict, page_texts: list[str]
) -> tuple[str | None, str | None]:
    """Extract section-specific text from markdown page_texts using header parsing.

    Finds the node's title as a markdown header in the concatenated page text,
    then extracts only the text belonging to that section (up to the next
    header at the same or higher level).

    For parent nodes, also computes prefix_text: text from the matched header
    to the first child's header.

    Args:
        node: Tree node with title, start_index, end_index, and optional nodes.
        page_texts: Per-page text from parser (1-indexed via slicing).

    Returns:
        (section_body, prefix_text) — either may be None if no match found.
    """
    sp = node.get("start_index")
    ep = node.get("end_index")
    title = node.get("title", "")
    if sp is None or ep is None or not page_texts or not title:
        return None, None

    full_text = "\n\n".join(page_texts[sp - 1 : ep])

    # Find all markdown headers
    header_pattern = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)
    headers = list(header_pattern.finditer(full_text))

    if not headers:
        return None, None

    # Fuzzy-match node title to a header
    title_norm = normalize_text(title)
    matched_idx = None
    matched_level = None
    for i, m in enumerate(headers):
        header_text_norm = normalize_text(m.group(2))
        if title_norm in header_text_norm or header_text_norm in title_norm:
            matched_idx = i
            matched_level = len(m.group(1))  # number of # chars
            break

    if matched_idx is None:
        # Fallback for Abstract: find "abstract" as a standalone label at line start.
        # Requires start-of-line to avoid false positives in titles or sentences
        # (e.g., "An Abstract Model for..." won't match; "**Abstract**\n" will).
        if normalize_text(title) == "abstract":
            label_pat = re.compile(r'(?im)^\s*\*{0,2}\s*abstract\s*\*{0,2}\s*[:.]*')
            abstract_match = label_pat.search(full_text)
            if abstract_match:
                end_pos = len(full_text)
                next_header = header_pattern.search(full_text, abstract_match.end())
                if next_header:
                    end_pos = next_header.start()
                section = full_text[abstract_match.start():end_pos].strip()
                if section:
                    return section, None
        return None, None

    # Extract text from matched header to next header at same-or-higher level
    start_pos = headers[matched_idx].start()
    end_pos = len(full_text)
    for j in range(matched_idx + 1, len(headers)):
        next_level = len(headers[j].group(1))
        if next_level <= matched_level:
            end_pos = headers[j].start()
            break

    section_body = full_text[start_pos:end_pos].strip()

    # For parent nodes, compute prefix_text (header to first child's header)
    prefix_text = None
    children = node.get("nodes")
    if children:
        first_child_title = children[0].get("title", "")
        if first_child_title:
            first_child_norm = normalize_text(first_child_title)
            for j in range(matched_idx + 1, len(headers)):
                child_header_norm = normalize_text(headers[j].group(2))
                if first_child_norm in child_header_norm or child_header_norm in first_child_norm:
                    prefix_text = full_text[start_pos:headers[j].start()].strip()
                    break

    return section_body, prefix_text


def print_json(data, max_len=40, indent=2):
    """Print JSON with truncated long strings."""
    def simplify(obj):
        if isinstance(obj, dict):
            return {k: simplify(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [simplify(item) for item in obj]
        elif isinstance(obj, str) and len(obj) > max_len:
            return obj[:max_len] + "..."
        return obj

    print(json.dumps(simplify(data), indent=indent, ensure_ascii=False))

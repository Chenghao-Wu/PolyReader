"""Unit tests for tree_builder internals (no LLM calls needed)."""

import logging
import pytest

from polyreader.indexing.tree_builder import (
    _structure_to_tree,
    _validate_tree_invariants,
)
from polyreader.utils.constants import normalize_text
from polyreader.utils.tree_utils import extract_section_text


# ---------------------------------------------------------------------------
# _structure_to_tree
# ---------------------------------------------------------------------------

class TestStructureToTree:
    """Test flat structure -> hierarchical tree conversion."""

    def test_basic_hierarchy(self):
        """Top-level sections become roots, sub-sections become children."""
        structure = [
            {"structure": "1", "title": "Introduction", "page_num": 1, "appear_start": "yes"},
            {"structure": "2", "title": "Methods", "page_num": 3, "appear_start": "yes"},
            {"structure": "2.1", "title": "Sample Prep", "page_num": 3, "appear_start": "no"},
            {"structure": "2.2", "title": "Analysis", "page_num": 5, "appear_start": "yes"},
            {"structure": "3", "title": "Results", "page_num": 7, "appear_start": "yes"},
        ]
        tree = _structure_to_tree(structure, total_pages=10)

        assert len(tree) == 3
        assert tree[0]["title"] == "Introduction"
        assert tree[1]["title"] == "Methods"
        assert tree[2]["title"] == "Results"

        # Methods has 2 children
        assert len(tree[1]["nodes"]) == 2
        assert tree[1]["nodes"][0]["title"] == "Sample Prep"
        assert tree[1]["nodes"][1]["title"] == "Analysis"

    def test_appear_start_yes_clean_boundary(self):
        """appear_start='yes' gives clean break: end on prev page."""
        structure = [
            {"structure": "1", "title": "A", "page_num": 1, "appear_start": "yes"},
            {"structure": "2", "title": "B", "page_num": 5, "appear_start": "yes"},
        ]
        tree = _structure_to_tree(structure, total_pages=10)

        # A ends on page 4 (one before B starts)
        assert tree[0]["end_index"] == 4
        # B ends on last page
        assert tree[1]["end_index"] == 10

    def test_appear_start_no_shared_page(self):
        """appear_start='no' includes shared page in current section."""
        structure = [
            {"structure": "1", "title": "A", "page_num": 1, "appear_start": "yes"},
            {"structure": "2", "title": "B", "page_num": 5, "appear_start": "no"},
        ]
        tree = _structure_to_tree(structure, total_pages=10)

        # A includes the shared page 5
        assert tree[0]["end_index"] == 5
        assert tree[1]["start_index"] == 5
        assert tree[1]["end_index"] == 10

    def test_last_section_ends_at_total_pages(self):
        """Last section always ends at total_pages."""
        structure = [
            {"structure": "1", "title": "Only", "page_num": 3, "appear_start": "yes"},
        ]
        tree = _structure_to_tree(structure, total_pages=27)
        assert tree[0]["end_index"] == 27

    def test_empty_structure(self):
        """Empty structure returns empty tree."""
        assert _structure_to_tree([], total_pages=10) == []


# ---------------------------------------------------------------------------
# _validate_tree_invariants
# ---------------------------------------------------------------------------

class TestValidateTreeInvariants:
    """Test that validation detects invariant violations."""

    def test_valid_tree_no_warnings(self, caplog):
        """A valid tree produces no warnings."""
        tree = [
            {
                "title": "A", "start_index": 1, "end_index": 3,
                "nodes": [
                    {"title": "A1", "start_index": 2, "end_index": 3, "nodes": []},
                ],
            },
            {"title": "B", "start_index": 4, "end_index": 10, "nodes": []},
        ]
        with caplog.at_level(logging.WARNING):
            _validate_tree_invariants(tree)
        assert len(caplog.records) == 0

    def test_inverted_range(self, caplog):
        """start_index > end_index triggers a warning."""
        tree = [
            {"title": "Bad", "start_index": 5, "end_index": 3, "nodes": []},
        ]
        with caplog.at_level(logging.WARNING):
            _validate_tree_invariants(tree)
        assert any("Inverted range" in r.message for r in caplog.records)

    def test_child_before_parent(self, caplog):
        """child.start_index < parent.start_index triggers a warning."""
        tree = [
            {
                "title": "Parent", "start_index": 5, "end_index": 10,
                "nodes": [
                    {"title": "Child", "start_index": 3, "end_index": 8, "nodes": []},
                ],
            },
        ]
        with caplog.at_level(logging.WARNING):
            _validate_tree_invariants(tree)
        assert any("Child before parent" in r.message for r in caplog.records)

    def test_duplicate_siblings(self, caplog):
        """Same normalized title + same start_index triggers a warning."""
        tree = [
            {"title": "Rheological Measurements", "start_index": 18, "end_index": 18, "nodes": []},
            {"title": "Rheological Measurements", "start_index": 18, "end_index": 18, "nodes": []},
        ]
        with caplog.at_level(logging.WARNING):
            _validate_tree_invariants(tree)
        assert any("Duplicate sibling" in r.message for r in caplog.records)

    def test_different_titles_same_page_no_warning(self, caplog):
        """Different titles on same page are fine."""
        tree = [
            {"title": "Alpha", "start_index": 5, "end_index": 5, "nodes": []},
            {"title": "Beta", "start_index": 5, "end_index": 5, "nodes": []},
        ]
        with caplog.at_level(logging.WARNING):
            _validate_tree_invariants(tree)
        assert len(caplog.records) == 0


# ---------------------------------------------------------------------------
# Boundary trimming logic (effective_ep computation)
# ---------------------------------------------------------------------------

class TestBoundaryTrimming:
    """Test the effective_ep computation used in _split_large_nodes."""

    @staticmethod
    def _compute_effective_ep(nodes, index):
        """Replicate the boundary trimming logic from _split_large_nodes."""
        node = nodes[index]
        ep = node.get("end_index")
        effective_ep = ep
        if index + 1 < len(nodes):
            next_sp = nodes[index + 1].get("start_index", ep + 1)
            if ep >= next_sp:
                effective_ep = next_sp - 1
        return effective_ep

    def test_no_overlap_no_trim(self):
        """When ep < next_sp, effective_ep == ep (no trimming)."""
        nodes = [
            {"title": "A", "start_index": 1, "end_index": 5},
            {"title": "B", "start_index": 6, "end_index": 10},
        ]
        assert self._compute_effective_ep(nodes, 0) == 5

    def test_shared_page_trimmed(self):
        """When ep == next_sp (shared page), effective_ep = next_sp - 1."""
        nodes = [
            {"title": "SAXS", "start_index": 4, "end_index": 18},
            {"title": "Rheological", "start_index": 18, "end_index": 18},
        ]
        assert self._compute_effective_ep(nodes, 0) == 17

    def test_ep_exceeds_next_sp_trimmed(self):
        """When ep > next_sp (larger overlap), effective_ep = next_sp - 1."""
        nodes = [
            {"title": "A", "start_index": 1, "end_index": 20},
            {"title": "B", "start_index": 15, "end_index": 25},
        ]
        assert self._compute_effective_ep(nodes, 0) == 14

    def test_last_node_no_trim(self):
        """Last node in list has no next sibling — no trimming."""
        nodes = [
            {"title": "A", "start_index": 1, "end_index": 5},
            {"title": "B", "start_index": 6, "end_index": 10},
        ]
        assert self._compute_effective_ep(nodes, 1) == 10


# ---------------------------------------------------------------------------
# extract_section_text
# ---------------------------------------------------------------------------

class TestNormalizeText:
    """Test normalize_text strips HTML tags and normalizes text."""

    def test_strip_sup_tags(self):
        """<sup> tags are stripped so '13C' matches '<sup>13</sup>C'."""
        assert normalize_text("<sup>13</sup>C NMR") == normalize_text("13C NMR")

    def test_strip_sub_tags(self):
        """<sub> tags are stripped."""
        assert normalize_text("H<sub>2</sub>O") == normalize_text("H2O")

    def test_strip_span_tags(self):
        """<span> with attributes are stripped."""
        assert normalize_text('<span style="color:red">Important</span>') == "important"

    def test_nested_html_tags(self):
        """Nested tags are fully stripped."""
        assert normalize_text("<b><sup>13</sup>C</b> CP/MAS NMR") == "13c cp/mas nmr"

    def test_plain_text_unchanged(self):
        """Text without HTML is unaffected."""
        assert normalize_text("Simple Text") == "simple text"


class TestExtractSectionText:
    """Test section-specific text extraction from markdown page_texts."""

    def test_leaf_node_single_page(self):
        """Leaf node on a single page gets only its section text."""
        page_texts = [
            "### Introduction\n\nThis is the intro.\n\n### Methods\n\nWe used XRD and DSC."
        ]
        node = {"title": "Methods", "start_index": 1, "end_index": 1}
        body, prefix = extract_section_text(node, page_texts)

        assert body is not None
        assert "XRD and DSC" in body
        assert "This is the intro" not in body
        assert prefix is None  # leaf node, no children

    def test_parent_node_prefix_text(self):
        """Parent node gets prefix_text before first child heading."""
        page_texts = [
            "### Experimental\n\nGeneral methods here.\n\n#### Sample Prep\n\nPrep details.\n\n#### Analysis\n\nAnalysis details."
        ]
        children = [
            {"title": "Sample Prep", "start_index": 1, "end_index": 1},
            {"title": "Analysis", "start_index": 1, "end_index": 1},
        ]
        node = {"title": "Experimental", "start_index": 1, "end_index": 1, "nodes": children}
        body, prefix = extract_section_text(node, page_texts)

        assert body is not None
        assert "General methods" in body
        assert prefix is not None
        assert "General methods" in prefix
        # prefix should NOT contain child content
        assert "Prep details" not in prefix

    def test_title_not_found(self):
        """Title not matching any header returns (None, None)."""
        page_texts = [
            "### Introduction\n\nSome text.\n\n### Methods\n\nMore text."
        ]
        node = {"title": "Conclusion", "start_index": 1, "end_index": 1}
        body, prefix = extract_section_text(node, page_texts)

        assert body is None
        assert prefix is None

    def test_special_chars_in_headers(self):
        """Marker special chars like ■ in headers are handled."""
        page_texts = [
            "### ■ ASSOCIATED CONTENT\n\nSupporting info here.\n\n### ■ AUTHOR INFORMATION\n\nAuthor details."
        ]
        node = {"title": "ASSOCIATED CONTENT", "start_index": 1, "end_index": 1}
        body, prefix = extract_section_text(node, page_texts)

        assert body is not None
        assert "Supporting info" in body
        assert "Author details" not in body

    def test_html_tags_in_headers(self):
        """Headers with HTML tags like <sup> match plain-text titles."""
        page_texts = [
            "### <sup>13</sup>C NMR\n\nCarbon NMR data here.\n\n### FTIR\n\nInfrared results."
        ]
        node = {"title": "13C NMR", "start_index": 1, "end_index": 1}
        body, prefix = extract_section_text(node, page_texts)

        assert body is not None
        assert "Carbon NMR data" in body
        assert "Infrared results" not in body

    def test_no_headers_returns_none(self):
        """Page with no markdown headers returns (None, None)."""
        page_texts = ["Plain text with no headers at all."]
        node = {"title": "Something", "start_index": 1, "end_index": 1}
        body, prefix = extract_section_text(node, page_texts)

        assert body is None
        assert prefix is None

    def test_multi_page_extraction(self):
        """Section spanning multiple pages extracts correctly."""
        page_texts = [
            "### Introduction\n\nPage 1 intro content.",
            "More intro on page 2.\n\n### Methods\n\nMethods start here.",
        ]
        node = {"title": "Introduction", "start_index": 1, "end_index": 2}
        body, prefix = extract_section_text(node, page_texts)

        assert body is not None
        assert "Page 1 intro content" in body
        assert "More intro on page 2" in body
        assert "Methods start here" not in body



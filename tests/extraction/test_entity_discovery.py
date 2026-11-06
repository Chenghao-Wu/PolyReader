"""Unit tests for entity_discovery module (no LLM required)."""

import json
from unittest.mock import MagicMock

import pytest

from polyreader.extraction.entity_discovery import (
    build_entity_discovery_prompt,
    parse_entity_answer,
    validate_entity_map,
)
from polyreader.extraction.tools import dispatch_tool as dispatch_entity_tool
from polyreader.extraction.document_set import DocumentSet


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_index():
    """Build a minimal PaperIndex-like mock."""
    index = MagicMock()
    index.doc_name = "Test Paper"
    index.metadata.authors = ["Alice", "Bob"]
    index.page_texts = ["Page 1 text", "Page 2 text", "Page 3 text"]
    index.get_document_structure.return_value = [
        {"title": "Abstract", "start_index": 1, "end_index": 1},
        {"title": "Experimental", "start_index": 2, "end_index": 2},
        {"title": "Results", "start_index": 3, "end_index": 3},
    ]
    index.get_page_content.return_value = {
        "success": True,
        "content": [{"page": 1, "text": "Page 1 text"}],
    }
    index.get_page_image.return_value = {
        "success": True,
        "img_path": "_page_0_Picture_0.jpeg",
        "image_base64": "aGVsbG8=",  # "hello" in base64
        "format": "jpeg",
    }
    return index


@pytest.fixture
def mock_doc_set(mock_index):
    """Wrap mock_index in a DocumentSet."""
    return DocumentSet({"test_paper": mock_index})


def _good_entity_map():
    """Return a valid entity map dict."""
    return {
        "entities": {
            "PLLA": {
                "full_name": "poly(L-lactic acid)",
                "type": "linear",
                "extraction_class": "base",
                "depends_on": [],
                "aliases": [],
                "sample_ids": ["P1", "P2"],
                "triage_reason": "synthesized via ROP, Mn in Table 1",
            },
            "PDLA": {
                "full_name": "poly(D-lactic acid)",
                "type": "linear",
                "extraction_class": "base",
                "depends_on": [],
                "aliases": [],
                "sample_ids": ["P3"],
                "triage_reason": "synthesized via ROP, Mn in Table 1",
            },
            "SC-PLA": {
                "full_name": "stereocomplex PLA",
                "type": "blend",
                "extraction_class": "derived",
                "depends_on": ["PLLA", "PDLA"],
                "aliases": [],
                "sample_ids": ["B1"],
                "triage_reason": "blend of PLLA+PDLA, Tm in Table 2",
            },
        },
        "sample_list": ["P1", "P2", "P3", "B1"],
        "paper_polymer_focus": "PLA stereocomplex",
        "base_entities": ["PLLA", "PDLA"],
        "derived_entities": ["SC-PLA"],
    }


# ---------------------------------------------------------------------------
# Tests: build_entity_discovery_prompt
# ---------------------------------------------------------------------------


class TestBuildPrompt:
    def test_contains_all_sections(self, mock_doc_set):
        prompt = build_entity_discovery_prompt(mock_doc_set)
        for section in [
            "## 1. Role",
            "## 2. Entity Triage Gate",
            "## 3. Domain Knowledge",
            "## 4. JSON Output Schema",
            "## 5. Validation Rules",
            "## 6. Paper Metadata & Document Structure",
            "## 7. Execution Strategy",
        ]:
            assert section in prompt, f"Missing section: {section}"

    def test_contains_paper_metadata(self, mock_doc_set):
        prompt = build_entity_discovery_prompt(mock_doc_set)
        assert "Test Paper" in prompt
        assert "Alice" in prompt
        assert "Bob" in prompt

    def test_contains_document_structure(self, mock_doc_set):
        prompt = build_entity_discovery_prompt(mock_doc_set)
        assert "Abstract" in prompt
        assert "Experimental" in prompt
        assert "Results" in prompt


# ---------------------------------------------------------------------------
# Tests: parse_entity_answer
# ---------------------------------------------------------------------------


class TestParseEntityAnswer:
    def test_valid_json(self):
        answer = json.dumps(_good_entity_map())
        result = parse_entity_answer(answer)
        assert "PLLA" in result["entities"]
        assert result["base_entities"] == ["PLLA", "PDLA"]

    def test_backfills_extraction_class(self):
        answer = json.dumps({
            "entities": {
                "PCL": {
                    "full_name": "polycaprolactone",
                    "type": "linear",
                },
            },
        })
        result = parse_entity_answer(answer)
        pcl = result["entities"]["PCL"]
        assert pcl["extraction_class"] == "base"
        assert pcl["depends_on"] == []
        assert pcl["aliases"] == []
        assert pcl["sample_ids"] == []
        assert pcl["triage_reason"] == ""

    def test_backfills_derived_from_list(self):
        answer = json.dumps({
            "entities": {
                "A": {"full_name": "Polymer A", "type": "linear"},
                "B": {"full_name": "Blend B", "type": "blend"},
            },
            "derived_entities": ["B"],
            "base_entities": ["A"],
        })
        result = parse_entity_answer(answer)
        assert result["entities"]["B"]["extraction_class"] == "derived"
        assert result["entities"]["A"]["extraction_class"] == "base"

    def test_non_json_returns_empty(self):
        result = parse_entity_answer("This is not JSON at all.")
        assert result["entities"] == {}

    def test_backfills_sample_list(self):
        answer = json.dumps({"entities": {"X": {"full_name": "X", "type": "linear"}}})
        result = parse_entity_answer(answer)
        assert result["sample_list"] == []
        assert result["paper_polymer_focus"] == ""


# ---------------------------------------------------------------------------
# Tests: validate_entity_map
# ---------------------------------------------------------------------------


class TestValidateEntityMap:
    def test_good_map_passes(self):
        is_valid, issues = validate_entity_map(_good_entity_map())
        assert is_valid is True
        assert issues == []

    def test_empty_entities_fails(self):
        is_valid, issues = validate_entity_map({"entities": {}})
        assert is_valid is False
        assert any("No entities found" in i for i in issues)

    def test_missing_full_name(self):
        m = _good_entity_map()
        del m["entities"]["PLLA"]["full_name"]
        is_valid, issues = validate_entity_map(m)
        assert is_valid is False
        assert any("missing full_name" in i for i in issues)

    def test_missing_type(self):
        m = _good_entity_map()
        del m["entities"]["PLLA"]["type"]
        is_valid, issues = validate_entity_map(m)
        assert is_valid is False
        assert any("missing type" in i for i in issues)

    def test_missing_sample_ids(self):
        m = _good_entity_map()
        m["entities"]["PLLA"]["sample_ids"] = []
        is_valid, issues = validate_entity_map(m)
        assert is_valid is False
        assert any("no sample_ids" in i for i in issues)

    def test_derived_missing_depends_on(self):
        m = _good_entity_map()
        m["entities"]["SC-PLA"]["depends_on"] = []
        is_valid, issues = validate_entity_map(m)
        assert is_valid is False
        assert any("missing depends_on" in i for i in issues)

    def test_derived_broken_ref(self):
        m = _good_entity_map()
        m["entities"]["SC-PLA"]["depends_on"] = ["NONEXISTENT"]
        is_valid, issues = validate_entity_map(m)
        assert is_valid is False
        assert any("NONEXISTENT" in i for i in issues)

    def test_duplicate_sample_ids(self):
        m = _good_entity_map()
        m["entities"]["PDLA"]["sample_ids"] = ["P1"]  # P1 also in PLLA
        is_valid, issues = validate_entity_map(m)
        assert is_valid is False
        assert any("Duplicate" in i for i in issues)

    def test_entity_missing_from_base_derived_lists(self):
        m = _good_entity_map()
        m["base_entities"] = ["PLLA"]  # missing PDLA
        is_valid, issues = validate_entity_map(m)
        assert is_valid is False
        assert any("not in base_entities" in i for i in issues)

    def test_extra_in_base_derived_lists(self):
        m = _good_entity_map()
        m["base_entities"].append("GHOST")
        is_valid, issues = validate_entity_map(m)
        assert is_valid is False
        assert any("non-existent" in i for i in issues)


# ---------------------------------------------------------------------------
# Tests: dispatch_entity_tool
# ---------------------------------------------------------------------------


class TestDispatchEntityTool:
    def test_get_page_content(self, mock_doc_set):
        result_str, _ = dispatch_entity_tool(mock_doc_set, "get_page_content", {"pages": "1"})
        result = json.loads(result_str)
        assert result["success"] is True

    def test_get_page_image_strips_base64(self, mock_doc_set):
        result_str, _ = dispatch_entity_tool(
            mock_doc_set, "get_page_image", {"img_path": "_page_0_Picture_0.jpeg"}, vision_enabled=True,
        )
        result = json.loads(result_str)
        assert result["success"] is True
        assert "image_base64" not in result
        assert result["image_rendered"] is True
        assert "visual context" in result["note"].lower()

    def test_get_document_structure(self, mock_doc_set):
        result_str, _ = dispatch_entity_tool(mock_doc_set, "get_document_structure", {})
        result = json.loads(result_str)
        assert isinstance(result, list)
        assert result[0]["title"] == "Abstract"

    def test_unknown_tool(self, mock_doc_set):
        result_str, _ = dispatch_entity_tool(mock_doc_set, "nonexistent_tool", {})
        result = json.loads(result_str)
        assert "error" in result

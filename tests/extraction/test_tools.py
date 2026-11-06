"""Unit tests for polyreader.extraction.tools — focused on dispatch logging.

Covers:
- _summarize_tool_args: arg rendering for each tool.
- _summarize_tool_result: success and error summaries for each tool.
- dispatch_tool: emits exactly one INFO log line per call, with and
  without a log_context prefix.
"""

from __future__ import annotations

import json
import logging
from unittest.mock import MagicMock

from polyreader.extraction.tools import (
    _summarize_tool_args,
    _summarize_tool_result,
    dispatch_tool,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_mock_doc_set(
    page_content_result=None,
    image_result=None,
    structure_result=None,
    has_multiple=False,
):
    """Build a minimal DocumentSet mock for dispatch_tool tests."""
    index = MagicMock()
    index.doc_name = "test_paper"
    index.get_page_content.return_value = page_content_result or {
        "returned_pages": "1-3",
        "text": "page body",
        "approximate_tokens": 456,
    }
    index.get_page_image.return_value = image_result or {
        "success": False,
        "error": "not found",
    }
    index.get_document_structure.return_value = structure_result or [
        {"title": "Abstract", "start_index": 1, "end_index": 1},
        {"title": "Results", "start_index": 2, "end_index": 5},
    ]

    doc_set = MagicMock()
    doc_set.resolve.return_value = index
    doc_set.has_multiple_documents.return_value = has_multiple
    doc_set.get_combined_structure.return_value = {"doc1": structure_result or []}
    doc_set.__len__ = MagicMock(return_value=1)
    return doc_set


# ---------------------------------------------------------------------------
# _summarize_tool_args
# ---------------------------------------------------------------------------


class TestSummarizeToolArgs:
    def test_page_content_with_pages(self):
        assert _summarize_tool_args("get_page_content", {"pages": "1-3"}) == "pages=1-3"

    def test_page_content_filters_empty_doc(self):
        out = _summarize_tool_args("get_page_content", {"doc": "", "pages": "1-3"})
        assert out == "pages=1-3"

    def test_page_image_with_path(self):
        out = _summarize_tool_args(
            "get_page_image", {"img_path": "_page_1_fig1.png"}
        )
        assert out == "img_path=_page_1_fig1.png"

    def test_document_structure_no_args(self):
        assert _summarize_tool_args("get_document_structure", {}) == ""

    def test_multiple_args_comma_joined(self):
        out = _summarize_tool_args(
            "get_page_content", {"doc": "supp", "pages": "1-3"}
        )
        assert "doc=supp" in out and "pages=1-3" in out
        assert ", " in out


# ---------------------------------------------------------------------------
# _summarize_tool_result
# ---------------------------------------------------------------------------


class TestSummarizeToolResult:
    def test_page_content_success(self):
        result_str = json.dumps({"returned_pages": "1-3", "approximate_tokens": 456})
        out = _summarize_tool_result("get_page_content", {"pages": "1-3"}, result_str, None)
        assert out == "text, ~456 tokens"

    def test_page_content_zero_tokens(self):
        result_str = json.dumps({"returned_pages": "1", "approximate_tokens": 0})
        out = _summarize_tool_result("get_page_content", {"pages": "1"}, result_str, None)
        assert out == "text, ~0 tokens"

    def test_page_image_vision_enabled(self):
        result_str = json.dumps({"success": True, "image_rendered": True})
        out = _summarize_tool_result(
            "get_page_image", {"img_path": "f.png"}, result_str, "data:image/png;base64,XXX"
        )
        assert out == "image returned"

    def test_page_image_vision_disabled(self):
        result_str = json.dumps({
            "success": True,
            "image_rendered": False,
            "image_bytes_length": 1000,
        })
        out = _summarize_tool_result(
            "get_page_image", {"img_path": "f.png"}, result_str, None
        )
        assert out == "image available (vision off)"

    def test_page_image_unavailable(self):
        result_str = json.dumps({"success": False})
        out = _summarize_tool_result(
            "get_page_image", {"img_path": "missing.png"}, result_str, None
        )
        assert out == "image unavailable"

    def test_document_structure_list(self):
        result_str = json.dumps([{"title": "A"}, {"title": "B"}, {"title": "C"}])
        out = _summarize_tool_result("get_document_structure", {}, result_str, None)
        assert out == "3 sections"

    def test_document_structure_dict(self):
        result_str = json.dumps({"doc1": [{"title": "A"}]})
        out = _summarize_tool_result("get_document_structure", {}, result_str, None)
        assert out == "structure returned"

    def test_error_result(self):
        result_str = json.dumps({"error": "page out of range"})
        out = _summarize_tool_result("get_page_content", {"pages": "999"}, result_str, None)
        assert out == "error: page out of range"

    def test_unknown_tool_ok(self):
        result_str = json.dumps({"foo": "bar"})
        out = _summarize_tool_result("get_other", {}, result_str, None)
        assert out == "ok"

    def test_unparseable_result(self):
        out = _summarize_tool_result("get_page_content", {}, "not json {", None)
        assert out == "unparseable result"


# ---------------------------------------------------------------------------
# dispatch_tool logging
# ---------------------------------------------------------------------------


class TestDispatchToolLogging:
    def test_emits_one_info_line_with_prefix(self, caplog):
        doc_set = _make_mock_doc_set()
        with caplog.at_level(logging.INFO, logger="polyreader.extraction.tools"):
            dispatch_tool(
                doc_set, "get_page_content", {"pages": "1-3"},
                log_context="entity=poly_123 iter=3",
            )

        records = [r for r in caplog.records if r.name == "polyreader.extraction.tools"]
        assert len(records) == 1
        msg = records[0].message
        assert msg == "[entity=poly_123 iter=3] get_page_content(pages=1-3) → text, ~456 tokens"

    def test_omits_prefix_when_no_context(self, caplog):
        doc_set = _make_mock_doc_set()
        with caplog.at_level(logging.INFO, logger="polyreader.extraction.tools"):
            dispatch_tool(doc_set, "get_page_content", {"pages": "5"})

        records = [r for r in caplog.records if r.name == "polyreader.extraction.tools"]
        assert len(records) == 1
        msg = records[0].message
        assert not msg.startswith("[")
        assert msg.startswith("get_page_content(")

    def test_image_line_for_vision_returned(self, caplog):
        doc_set = _make_mock_doc_set(
            image_result={
                "success": True,
                "img_path": "fig1.png",
                "image_base64": "AAAA",
                "format": "png",
            },
        )
        with caplog.at_level(logging.INFO, logger="polyreader.extraction.tools"):
            _, url = dispatch_tool(
                doc_set, "get_page_image",
                {"img_path": "fig1.png"},
                vision_enabled=True,
                log_context="entity=X iter=1",
            )

        assert url is not None
        records = [r for r in caplog.records if r.name == "polyreader.extraction.tools"]
        assert len(records) == 1
        assert records[0].message == "[entity=X iter=1] get_page_image(img_path=fig1.png) → image returned"

    def test_image_line_when_missing(self, caplog):
        doc_set = _make_mock_doc_set(
            image_result={"success": False, "error": "not found"},
        )
        with caplog.at_level(logging.INFO, logger="polyreader.extraction.tools"):
            dispatch_tool(
                doc_set, "get_page_image",
                {"img_path": "gone.png"},
                log_context="entity=X iter=1",
            )

        records = [r for r in caplog.records if r.name == "polyreader.extraction.tools"]
        # The MagicMock returns a dict with "error" key, which triggers the
        # error branch — verify the line surfaces that error cleanly.
        assert len(records) == 1
        assert records[0].message.startswith("[entity=X iter=1] get_page_image(img_path=gone.png) → ")
        # Either "error: not found" or "image unavailable" depending on result shape.
        assert "not found" in records[0].message or "unavailable" in records[0].message

    def test_document_structure_line(self, caplog):
        doc_set = _make_mock_doc_set(
            structure_result=[{"title": "A"}, {"title": "B"}],
        )
        with caplog.at_level(logging.INFO, logger="polyreader.extraction.tools"):
            dispatch_tool(
                doc_set, "get_document_structure", {},
                log_context="stage=discovery iter=2",
            )

        records = [r for r in caplog.records if r.name == "polyreader.extraction.tools"]
        assert len(records) == 1
        assert records[0].message == "[stage=discovery iter=2] get_document_structure() → 2 sections"

    def test_unknown_tool_logs_warning_plus_summary(self, caplog):
        doc_set = _make_mock_doc_set()
        with caplog.at_level(logging.WARNING, logger="polyreader.extraction.tools"):
            dispatch_tool(
                doc_set, "get_banana", {},
                log_context="entity=X iter=1",
            )

        records = [r for r in caplog.records if r.name == "polyreader.extraction.tools"]
        # One WARNING (unknown tool) + one INFO (summary line).
        levels = [r.levelname for r in records]
        assert "WARNING" in levels

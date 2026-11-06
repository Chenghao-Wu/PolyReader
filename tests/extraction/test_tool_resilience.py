"""Regression tests: malformed tool calls, document routing, failure isolation, outputs."""

from __future__ import annotations

import asyncio
import dataclasses
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import openai
import pytest

from polyreader.indexing import PaperIndex
from polyreader.llm.client import ChatWithToolsResponse, ToolCallResult
from polyreader.extraction.config import ExtractionConfig
from polyreader.extraction.document_set import DocumentSet, UnknownDocumentError
from polyreader.extraction.entity_discovery import discover_entities, parse_entity_answer
from polyreader.extraction.extraction_loop import EntityExtractionOutcome
from polyreader.extraction.orchestrator import extract, save_samples
from polyreader.schema.polymer import PolymerSample
from polyreader.extraction.tools import dispatch_tool, parse_tool_arguments


def _index(name: str, pages: int = 3) -> PaperIndex:
    return PaperIndex(
        tree=[{"title": "Results", "start_index": 1, "end_index": pages, "node_id": "0000"}],
        doc_name=name,
        page_texts=[f"{name} page {i}" for i in range(1, pages + 1)],
    )


def _reply(content="", tool_calls=()) -> ChatWithToolsResponse:
    calls = [ToolCallResult(id=f"c{i}", name=n, arguments=a) for i, (n, a) in enumerate(tool_calls)]
    raw = {"role": "assistant", "content": content}
    if calls:
        raw["tool_calls"] = [
            {"id": c.id, "type": "function", "function": {"name": c.name, "arguments": c.arguments}}
            for c in calls
        ]
    return ChatWithToolsResponse(content=content, tool_calls=calls, raw_message_dict=raw)


# ---------------------------------------------------------------------------
# Tool arguments
# ---------------------------------------------------------------------------


class TestParseToolArguments:
    @pytest.mark.parametrize("raw,expected", [
        ('{"pages": "1-2"}', {"pages": "1-2"}),
        ("", {}),
        (None, {}),
        ({"pages": "1"}, {"pages": "1"}),
        ('```json\n{"pages": "3"}\n```', {"pages": "3"}),
        ('{"pages": "3",}', {"pages": "3"}),
    ])
    def test_valid(self, raw, expected):
        assert parse_tool_arguments(raw) == (expected, None)

    @pytest.mark.parametrize("raw", ['{"pages": ', "not json", "[1, 2]", "42", 7])
    def test_invalid(self, raw):
        args, error = parse_tool_arguments(raw)
        assert args == {} and error


# ---------------------------------------------------------------------------
# Document routing
# ---------------------------------------------------------------------------


class TestRouting:
    def test_multi_doc_unknown_label_is_reported_not_misrouted(self):
        ds = DocumentSet({"main": _index("main.pdf"), "si": _index("si.pdf")})
        result = json.loads(dispatch_tool(ds, "get_page_content", {"doc": "supporting", "pages": "1"})[0])
        assert result["success"] is False
        assert result["available_documents"] == ["main", "si"]
        assert "main" in result["error"] and "si" in result["error"]

    def test_multi_doc_resolves_by_file_name(self):
        ds = DocumentSet({"main": _index("paper.pdf"), "si": _index("paper_SI.pdf")})
        result = json.loads(dispatch_tool(ds, "get_page_content", {"doc": "paper_SI.pdf", "pages": "1"})[0])
        assert result["content"][0]["text"].startswith("paper_SI.pdf")

    def test_multi_doc_structure_ignores_bad_label(self):
        ds = DocumentSet({"main": _index("main.pdf"), "si": _index("si.pdf")})
        result = json.loads(dispatch_tool(ds, "get_document_structure", {"doc": "whatever"})[0])
        assert set(result) == {"main", "si"}

    def test_single_doc_unchanged_fallback(self):
        ds = DocumentSet({"paper": _index("paper.pdf")})
        result = json.loads(dispatch_tool(ds, "get_page_content", {"doc": "si", "pages": "1"})[0])
        assert result["success"] is True

    def test_resolve_strict_and_legacy(self):
        ds = DocumentSet({"main": _index("main.pdf"), "si": _index("si.pdf")})
        with pytest.raises(UnknownDocumentError):
            ds.resolve("nope", strict=True)
        assert ds.resolve("nope") is ds.primary  # public API default unchanged
        assert ds.resolve("SI") is ds["si"]

    def test_failing_tool_returns_error(self):
        index = _index("paper.pdf")
        index.get_page_content = MagicMock(side_effect=RuntimeError("disk gone"))
        ds = DocumentSet({"paper": index})
        result = json.loads(dispatch_tool(ds, "get_page_content", {"pages": "1"})[0])
        assert result["success"] is False and "disk gone" in result["error"]

    def test_unknown_tool_lists_tools(self):
        ds = DocumentSet({"paper": _index("paper.pdf")})
        result = json.loads(dispatch_tool(ds, "get_table", {})[0])
        assert "Unknown tool" in result["error"] and "get_page_content" in result["available_tools"]


# ---------------------------------------------------------------------------
# Entity discovery resilience
# ---------------------------------------------------------------------------


_GOOD_MAP = json.dumps({
    "entities": {"PLLA": {"full_name": "poly(L-lactide)", "type": "linear",
                          "extraction_class": "base", "sample_ids": ["P1"]}},
    "base_entities": ["PLLA"], "derived_entities": [],
})


class TestDiscovery:
    @pytest.mark.asyncio
    async def test_malformed_tool_arguments_do_not_crash(self):
        llm = MagicMock()
        llm.chat_with_tools = AsyncMock(side_effect=[
            _reply(tool_calls=[("get_page_content", '{"pages": ')]),
            _reply(tool_calls=[("get_page_content", '{"pages": "1"}')]),
            _reply(content=_GOOD_MAP),
        ])
        ds = DocumentSet.from_index(_index("paper.pdf"))
        entity_map = await discover_entities(ds, llm, ExtractionConfig(api_key="k"))
        assert "PLLA" in entity_map["entities"]
        second_call_messages = llm.chat_with_tools.await_args_list[1].kwargs["messages"]
        tool_msg = next(m for m in second_call_messages if m.get("role") == "tool")
        assert "Invalid JSON" in tool_msg["content"]

    @pytest.mark.asyncio
    async def test_transport_failure_everywhere_raises(self):
        llm = MagicMock()
        llm.chat_with_tools = AsyncMock(side_effect=openai.APIConnectionError(request=MagicMock()))
        ds = DocumentSet.from_index(_index("paper.pdf"))
        with patch("polyreader.extraction.entity_discovery.asyncio.sleep", new=AsyncMock()):
            with pytest.raises(openai.APIConnectionError):
                await discover_entities(ds, llm, ExtractionConfig(api_key="k"))

    def test_unexpected_answer_shapes_normalized(self):
        answer = json.dumps({
            "entities": [
                {"abbreviation": "PCL", "full_name": "polycaprolactone", "type": "linear",
                 "sample_ids": "PCL-1", "depends_on": None},
                {"name": "Blend", "extraction_class": "derived", "sample_ids": [1, 2, 2],
                 "depends_on": "PCL"},
                "garbage",
            ],
            "base_entities": "PCL",
        })
        parsed = parse_entity_answer(answer)
        assert parsed["entities"]["PCL"]["sample_ids"] == ["PCL-1"]  # not ['P','C','L','-','1']
        assert parsed["entities"]["Blend"]["sample_ids"] == ["1", "2"]
        assert parsed["entities"]["Blend"]["depends_on"] == ["PCL"]
        assert parsed["base_entities"] == ["PCL"]
        assert parsed["derived_entities"] == ["Blend"]  # missing wave entry repaired

    def test_entities_not_listed_in_waves_are_added(self):
        answer = json.dumps({
            "entities": {"A": {"extraction_class": "base"}, "B": {"extraction_class": "base"}},
            "base_entities": ["A"], "derived_entities": [],
        })
        assert parse_entity_answer(answer)["base_entities"] == ["A", "B"]


# ---------------------------------------------------------------------------
# Guided extraction: malformed tool arguments
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_guided_loop_survives_malformed_arguments():
    from polyreader.extraction.guided_extraction import extract_entity

    llm = MagicMock()
    llm.chat_with_tools = AsyncMock(side_effect=[
        _reply(tool_calls=[("get_page_content", "{pages: 1}"), ("get_page_content", '{"pages": "2"}')]),
        _reply(content='{"sample_id": "P1"}'),
    ])
    llm.chat_structured = AsyncMock(return_value=PolymerSample(sample_id="P1"))
    llm._is_reasoner_model = MagicMock(return_value=False)
    llm._uses_thinking_mode = MagicMock(return_value=False)
    ds = DocumentSet.from_index(_index("paper.pdf"))
    result = await extract_entity(ds, llm, "PLLA", {"extraction_class": "base"}, {}, ExtractionConfig(api_key="k"))
    assert result["sample_id"] == "P1"
    messages = llm.chat_with_tools.await_args_list[1].kwargs["messages"]
    tool_msgs = [m for m in messages if m.get("role") == "tool"]
    assert len(tool_msgs) == 2  # every tool_call_id gets a response
    assert json.loads(tool_msgs[0]["content"])["success"] is False
    assert json.loads(tool_msgs[1]["content"])["success"] is True


# ---------------------------------------------------------------------------
# Orchestrator: failure isolation, config, outputs
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_one_failing_sample_does_not_stop_others(tmp_path):
    entity_map = {
        "entities": {"PLLA": {"full_name": "PLLA", "type": "linear", "extraction_class": "base",
                              "sample_ids": ["P1", "P2", "P3"]}},
        "base_entities": ["PLLA"], "derived_entities": [],
    }

    async def fake_extract(**kwargs):
        sid = kwargs["target_sample_id"]
        if sid == "P2":
            raise RuntimeError("unexpected bug")
        return EntityExtractionOutcome.SUCCESS, PolymerSample(sample_id=sid, quality_tier="bronze"), []

    config = ExtractionConfig(api_key="k", guided_max_iterations=12)
    ds = DocumentSet({"main": _index("main.pdf"), "si": _index("si.pdf")})
    fake_llm = SimpleNamespace(model="m", usage={"requests": 0})
    with patch("polyreader.extraction.orchestrator.discover_entities", new=AsyncMock(return_value=entity_map)), \
         patch("polyreader.extraction.orchestrator.extract_single_entity", new=AsyncMock(side_effect=fake_extract)), \
         patch("polyreader.extraction.orchestrator.create_llm_client", return_value=fake_llm):
        result = await extract(ds, config)
    assert sorted(s.sample_id for s in result.samples) == ["P1", "P3"]
    base_wave = next(t for t in result.extraction_trace if t.get("wave") == "base_entities")
    assert base_wave["outcomes"]["P2"] == "fatal"
    assert config.guided_max_iterations == 12  # caller's config not mutated
    assert any(t.get("agent") == "llm_usage" for t in result.extraction_trace)


def test_save_samples_unique_names(tmp_path):
    samples = [PolymerSample(sample_id="P-1"), PolymerSample(sample_id="P_1"), PolymerSample(sample_id="P 1")]
    out = save_samples(samples, str(tmp_path), "paper")
    names = sorted(p.name for p in out.iterdir())
    assert names == ["paper_P-1.json", "paper_P_1.json", "paper_P_1_2.json"]


@pytest.mark.asyncio
async def test_discovery_wrap_up_transport_error_uses_next_attempt():
    """A transient error in the max-iterations wrap-up must not abort the paper."""
    llm = MagicMock()
    llm.chat_with_tools = AsyncMock(side_effect=[
        _reply(tool_calls=[("get_page_content", '{"pages": "1"}')]),   # attempt 1, iteration 1
        openai.APIConnectionError(request=MagicMock()),                 # attempt 1 wrap-up
        _reply(content=_GOOD_MAP),                                      # attempt 2
    ])
    ds = DocumentSet.from_index(_index("paper.pdf"))
    config = ExtractionConfig(api_key="k", entity_max_iterations=1)
    with patch("polyreader.extraction.entity_discovery.asyncio.sleep", new=AsyncMock()):
        entity_map = await discover_entities(ds, llm, config)
    assert "PLLA" in entity_map["entities"]

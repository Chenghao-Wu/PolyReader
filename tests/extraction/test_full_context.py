"""Full-context extraction lane (LLM mocked)."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from polyreader.indexing import PaperIndex
from polyreader.extraction.config import ExtractionConfig
from polyreader.extraction.document_set import DocumentSet
from polyreader.extraction.extraction_loop import EntityExtractionOutcome
from polyreader.extraction.full_context import (
    BatchStats,
    _assign_items,
    build_batch_prompt,
    choose_extraction_mode,
    discover_entities_full_context,
    extract_batch,
    group_tasks,
    pack_document_set,
)
from polyreader.extraction.orchestrator import ExtractionTask, extract
from polyreader.schema.polymer import PolymerSample


def _index(name: str, pages: list[str]) -> PaperIndex:
    return PaperIndex(tree=[{"title": "All", "start_index": 1, "end_index": len(pages), "node_id": "0000"}],
                      doc_name=name, page_texts=pages)


def _sample(sid, mn=10000, mw=15000, pdi=1.5):
    return {
        "sample_id": sid, "polymer_id": "poly(L-lactide)", "topology": "linear",
        "constituents": [{"name": "L-lactide", "smiles": "C[C@H]1C(=O)O[C@H](C)C(=O)O1"}],
        "molecular_weight": {"mn": mn, "mw": mw, "pdi": pdi,
                             "provenance": {"page_numbers": [2]}},
        "properties": [{"property_name": "Tg", "value": 55, "unit": "°C"}],
    }


INFO = {"full_name": "poly(L-lactide)", "type": "linear", "extraction_class": "base",
        "sample_ids": ["S1", "S2", "S3"],
        "evidence": {"S1": [{"doc": "paper", "pages": [2], "kind": "table", "ref": "Table 1"}]}}


def _tasks(ids=("S1", "S2", "S3")):
    return [ExtractionTask(entity_id="PLLA", entity_info=INFO, target_sample_id=s, is_multi_sample=True)
            for s in ids]


# ---------------------------------------------------------------------------
# Packing and mode
# ---------------------------------------------------------------------------


class TestPacking:
    def test_deterministic_with_page_markers(self):
        ds = DocumentSet({"main": _index("p.pdf", ["A", "B"]), "si": _index("s.pdf", ["C"])})
        packed = pack_document_set(ds)
        assert packed == pack_document_set(ds)
        assert '<<doc="main" page=2>>\nB' in packed and '<<doc="si" page=1>>\nC' in packed
        assert "supplementary documents" in packed

    def test_mode_selection(self):
        small = "word " * 100
        assert choose_extraction_mode(ExtractionConfig(api_key="k"), small)[0] == "agentic"
        cfg = ExtractionConfig(api_key="k", extraction_mode="auto", full_context_budget=50)
        assert choose_extraction_mode(cfg, small)[0] == "agentic"
        cfg.full_context_budget = 10_000
        assert choose_extraction_mode(cfg, small)[0] == "full_context"
        cfg.extraction_mode = "full_context"
        cfg.full_context_budget = 1
        assert choose_extraction_mode(cfg, small)[0] == "full_context"  # explicit mode wins
        with pytest.raises(ValueError):
            choose_extraction_mode(ExtractionConfig(api_key="k", extraction_mode="fast"), small)


class TestHelpers:
    def test_assign_items(self):
        items = [{"sample_id": "s2"}, {"sample_id": "S1"}]
        assert set(_assign_items(items, ["S1", "S2"])) == {"S1", "S2"}
        renamed = _assign_items([{"sample_id": "Table 1 run 1"}, {"sample_id": "Table 1 run 2"}], ["run 1", "run 2"])
        assert renamed["run 2"]["sample_id"] == "Table 1 run 2"
        partial = _assign_items([{"sample_id": "S1"}, {"sample_id": "x"}, {"sample_id": "y"}], ["S1", "S2"])
        assert set(partial) == {"S1"}  # ambiguous leftovers are not guessed

    def test_group_tasks(self):
        tasks = _tasks(("a", "b", "c", "d", "e")) + [ExtractionTask("PCL", {}, None)]
        batches = group_tasks(tasks, 2)
        assert [len(b) for b in batches] == [2, 2, 1, 1]
        assert batches[-1][0].entity_id == "PCL"

    def test_batch_prompt_reuses_core_sections(self):
        prompt = build_batch_prompt("PLLA", INFO, {"PLLA": INFO}, ["S1", "S2"])
        for section in ("## 1. Role", "## 3. Domain Ontology", "## 4. JSON Output Schema",
                        "## 5. Validation Rules", "## 6. Where the Data Is", '"samples"'):
            assert section in prompt
        assert "get_page_content" not in prompt  # no tools in this lane
        assert '"evidence"' not in prompt.split("## 2. Entity Context")[1].split("## 3.")[0]


# ---------------------------------------------------------------------------
# Batch extraction
# ---------------------------------------------------------------------------


def _llm(*answers):
    llm = MagicMock()
    llm.chat = AsyncMock(side_effect=[json.dumps(a) if not isinstance(a, Exception) else a for a in answers])
    return llm


async def _run_batch(llm, tasks=None):
    tasks = tasks or _tasks()
    fallback = AsyncMock(side_effect=lambda task, ctx: (task, EntityExtractionOutcome.SUCCESS,
                                                         PolymerSample(sample_id=f"agentic-{task.target_sample_id}"), []))
    stats = BatchStats()
    ds = DocumentSet.from_index(_index("paper.pdf", ["intro", "table"]))
    results = await extract_batch(tasks, {"PLLA": INFO}, ds, llm, ExtractionConfig(api_key="k"),
                                  "PAPER", None, asyncio.Semaphore(4), fallback, stats)
    return results, fallback, stats


class TestExtractBatch:
    @pytest.mark.asyncio
    async def test_success(self):
        results, fallback, stats = await _run_batch(_llm({"samples": [_sample("S1"), _sample("S2"), _sample("S3")]}))
        assert [r[1] for r in results] == [EntityExtractionOutcome.SUCCESS] * 3
        assert [r[2].sample_id for r in results] == ["S1", "S2", "S3"]
        assert results[0][2].quality_tier == "gold"
        fallback.assert_not_awaited()
        assert stats.batches == 1 and stats.batch_samples == 3

    @pytest.mark.asyncio
    async def test_missing_sample_falls_back(self):
        results, fallback, stats = await _run_batch(_llm({"samples": [_sample("S1"), _sample("S3")]}))
        assert results[1][2].sample_id == "agentic-S2"
        assert fallback.await_count == 1 and stats.agentic_fallbacks == 1

    @pytest.mark.asyncio
    async def test_critical_flag_single_retry(self):
        llm = _llm({"samples": [_sample("S1", mn=20000, mw=15000), _sample("S2"), _sample("S3")]},
                   {"samples": [_sample("S1", mn=15000, mw=20000, pdi=1.33)]})
        results, fallback, stats = await _run_batch(llm)
        assert results[0][2].molecular_weight.mn == 15000
        assert stats.single_retries == 1
        retry_prompt = llm.chat.await_args_list[1].args[0][1]["content"]
        assert "Critical validation issue" in retry_prompt

    @pytest.mark.asyncio
    async def test_critical_flag_kept_when_retry_also_flagged(self):
        bad = _sample("S1", mn=20000, mw=15000)
        results, _, _ = await _run_batch(_llm({"samples": [bad, _sample("S2"), _sample("S3")]}, {"samples": [bad]}))
        assert results[0][1] == EntityExtractionOutcome.SUCCESS
        assert any("Mw (" in f for f in results[0][2].validation_flags)

    @pytest.mark.asyncio
    async def test_skipped_and_invalid(self):
        bad_schema = _sample("S3")
        bad_schema["properties"] = "not a list"
        llm = _llm({"samples": [_sample("S1"), {"sample_id": "S2", "skipped": True, "reason": "cited"}, bad_schema]})
        results, fallback, stats = await _run_batch(llm)
        assert results[1][1] == EntityExtractionOutcome.SKIPPED
        assert results[2][2].sample_id == "agentic-S3"
        assert stats.reasons and "schema validation" in stats.reasons[0]

    @pytest.mark.asyncio
    async def test_failed_batch_falls_back_for_all(self):
        results, fallback, stats = await _run_batch(_llm(ValueError("boom")))
        assert fallback.await_count == 3 and stats.failed_batches == 1

    @pytest.mark.asyncio
    async def test_uses_json_mode_and_cached_prefix(self):
        llm = _llm({"samples": [_sample("S1"), _sample("S2"), _sample("S3")]})
        await _run_batch(llm)
        messages = llm.chat.await_args.args[0]
        assert messages[0] == {"role": "system", "content": "PAPER"}
        assert llm.chat.await_args.kwargs["json_mode"] is True


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


GOOD_MAP = {"entities": {"PLLA": {"full_name": "poly(L-lactide)", "type": "linear", "extraction_class": "base",
                                  "sample_ids": ["S1"],
                                  "evidence": {"S1": {"doc": "paper", "pages": [2], "kind": "table"}}}},
            "base_entities": ["PLLA"], "derived_entities": []}


class TestDiscovery:
    @pytest.mark.asyncio
    async def test_evidence_normalized(self):
        ds = DocumentSet.from_index(_index("paper.pdf", ["a", "b"]))
        entity_map = await discover_entities_full_context(ds, _llm(GOOD_MAP), "PAPER")
        assert entity_map["entities"]["PLLA"]["evidence"]["S1"][0]["pages"] == [2]

    @pytest.mark.asyncio
    async def test_unparseable_raises(self):
        llm = MagicMock()
        llm.chat = AsyncMock(return_value="no json here")
        with pytest.raises(ValueError):
            await discover_entities_full_context(DocumentSet.from_index(_index("p.pdf", ["a"])), llm, "PAPER")

    @pytest.mark.asyncio
    async def test_empty_map_retried_then_returned(self):
        llm = _llm({"entities": {}}, {"entities": {}})
        result = await discover_entities_full_context(DocumentSet.from_index(_index("p.pdf", ["a"])), llm, "PAPER")
        assert result["entities"] == {} and llm.chat.await_count == 2


# ---------------------------------------------------------------------------
# Orchestrator wiring
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_orchestrator_full_context_flow():
    entity_map = {
        "entities": {
            "PLLA": dict(INFO),
            "Blend": {"full_name": "PLLA blend", "type": "blend", "extraction_class": "derived",
                      "depends_on": ["PLLA"], "sample_ids": ["B1"], "evidence": {}},
        },
        "base_entities": ["PLLA"], "derived_entities": ["Blend"],
    }
    llm = MagicMock(model="deepseek-flash", usage={"requests": 0})
    llm.chat = AsyncMock(side_effect=[
        json.dumps({"samples": [_sample("S1"), _sample("S2"), _sample("S3")]}),
        json.dumps({"samples": [dict(_sample("B1"), topology="blend")]}),
    ])
    ds = DocumentSet.from_index(_index("paper.pdf", ["intro", "table"]))
    config = ExtractionConfig(api_key="k", extraction_mode="full_context", batch_size=4)
    with patch("polyreader.extraction.orchestrator.discover_entities_full_context", new=AsyncMock(return_value=entity_map)), \
         patch("polyreader.extraction.orchestrator.discover_entities", new=AsyncMock()) as agentic_discovery, \
         patch("polyreader.extraction.orchestrator.extract_single_entity", new=AsyncMock()) as agentic_extract, \
         patch("polyreader.extraction.orchestrator.create_llm_client", return_value=llm):
        result = await extract(ds, config)
    agentic_discovery.assert_not_awaited()
    agentic_extract.assert_not_awaited()
    assert sorted(s.sample_id for s in result.samples) == ["B1", "S1", "S2", "S3"]
    fc = next(t for t in result.extraction_trace if t.get("agent") == "full_context")
    assert fc["batches"] == 2 and fc["agentic_fallbacks"] == 0
    blend_prompt = llm.chat.await_args_list[1].args[0][1]["content"]
    assert "Base Polymer Data" in blend_prompt  # derived wave gets base context


def test_json_mode_only_for_deepseek():
    from polyreader.llm.client import LLMClient

    for base_url, expected in (("https://api.deepseek.com", True), ("https://openrouter.ai/api/v1", False)):
        llm = LLMClient(api_key="k", base_url=base_url, model="deepseek/deepseek-chat")
        create = AsyncMock(return_value=SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="{}", tool_calls=None),
                                     finish_reason="stop")], usage=None))
        llm._client = MagicMock()
        llm._client.chat.completions.create = create
        asyncio.run(llm.chat([{"role": "user", "content": "json"}], json_mode=True))
        assert ("response_format" in create.await_args.kwargs) is expected


def test_characterization_figures_attached(tmp_path):
    """Values only legible in a figure (e.g. Tg labels on DSC curves) need the image."""
    from polyreader.extraction.full_context import _relevant_figures, figure_inventory

    pages = ["Table 1 lists runs.",
             "![](fig26.png)\nSupplementary Figure 26. DSC of poly(EVP)ROP samples with different Mns obtained in Table 1.\n"
             "![](fig25.png)\nSupplementary Figure 25. GPC traces of the recycled monomer.\n"
             "![](logo.png)\n"]
    ds = DocumentSet.from_index(_index("si.pdf", pages))
    figs = figure_inventory(ds)
    assert [f.image for f in figs] == ["fig26.png", "fig25.png", "logo.png"]
    info = {"full_name": "poly(EVP)ROP", "aliases": [], "evidence": {
        "run 14": [{"doc": "si", "pages": [1], "kind": "table", "ref": "Table 1, run 14"}]}}
    chosen = _relevant_figures(figs, "poly(EVP)ROP", info, ["run 14"])
    assert [f.image for f in chosen] == ["fig26.png"]  # GPC caption names neither entity nor table

    pages = ["![](gpc.png)\nFigure 1. GPC traces of poly(EVP)ROP.\n![](dsc.png)\nFigure 2. DSC curves of poly(EVP)ROP."]
    ranked = _relevant_figures(figure_inventory(DocumentSet.from_index(_index("m.pdf", pages))),
                               "poly(EVP)ROP", info, ["run 14"])
    assert [f.image for f in ranked] == ["dsc.png", "gpc.png"]  # value-bearing plots first

"""Tests for transport-error retry routing across guided extraction callsites.

Mocks the LLMClient — no API calls. Verifies that transient transport errors
re-raised by guided_extraction trigger the existing retry+backoff machinery
in extraction_loop, while non-transport errors preserve current behavior.
"""

from __future__ import annotations

import pytest

from polyreader.extraction._transport import is_transport_error


class TestIsTransportError:
    @pytest.mark.parametrize("message", [
        "Connection error.",
        "Connection error",
        "connection refused",
        "Read timeout occurred",
        "Request timeout",
        "rate limit exceeded",
        "Service Unavailable",
        "503 Server Error",
        "502 Bad Gateway",
        "429 Too Many Requests",
        "Server overloaded",
        "Network is unreachable",
        "Client disconnected",
    ])
    def test_classifies_transport_messages(self, message: str) -> None:
        assert is_transport_error(Exception(message)) is True

    @pytest.mark.parametrize("message", [
        "Invalid JSON",
        "ValidationError: field required",
        "Permission denied",
        "Unknown tool name",
        "",
    ])
    def test_rejects_non_transport_messages(self, message: str) -> None:
        assert is_transport_error(Exception(message)) is False


from unittest.mock import AsyncMock, MagicMock


@pytest.fixture
def fake_llm() -> MagicMock:
    """LLMClient stub with awaitable chat_with_tools / chat_structured."""
    llm = MagicMock()
    llm.chat_with_tools = AsyncMock()
    llm.chat_structured = AsyncMock()
    llm._is_reasoner_model = MagicMock(return_value=False)
    llm.model = "fake/model"
    return llm


class TestGuidedLoopReraise:
    @pytest.mark.asyncio
    async def test_main_loop_reraises_transport_error(self, fake_llm: MagicMock) -> None:
        """A connection error from chat_with_tools must propagate, not be swallowed."""
        from polyreader.extraction.guided_extraction import extract_entity
        from polyreader.extraction.config import ExtractionConfig
        from polyreader.extraction.document_set import DocumentSet

        fake_llm.chat_with_tools.side_effect = RuntimeError("Connection error")

        doc_set = MagicMock(spec=DocumentSet)
        doc_set.labels = MagicMock(return_value=["doc1"])
        doc_set.has_vision_pages = MagicMock(return_value=False)
        doc_set.get_document_structure = MagicMock(return_value=[])

        config = ExtractionConfig()
        entity_info = {"abbreviation": "X", "type": "linear", "extraction_class": "base"}

        with pytest.raises(RuntimeError, match="Connection error"):
            await extract_entity(
                doc_set=doc_set,
                llm=fake_llm,
                entity_id="X",
                entity_info=entity_info,
                entity_map={"X": entity_info},
                config=config,
            )

    @pytest.mark.asyncio
    async def test_main_loop_swallows_non_transport_error(self, fake_llm: MagicMock) -> None:
        """A non-transport exception keeps the current behavior: log + break + return error dict."""
        from polyreader.extraction.guided_extraction import extract_entity
        from polyreader.extraction.config import ExtractionConfig
        from polyreader.extraction.document_set import DocumentSet

        fake_llm.chat_with_tools.side_effect = ValueError("malformed tool response")
        # Also stub structured output so the post-loop call returns an error dict, not a real sample.
        fake_llm.chat_structured.side_effect = ValueError("no usable conversation")

        doc_set = MagicMock(spec=DocumentSet)
        doc_set.labels = MagicMock(return_value=["doc1"])
        doc_set.has_vision_pages = MagicMock(return_value=False)
        doc_set.get_document_structure = MagicMock(return_value=[])

        config = ExtractionConfig()
        entity_info = {"abbreviation": "X", "type": "linear", "extraction_class": "base"}

        result = await extract_entity(
            doc_set=doc_set,
            llm=fake_llm,
            entity_id="X",
            entity_info=entity_info,
            entity_map={"X": entity_info},
            config=config,
        )
        assert isinstance(result, dict)
        assert "error" in result


class TestStructuredOutputReraise:
    @pytest.mark.asyncio
    async def test_structured_output_reraises_transport_error(
        self, fake_llm: MagicMock
    ) -> None:
        """chat_structured raising a transport error must propagate, not become an error dict."""
        from polyreader.extraction.guided_extraction import _extract_with_structured_output

        fake_llm.chat_structured.side_effect = RuntimeError("Read timeout")

        with pytest.raises(RuntimeError, match="Read timeout"):
            await _extract_with_structured_output(
                messages=[{"role": "user", "content": "hi"}],
                llm=fake_llm,
                entity_id="X",
                is_skipped=False,
            )

    @pytest.mark.asyncio
    async def test_structured_output_swallows_non_transport_error(
        self, fake_llm: MagicMock
    ) -> None:
        """Non-transport failure keeps the fallback error-dict behavior."""
        from polyreader.extraction.guided_extraction import _extract_with_structured_output

        fake_llm.chat_structured.side_effect = ValueError("bad schema")

        result = await _extract_with_structured_output(
            messages=[{"role": "user", "content": "hi"}],
            llm=fake_llm,
            entity_id="X",
            is_skipped=False,
        )
        assert result == {
            "sample_id": "",
            "polymer_id": "X",
            "error": "bad schema",
        }


class TestWrapUpReraise:
    @pytest.mark.asyncio
    async def test_wrap_up_reraises_transport_error(self, fake_llm: MagicMock) -> None:
        from polyreader.extraction.tools import wrap_up_conversation

        fake_llm.chat_with_tools.side_effect = RuntimeError("503 Service Unavailable")

        with pytest.raises(RuntimeError, match="503"):
            await wrap_up_conversation(fake_llm, messages=[], tools=[])

    @pytest.mark.asyncio
    async def test_wrap_up_swallows_non_transport_error(self, fake_llm: MagicMock) -> None:
        from polyreader.extraction.tools import wrap_up_conversation

        fake_llm.chat_with_tools.side_effect = ValueError("bad request")

        result = await wrap_up_conversation(fake_llm, messages=[], tools=[])
        assert result == ""


class TestExtractionLoopRetry:
    @pytest.mark.asyncio
    async def test_transport_error_on_first_attempt_triggers_retry(
        self, fake_llm: MagicMock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """First chat_with_tools attempt fails with connection error, second succeeds.

        Verifies the bug AMBA3 hit is fixed end-to-end: the extraction_loop
        retries the entity instead of accepting a reject-tier sample on attempt 1.
        """
        from polyreader.extraction.extraction_loop import (
            extract_single_entity,
            EntityExtractionOutcome,
        )
        from polyreader.extraction.config import ExtractionConfig
        from polyreader.extraction.document_set import DocumentSet

        # Patch extract_entity directly so we don't have to drive the whole
        # tool-calling protocol through the mock. First call raises, second
        # returns a minimal-but-valid raw_data dict.
        call_count = {"n": 0}

        async def fake_extract_entity(*args, **kwargs):
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise RuntimeError("Connection error")
            return {
                "sample_id": "X1",
                "polymer_id": "X",
                "topology": "linear",
                "constituents": [{"name": "monomer1", "smiles": None}],
                "molecular_weight": {"mn": 10000.0, "mw": 12000.0, "pdi": 1.2},
                "properties": [],
                "synthesis": {"method": "RAFT"},
                "processing_history": [],
            }

        monkeypatch.setattr(
            "polyreader.extraction.extraction_loop.extract_entity",
            fake_extract_entity,
        )
        # Make backoff sleep instant so the test runs fast.
        async def no_sleep(_seconds: float) -> None:
            return None
        monkeypatch.setattr("polyreader.extraction.extraction_loop.asyncio.sleep", no_sleep)

        doc_set = MagicMock(spec=DocumentSet)
        doc_set.labels = MagicMock(return_value=["doc1"])
        config = ExtractionConfig()
        entity_info = {"abbreviation": "X", "type": "linear", "extraction_class": "base"}

        outcome, sample, flags = await extract_single_entity(
            entity_id="X",
            entity_info=entity_info,
            entity_map={"X": entity_info},
            doc_set=doc_set,
            llm=fake_llm,
            config=config,
            max_retries=3,
        )

        assert outcome == EntityExtractionOutcome.SUCCESS
        assert sample is not None
        assert sample.sample_id == "X1"
        assert call_count["n"] == 2  # one failure + one success

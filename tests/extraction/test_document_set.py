"""Unit tests for DocumentSet (no LLM required)."""

from unittest.mock import MagicMock

import pytest

from polyreader.extraction.document_set import DocumentSet, _sanitize_label


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_mock_index(doc_name: str = "Test Paper", pages: int = 3):
    """Build a minimal PaperIndex-like mock."""
    index = MagicMock()
    index.doc_name = doc_name
    index.metadata.authors = ["Alice"]
    index.page_texts = [f"Page {i + 1}" for i in range(pages)]
    index.get_document_structure.return_value = [
        {"title": "Abstract", "start_index": 1, "end_index": 1},
    ]
    index._lazy_load_pages = MagicMock()
    return index


# ---------------------------------------------------------------------------
# Tests: _sanitize_label
# ---------------------------------------------------------------------------


class TestSanitizeLabel:
    def test_simple_name(self):
        assert _sanitize_label("Test Paper") == "test_paper"

    def test_special_chars(self):
        assert _sanitize_label("Paper (2024) [v2]") == "paper_2024_v2"

    def test_empty_string(self):
        assert _sanitize_label("") == "doc"

    def test_long_name_truncated(self):
        label = _sanitize_label("A" * 100)
        assert len(label) <= 40

    def test_unicode(self):
        label = _sanitize_label("Polymer Résearch — 2024")
        assert "r" in label  # basic chars survive


# ---------------------------------------------------------------------------
# Tests: DocumentSet creation
# ---------------------------------------------------------------------------


class TestDocumentSetCreation:
    def test_from_index(self):
        idx = _make_mock_index("Main Paper")
        ds = DocumentSet.from_index(idx)
        assert len(ds) == 1
        assert ds.primary is idx
        assert not ds.has_multiple_documents()

    def test_from_indices(self):
        main = _make_mock_index("Main Paper")
        si = _make_mock_index("Supplementary Info")
        ds = DocumentSet.from_indices(main, si)
        assert len(ds) == 2
        assert ds.primary is main
        assert ds.has_multiple_documents()

    def test_from_indices_single(self):
        idx = _make_mock_index("Only Paper")
        ds = DocumentSet.from_indices(idx)
        assert len(ds) == 1
        assert not ds.has_multiple_documents()

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            DocumentSet({})

    def test_from_indices_empty_raises(self):
        with pytest.raises(ValueError):
            DocumentSet.from_indices()

    def test_dict_constructor(self):
        idx = _make_mock_index("Paper")
        ds = DocumentSet({"main": idx})
        assert ds.labels() == ["main"]

    def test_collision_handling(self):
        """Two indices with the same doc_name should get different labels."""
        a = _make_mock_index("Paper")
        b = _make_mock_index("Paper")
        ds = DocumentSet.from_indices(a, b)
        assert len(ds) == 2
        labels = ds.labels()
        assert labels[0] != labels[1]

    def test_empty_doc_name(self):
        a = _make_mock_index("")
        b = _make_mock_index("")
        ds = DocumentSet.from_indices(a, b)
        assert len(ds) == 2


# ---------------------------------------------------------------------------
# Tests: resolve
# ---------------------------------------------------------------------------


class TestResolve:
    def test_resolve_valid_label(self):
        main = _make_mock_index("Main")
        si = _make_mock_index("SI")
        ds = DocumentSet({"main": main, "si": si})
        assert ds.resolve("si") is si
        assert ds.resolve("main") is main

    def test_resolve_none_returns_primary(self):
        main = _make_mock_index("Main")
        si = _make_mock_index("SI")
        ds = DocumentSet({"main": main, "si": si})
        assert ds.resolve(None) is main

    def test_resolve_empty_returns_primary(self):
        main = _make_mock_index("Main")
        ds = DocumentSet({"main": main})
        assert ds.resolve("") is main

    def test_resolve_unknown_returns_primary(self):
        main = _make_mock_index("Main")
        ds = DocumentSet({"main": main})
        assert ds.resolve("nonexistent") is main

    def test_resolve_case_insensitive(self):
        idx = _make_mock_index("Test")
        ds = DocumentSet({"main_paper": idx})
        assert ds.resolve("MAIN_PAPER") is idx


# ---------------------------------------------------------------------------
# Tests: get_combined_structure
# ---------------------------------------------------------------------------


class TestGetCombinedStructure:
    def test_single_doc(self):
        idx = _make_mock_index("Paper", pages=5)
        ds = DocumentSet.from_index(idx)
        struct = ds.get_combined_structure()
        assert len(struct) == 1
        label = list(struct.keys())[0]
        assert struct[label]["doc_name"] == "Paper"
        assert struct[label]["pages"] == 5
        assert "structure" in struct[label]

    def test_multi_doc(self):
        main = _make_mock_index("Main", pages=10)
        si = _make_mock_index("SI", pages=20)
        ds = DocumentSet({"main": main, "si": si})
        struct = ds.get_combined_structure()
        assert len(struct) == 2
        assert struct["main"]["pages"] == 10
        assert struct["si"]["pages"] == 20


# ---------------------------------------------------------------------------
# Tests: available_documents, labels, has_multiple_documents
# ---------------------------------------------------------------------------


class TestProperties:
    def test_labels_multiple(self):
        main = _make_mock_index("Main")
        si = _make_mock_index("SI")
        ds = DocumentSet({"main": main, "si": si})
        assert ds.labels() == ["main", "si"]

    def test_labels(self):
        idx = _make_mock_index("Paper")
        ds = DocumentSet({"paper": idx})
        assert ds.labels() == ["paper"]

    def test_has_multiple_documents_false(self):
        ds = DocumentSet.from_index(_make_mock_index("Paper"))
        assert ds.has_multiple_documents() is False

    def test_has_multiple_documents_true(self):
        ds = DocumentSet.from_indices(
            _make_mock_index("A"),
            _make_mock_index("B"),
        )
        assert ds.has_multiple_documents() is True


# ---------------------------------------------------------------------------
# Tests: ensure_pages_loaded
# ---------------------------------------------------------------------------


class TestEnsurePagesLoaded:
    def test_calls_lazy_load_on_all(self):
        a = _make_mock_index("A")
        b = _make_mock_index("B")
        ds = DocumentSet({"a": a, "b": b})
        ds.ensure_pages_loaded()
        a._lazy_load_pages.assert_called_once()
        b._lazy_load_pages.assert_called_once()


# ---------------------------------------------------------------------------
# Tests: iteration
# ---------------------------------------------------------------------------


class TestIteration:
    def test_len(self):
        ds = DocumentSet.from_indices(
            _make_mock_index("A"),
            _make_mock_index("B"),
        )
        assert len(ds) == 2

    def test_iter(self):
        a = _make_mock_index("A")
        b = _make_mock_index("B")
        ds = DocumentSet({"a": a, "b": b})
        items = list(ds)
        assert items[0] == ("a", a)
        assert items[1] == ("b", b)

    def test_getitem(self):
        idx = _make_mock_index("Paper")
        ds = DocumentSet({"paper": idx})
        assert ds["paper"] is idx

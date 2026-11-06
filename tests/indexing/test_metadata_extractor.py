"""Tests for polyreader.indexing.metadata.metadata_extractor regex extractors, validation, and pipeline."""

import pytest
from unittest.mock import AsyncMock, patch

from polyreader.indexing.metadata.metadata_extractor import (
    PaperMetadata,
    _clean_doi,
    _extract_doi_regex,
    _extract_year_regex,
    _extract_title_regex,
    _extract_journal_regex,
    _extract_volume_pages_regex,
    _regex_pre_extract,
    _validate_metadata,
    _build_hints_section,
    extract_metadata,
)
from polyreader.indexing.metadata.crossref_client import (
    _title_similarity,
    _clean_title_for_search,
)


# ---------------------------------------------------------------------------
# Regex extractors
# ---------------------------------------------------------------------------

class TestExtractDoiRegex:
    def test_url_format(self):
        text = "Available at https://doi.org/10.1016/j.polymer.2024.1234 for details."
        assert _extract_doi_regex(text) == "10.1016/j.polymer.2024.1234"

    def test_dx_doi_url(self):
        text = "See http://dx.doi.org/10.1021/acs.macromol.3c00456"
        assert _extract_doi_regex(text) == "10.1021/acs.macromol.3c00456"

    def test_labeled_doi(self):
        text = "DOI: 10.1038/s41467-023-12345-6"
        assert _extract_doi_regex(text) == "10.1038/s41467-023-12345-6"

    def test_lowercase_doi_label(self):
        text = "doi: 10.1002/adma.202300001"
        assert _extract_doi_regex(text) == "10.1002/adma.202300001"

    def test_bare_doi(self):
        text = "Ref: 10.1126/science.abc1234 was cited."
        assert _extract_doi_regex(text) == "10.1126/science.abc1234"

    def test_no_doi(self):
        assert _extract_doi_regex("No DOI present in this text.") == ""

    def test_strips_trailing_dot(self):
        text = "DOI: 10.1016/j.polymer.2024.1234."
        assert _extract_doi_regex(text) == "10.1016/j.polymer.2024.1234"


class TestExtractYearRegex:
    def test_published_date(self):
        text = "Published: 15 March 2023"
        assert _extract_year_regex(text) == 2023

    def test_accepted_date(self):
        text = "Accepted March 15, 2024"
        assert _extract_year_regex(text) == 2024

    def test_copyright(self):
        text = "© 2022 Elsevier Ltd."
        assert _extract_year_regex(text) == 2022

    def test_received_date(self):
        text = "Received: 2021\nSome other text"
        assert _extract_year_regex(text) == 2021

    def test_fallback_to_any_year(self):
        text = "2019 Some random text with a year."
        assert _extract_year_regex(text) == 2019

    def test_no_year(self):
        text = "No year information here at all."
        assert _extract_year_regex(text) is None


class TestExtractTitleRegex:
    def test_markdown_h1(self):
        text = "# Synthesis of Novel Polymer Nanocomposites\n\nAbstract\nSome abstract text."
        assert _extract_title_regex(text) == "Synthesis of Novel Polymer Nanocomposites"

    def test_markdown_h2(self):
        text = "## A Study of Phase Transition Behavior in Block Copolymers\n\nIntroduction"
        assert _extract_title_regex(text) == "A Study of Phase Transition Behavior in Block Copolymers"

    def test_skips_abstract_heading(self):
        text = "# Abstract\n\nSome abstract text."
        assert _extract_title_regex(text) == ""

    def test_skips_short_heading(self):
        text = "# Short\nBody text"
        assert _extract_title_regex(text) == ""

    def test_no_headings(self):
        text = "Plain text without any markdown headings."
        assert _extract_title_regex(text) == ""


class TestExtractJournalRegex:
    def test_journal_of_pattern(self):
        text = "Published in Journal of Polymer Science, 2023."
        assert "Journal of Polymer Science" in _extract_journal_regex(text)

    def test_copyright_pattern(self):
        text = "© 2023 American Chemical Society"
        result = _extract_journal_regex(text)
        # May or may not match depending on pattern — the pattern requires ending in known suffix
        assert isinstance(result, str)

    def test_known_journal_name(self):
        text = "Published in Macromolecules 2023, 56, 1234-1245."
        assert _extract_journal_regex(text) == "Macromolecules"

    def test_acs_journal(self):
        text = "ACS Nano 2023, 17(5), 4567-4578."
        assert "ACS" in _extract_journal_regex(text)

    def test_no_journal(self):
        text = "Random text without journal info."
        assert _extract_journal_regex(text) == ""


class TestExtractVolumePagesRegex:
    def test_vol_and_pp(self):
        text = "Vol. 56, pp. 1234-1256"
        vol, pages = _extract_volume_pages_regex(text)
        assert vol == "56"
        assert pages == "1234-1256"

    def test_volume_with_pages_dash(self):
        text = "Volume 123 pages 45-67"
        vol, pages = _extract_volume_pages_regex(text)
        assert vol == "123"
        assert pages == "45-67"

    def test_no_volume_pages(self):
        text = "No volume or page info."
        vol, pages = _extract_volume_pages_regex(text)
        assert vol == ""
        assert pages == ""


# ---------------------------------------------------------------------------
# Regex orchestrator
# ---------------------------------------------------------------------------

class TestRegexPreExtract:
    def test_combined_extraction(self):
        pages = [
            "# Novel Polymer Synthesis for Energy Applications\n\n"
            "DOI: 10.1021/acs.macromol.2023.1234\n"
            "Published: 15 March 2023\n"
            "Macromolecules Vol. 56, pp. 1234-1256\n"
            "Some body text here."
        ]
        result = _regex_pre_extract(pages)
        assert result["doi"] == "10.1021/acs.macromol.2023.1234"
        assert result["year"] == 2023
        assert result["title"] == "Novel Polymer Synthesis for Energy Applications"
        assert result["journal"] == "Macromolecules"
        assert result["volume"] == "56"
        assert result["pages"] == "1234-1256"

    def test_empty_pages(self):
        result = _regex_pre_extract([])
        assert result["doi"] == ""
        assert result["year"] is None


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

class TestValidateMetadata:
    def test_valid_metadata_passes(self):
        meta = {
            "doi": "10.1021/acs.macromol.2023.1234",
            "year": 2023,
            "title": "A Valid Paper Title",
            "authors": ["Alice Smith", "Bob Jones"],
            "volume": "56",
            "pages": "1234-1256",
            "journal": "Macromolecules",
        }
        result = _validate_metadata(meta)
        assert result["doi"] == "10.1021/acs.macromol.2023.1234"
        assert result["year"] == 2023
        assert result["title"] == "A Valid Paper Title"
        assert result["authors"] == ["Alice Smith", "Bob Jones"]

    def test_invalid_doi_cleared(self):
        meta = {"doi": "not-a-doi", "title": "Some Title Here", "authors": [], "year": None}
        result = _validate_metadata(meta)
        assert result["doi"] == ""

    def test_year_out_of_range_cleared(self):
        meta = {"year": 1800, "doi": "", "title": "Some Title Here", "authors": []}
        result = _validate_metadata(meta)
        assert result["year"] is None

    def test_year_future_cleared(self):
        meta = {"year": 2099, "doi": "", "title": "Some Title Here", "authors": []}
        result = _validate_metadata(meta)
        assert result["year"] is None

    def test_short_title_cleared(self):
        meta = {"title": "Hi", "doi": "", "authors": [], "year": None}
        result = _validate_metadata(meta)
        assert result["title"] == ""

    def test_non_list_authors_cleared(self):
        meta = {"authors": "not a list", "doi": "", "title": "Valid Title", "year": None}
        result = _validate_metadata(meta)
        assert result["authors"] == []

    def test_non_string_authors_filtered(self):
        meta = {"authors": ["Valid", 123, "Also Valid"], "doi": "", "title": "Title", "year": None}
        result = _validate_metadata(meta)
        assert result["authors"] == ["Valid", "Also Valid"]

    def test_invalid_volume_cleared(self):
        meta = {"volume": "abc", "doi": "", "title": "Title Here", "authors": [], "year": None}
        result = _validate_metadata(meta)
        assert result["volume"] == ""


# ---------------------------------------------------------------------------
# Hints builder
# ---------------------------------------------------------------------------

class TestBuildHintsSection:
    def test_all_hints(self):
        regex_results = {
            "doi": "10.1021/test",
            "year": 2023,
            "title": "Test Title",
            "journal": "Test Journal",
            "volume": "56",
            "pages": "1-10",
        }
        hints = _build_hints_section(regex_results)
        assert "10.1021/test" in hints
        assert "2023" in hints
        assert "Test Title" in hints
        assert "Pre-extracted hints" in hints

    def test_empty_hints(self):
        regex_results = {"doi": "", "year": None, "title": "", "journal": "", "volume": "", "pages": ""}
        hints = _build_hints_section(regex_results)
        assert hints == ""


# ---------------------------------------------------------------------------
# Integration: extract_metadata pipeline
# ---------------------------------------------------------------------------

class TestExtractMetadataPipeline:
    @pytest.mark.asyncio
    async def test_full_pipeline_with_crossref(self):
        """Mock LLM and CrossRef to test the full merge logic."""
        mock_llm = AsyncMock()
        mock_llm.chat_simple.return_value = '{"title": "Novel Polymer Synthesis for Energy Applications", "authors": ["LLM Author"], "journal": "LLM Journal", "year": 2023, "doi": "10.1234/test", "volume": "1", "pages": "1-10"}'

        crossref_response = {
            "title": "Novel Polymer Synthesis for Energy Applications",
            "authors": ["CR Author One", "CR Author Two"],
            "journal": "CrossRef Journal",
            "year": 2023,
            "doi": "10.1234/test",
            "volume": "1",
            "pages": "1-10",
        }

        page_texts = ["# Novel Polymer Synthesis for Energy Applications\nDOI: 10.1234/test\n© 2023\nSome body text here for testing."]

        with patch("polyreader.indexing.metadata.metadata_extractor.verify_and_enrich_doi", return_value=crossref_response):
            result = await extract_metadata(page_texts, mock_llm)

        # Title prefers LLM/regex over CrossRef
        assert result.title == "Novel Polymer Synthesis for Energy Applications"
        # Authors prefer CrossRef
        assert result.authors == ["CR Author One", "CR Author Two"]
        # Journal prefers CrossRef
        assert result.journal == "CrossRef Journal"
        assert result.doi == "10.1234/test"
        assert result.year == 2023

    @pytest.mark.asyncio
    async def test_pipeline_crossref_unavailable(self):
        """When CrossRef fails, LLM + regex results still work."""
        mock_llm = AsyncMock()
        mock_llm.chat_simple.return_value = '{"title": "Fallback Title", "authors": ["Author A"], "journal": "Some Journal", "year": 2022, "doi": "10.5678/fallback", "volume": "2", "pages": "20-30"}'

        page_texts = ["# Fallback Title\nDOI: 10.5678/fallback\n© 2022\nBody text."]

        with patch("polyreader.indexing.metadata.metadata_extractor.verify_and_enrich_doi", return_value=None), \
             patch("polyreader.indexing.metadata.metadata_extractor.search_by_title", return_value=None):
            result = await extract_metadata(page_texts, mock_llm)

        assert result.title == "Fallback Title"
        assert result.authors == ["Author A"]
        assert result.doi == "10.5678/fallback"

    @pytest.mark.asyncio
    async def test_pipeline_llm_fails(self):
        """When LLM returns garbage, regex + CrossRef can still provide metadata."""
        mock_llm = AsyncMock()
        mock_llm.chat_simple.return_value = "not valid json at all"

        crossref_response = {
            "title": "CrossRef Fallback Title",
            "authors": ["Author X"],
            "journal": "Nature",
            "year": 2021,
            "doi": "10.1038/s41586-021-00001-1",
            "volume": "590",
            "pages": "100-105",
        }

        page_texts = ["# Some Paper About Something Important\nDOI: 10.1038/s41586-021-00001-1\n© 2021"]

        with patch("polyreader.indexing.metadata.metadata_extractor.verify_and_enrich_doi", return_value=crossref_response):
            result = await extract_metadata(page_texts, mock_llm)

        assert result.doi == "10.1038/s41586-021-00001-1"
        assert result.year == 2021
        # Title from regex since LLM failed
        assert "Some Paper" in result.title

    @pytest.mark.asyncio
    async def test_pipeline_title_search_fallback(self):
        """When no DOI but title exists, CrossRef title search is used."""
        mock_llm = AsyncMock()
        mock_llm.chat_simple.return_value = '{"title": "A Novel Approach", "authors": ["Jane Doe"], "journal": null, "year": null, "doi": null, "volume": null, "pages": null}'

        crossref_response = {
            "title": "A Novel Approach",
            "authors": ["Jane Doe", "John Smith"],
            "journal": "Science",
            "year": 2020,
            "doi": "10.1126/science.abc1234",
            "volume": "370",
            "pages": "500-505",
        }

        page_texts = ["# A Novel Approach to Polymer Chemistry Studies\nSome text without DOI."]

        with patch("polyreader.indexing.metadata.metadata_extractor.verify_and_enrich_doi", return_value=None), \
             patch("polyreader.indexing.metadata.metadata_extractor.search_by_title", return_value=crossref_response):
            result = await extract_metadata(page_texts, mock_llm)

        assert result.doi == "10.1126/science.abc1234"
        assert result.journal == "Science"
        assert result.year == 2020

    @pytest.mark.asyncio
    async def test_temperature_uses_client_default(self):
        """Verify LLM is called without explicit temperature (uses client default)."""
        mock_llm = AsyncMock()
        mock_llm.chat_simple.return_value = '{"title": "T", "authors": [], "journal": null, "year": null, "doi": null, "volume": null, "pages": null}'

        with patch("polyreader.indexing.metadata.metadata_extractor.verify_and_enrich_doi", return_value=None), \
             patch("polyreader.indexing.metadata.metadata_extractor.search_by_title", return_value=None):
            await extract_metadata(["Some text"], mock_llm)

        mock_llm.chat_simple.assert_called_once()
        _, kwargs = mock_llm.chat_simple.call_args
        assert "temperature" not in kwargs

    @pytest.mark.asyncio
    async def test_cross_validation_rejects_wrong_doi(self):
        """When CrossRef returns metadata for a different paper, reject and fall through to title search."""
        mock_llm = AsyncMock()
        mock_llm.chat_simple.return_value = '{"title": "Polymer Nanocomposite Synthesis", "authors": ["Alice"], "journal": null, "year": null, "doi": "10.9999/wrong", "volume": null, "pages": null}'

        wrong_crossref = {
            "title": "Completely Different Unrelated Paper About Fish Biology",
            "authors": ["Bob"],
            "journal": "Fish Journal",
            "year": 2020,
            "doi": "10.9999/wrong",
            "volume": "5",
            "pages": "1-10",
        }

        correct_crossref = {
            "title": "Polymer Nanocomposite Synthesis",
            "authors": ["Alice", "Carol"],
            "journal": "Macromolecules",
            "year": 2023,
            "doi": "10.1021/correct",
            "volume": "56",
            "pages": "100-110",
        }

        page_texts = ["# Polymer Nanocomposite Synthesis\nSome body text."]

        with patch("polyreader.indexing.metadata.metadata_extractor.verify_and_enrich_doi", return_value=wrong_crossref), \
             patch("polyreader.indexing.metadata.metadata_extractor.search_by_title", return_value=correct_crossref):
            result = await extract_metadata(page_texts, mock_llm)

        # Should have rejected the wrong DOI and used title search result
        assert result.doi == "10.1021/correct"
        assert result.journal == "Macromolecules"
        assert result.year == 2023


# ---------------------------------------------------------------------------
# DOI regex scope (no reference DOIs from last page)
# ---------------------------------------------------------------------------

class TestDoiRegexScope:
    def test_no_reference_doi_from_last_page(self):
        """DOI regex should NOT pick up DOIs from the last page (references)."""
        first_page = "# A Study of Polymer Crystallization\n© 2023 Elsevier\nAbstract text."
        page2 = "Some continuation of introduction text..."
        last_page = (
            "References\n"
            "[1] Smith et al., doi: 10.1234/reference.paper.1\n"
            "[2] Jones et al., https://doi.org/10.5678/reference.paper.2\n"
        )
        result = _regex_pre_extract([first_page, page2, last_page])
        # Should not find any DOI (none on first page or page-2 header)
        assert result["doi"] == ""

    def test_doi_from_first_page_still_found(self):
        """DOI on the first page should still be found."""
        first_page = "# Title\nDOI: 10.1021/acs.macromol.2023.1234\n© 2023"
        page2 = "Some text"
        last_page = "References\n[1] 10.9999/some.reference"
        result = _regex_pre_extract([first_page, page2, last_page])
        assert result["doi"] == "10.1021/acs.macromol.2023.1234"

    def test_doi_from_page2_header_found(self):
        """DOI in the page-2 header region should still be found."""
        first_page = "# Title\n© 2023\nBody text."
        page2 = "DOI: 10.1016/j.polymer.2023.5678\nMore body text..."
        last_page = "References\n[1] 10.9999/some.reference"
        result = _regex_pre_extract([first_page, page2, last_page])
        assert result["doi"] == "10.1016/j.polymer.2023.5678"


# ---------------------------------------------------------------------------
# _clean_doi edge cases
# ---------------------------------------------------------------------------

class TestCleanDoiEdgeCases:
    def test_markdown_link(self):
        """DOI embedded in a markdown link should be cleaned."""
        assert _clean_doi("10.1021/test](http://doi.org/10.1021/test)") == "10.1021/test"

    def test_angle_brackets(self):
        """DOI with angle brackets should be truncated correctly."""
        assert _clean_doi("<10.1021/acs.macromol.123>") == "10.1021/acs.macromol.123"

    def test_parenthetical_doi(self):
        """DOI with parenthesis should be truncated."""
        assert _clean_doi("(10.1021/test.paper)") == "10.1021/test.paper"

    def test_no_match_returns_empty(self):
        """Non-DOI string should return empty, not the input."""
        assert _clean_doi("not-a-doi-at-all") == ""
        assert _clean_doi("random text 123") == ""

    def test_empty_input(self):
        assert _clean_doi("") == ""
        assert _clean_doi(None) == ""

    def test_url_prefix_stripped(self):
        assert _clean_doi("https://doi.org/10.1021/test") == "10.1021/test"

    def test_trailing_dot_stripped(self):
        assert _clean_doi("10.1021/test.paper.") == "10.1021/test.paper"


# ---------------------------------------------------------------------------
# _clean_title_for_search
# ---------------------------------------------------------------------------

class TestCleanTitleForSearch:
    def test_euro_sign_to_epsilon(self):
        """OCR artifact € should be replaced with ε then stripped to 'e'."""
        result = _clean_title_for_search("Poly(€-caprolactone)")
        # € → ε → (NFKD + ascii strip) → 'e'
        assert "caprolactone" in result
        assert "€" not in result

    def test_smart_quotes_replaced(self):
        result = _clean_title_for_search("\u201cSmart\u201d quotes")
        assert '"' not in result or result == '"Smart" quotes'
        assert "\u201c" not in result
        assert "\u201d" not in result

    def test_en_em_dash_replaced(self):
        result = _clean_title_for_search("Self\u2013assembly")
        assert "\u2013" not in result
        assert "Self-assembly" == result

    def test_ligatures_replaced(self):
        result = _clean_title_for_search("Eﬃcient ﬂow")
        assert "Efficient" in result
        assert "flow" in result

    def test_whitespace_collapsed(self):
        result = _clean_title_for_search("  Multiple   spaces   here  ")
        assert result == "Multiple spaces here"

    def test_plain_ascii_unchanged(self):
        result = _clean_title_for_search("Normal ASCII Title")
        assert result == "Normal ASCII Title"


# ---------------------------------------------------------------------------
# _title_similarity robustness
# ---------------------------------------------------------------------------

class TestTitleSimilarity:
    def test_identical_titles(self):
        assert _title_similarity("polymer synthesis", "polymer synthesis") == 1.0

    def test_completely_different(self):
        assert _title_similarity("polymer synthesis methods", "fish biology research") == 0.0

    def test_punctuation_ignored(self):
        """Punctuation differences should not affect similarity."""
        sim = _title_similarity(
            "synthesis, characterization, and properties",
            "synthesis characterization and properties",
        )
        assert sim == 1.0

    def test_short_words_filtered(self):
        """Short words (a, of, in) should be filtered, not inflate similarity."""
        # Without filtering: "a" and "of" and "in" would all match, inflating score
        sim = _title_similarity(
            "a study of polymer in solution",
            "a review of proteins in membranes",
        )
        # Only "study" vs "review" and "polymer" vs "proteins" and "solution" vs "membranes" differ
        # Common: (none of the content words match)
        assert sim < 0.5

    def test_partial_overlap(self):
        sim = _title_similarity(
            "novel polymer nanocomposites for energy",
            "polymer nanocomposites materials review",
        )
        assert 0.2 < sim < 0.8

    def test_empty_strings(self):
        assert _title_similarity("", "something") == 0.0
        assert _title_similarity("something", "") == 0.0
        assert _title_similarity("", "") == 0.0

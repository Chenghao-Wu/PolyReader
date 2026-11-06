"""Extract DOI, authors, journal, year from paper header/footer.

Three-phase pipeline: Regex pre-extraction → LLM with hints → CrossRef verification.
"""

import logging
import re
from dataclasses import dataclass, field

from polyreader.llm.client import LLMClient
from polyreader.indexing.prompts import EXTRACT_METADATA_PROMPT
from polyreader.utils.json_utils import extract_json
from .crossref_client import verify_and_enrich_doi, search_by_title, _title_similarity

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------

@dataclass
class PaperMetadata:
    """Bibliographic metadata for a paper."""
    title: str = ""
    authors: list[str] = field(default_factory=list)
    journal: str = ""
    year: int = None
    doi: str = ""
    volume: str = ""
    pages: str = ""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clean_doi(doi: str) -> str:
    """Strip markdown link syntax, URL prefixes, and junk from DOI strings."""
    if not doi:
        return ""
    doi_match = re.search(r'(10\.\d{4,}/[^\s,;"\'\]\)(<>]+)', doi)
    if doi_match:
        return doi_match.group(1).rstrip('.')
    return ""


def _build_text(page_texts: list[str]) -> str:
    """Combine first page, page-2 header, and last page into extraction text."""
    text = page_texts[0] if page_texts else ""
    if len(page_texts) > 2:
        text += "\n\n--- Page 2 header ---\n" + page_texts[1][:1000]
    if len(page_texts) > 1:
        text += "\n\n--- Last page ---\n" + page_texts[-1]
    return text


# ---------------------------------------------------------------------------
# Phase 1: Regex pre-extraction
# ---------------------------------------------------------------------------

def _extract_doi_regex(text: str) -> str:
    """Extract DOI from text using multiple regex patterns."""
    patterns = [
        # URL format: https://doi.org/10.xxxx/...
        r'(?:https?://)?(?:dx\.)?doi\.org/(10\.\d{4,}/[^\s,;"\'\]\)]+)',
        # Labeled: DOI: 10.xxxx/...  or  doi: 10.xxxx/...
        r'[Dd][Oo][Ii]\s*[:=]\s*(10\.\d{4,}/[^\s,;"\'\]\)]+)',
        # Bare DOI in text
        r'(10\.\d{4,}/[^\s,;"\'\]\)]+)',
    ]
    for pattern in patterns:
        m = re.search(pattern, text)
        if m:
            return m.group(1).rstrip('.')
    return ""


def _extract_year_regex(text: str) -> int | None:
    """Extract publication year from contextual patterns, then fallback."""
    # Contextual: near publication-related keywords
    contextual_patterns = [
        r'(?:Published|Accepted|Received|Available online|Copyright|©)\s*:?\s*\d{1,2}\s+\w+\s+((?:19|20)\d{2})',
        r'(?:Published|Accepted|Received|Available online|Copyright|©)\s*:?\s*\w+\s+\d{1,2},?\s+((?:19|20)\d{2})',
        r'(?:Published|Accepted|Received|Available online|Copyright|©)\s*:?\s*((?:19|20)\d{2})',
        r'©\s*((?:19|20)\d{2})',
    ]
    for pattern in contextual_patterns:
        m = re.search(pattern, text)
        if m:
            return int(m.group(1))

    # Fallback: any 4-digit year in first 2000 chars
    first_chunk = text[:2000]
    m = re.search(r'\b((?:19|20)\d{2})\b', first_chunk)
    if m:
        return int(m.group(1))
    return None


def _extract_title_regex(first_page: str) -> str:
    """Extract title from Marker markdown headings on first page."""
    # Marker outputs titles as # Heading
    lines = first_page.split('\n')
    for line in lines:
        stripped = line.strip()
        # Match markdown headings (# or ##)
        m = re.match(r'^#{1,2}\s+(.+)', stripped)
        if m:
            title_candidate = m.group(1).strip()
            # Skip very short or obviously non-title headings
            if len(title_candidate) > 10 and not re.match(r'^(Abstract|Introduction|References|Acknowledgment)', title_candidate, re.IGNORECASE):
                return title_candidate
    return ""


def _extract_journal_regex(text: str) -> str:
    """Extract journal name from common patterns."""
    journal_patterns = [
        r'(?:Published in|Journal|©\s*\d{4})\s*[:\-]?\s*([A-Z][A-Za-z\s&]+(?:Journal|Letters|Reviews|Communications|Chemistry|Science|Materials|Polymers|Macromolecules))',
        r'(Journal of [A-Za-z\s&]+)',
        r'((?:ACS|RSC|Angewandte|Nature|Physical Review|Chemical Reviews|Advanced)[A-Za-z\s&]*)',
        # Standalone journal names — must not be preceded by a letter (avoids mid-title matches)
        # and must be followed by volume/year/punctuation context, not a title-like continuation
        r'(?<![A-Za-z])(Macromolecules|Langmuir|Biomacromolecules|Soft Matter|Chemistry of Materials)\b',
        r'(?<![A-Za-z])(Polymer)(?:\s+(?:Chemistry|Physics|Testing|Degradation|International|Journal))',
    ]
    for pattern in journal_patterns:
        m = re.search(pattern, text)
        if m:
            return m.group(1).strip().rstrip('.')
    return ""


def _extract_volume_pages_regex(text: str) -> tuple[str, str]:
    """Extract volume and page numbers from text."""
    volume = ""
    pages = ""

    # Volume patterns
    vol_patterns = [
        r'[Vv]ol(?:ume)?\.?\s*(\d+)',
        r',\s*(\d+)\s*[,(]\s*\d+',  # journal, vol(issue) or journal, vol, pages
    ]
    for pattern in vol_patterns:
        m = re.search(pattern, text)
        if m:
            volume = m.group(1)
            break

    # Pages patterns
    page_patterns = [
        r'pp\.?\s*(\d+[\-–]\d+)',
        r'[Pp]ages?\s*(\d+[\-–]\d+)',
        r',\s*(\d{4,}[\-–]\d{4,})',  # long page numbers like 12345-12356
    ]
    for pattern in page_patterns:
        m = re.search(pattern, text)
        if m:
            pages = m.group(1)
            break

    return volume, pages


def _regex_pre_extract(page_texts: list[str]) -> dict:
    """Orchestrate all regex extractors. Returns dict of high-confidence seeds."""
    text = _build_text(page_texts)
    first_page = page_texts[0] if page_texts else ""

    # Restrict DOI search to first page + page-2 header (avoid reference DOIs on last page)
    doi_text = first_page
    if len(page_texts) > 2:
        doi_text += "\n" + page_texts[1][:1000]
    doi = _clean_doi(_extract_doi_regex(doi_text))
    year = _extract_year_regex(text)
    title = _extract_title_regex(first_page)
    journal = _extract_journal_regex(text)
    volume, pages = _extract_volume_pages_regex(text)

    return {
        "doi": doi,
        "year": year,
        "title": title,
        "journal": journal,
        "volume": volume,
        "pages": pages,
    }


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _validate_metadata(meta: dict) -> dict:
    """Validate and sanitize metadata fields. Returns cleaned dict."""
    result = dict(meta)

    # Validate DOI format
    doi = result.get("doi", "") or ""
    if doi and not re.match(r'^10\.\d{4,}/', doi):
        logger.debug("Invalid DOI format, clearing: %s", doi)
        result["doi"] = ""

    # Validate year range
    year = result.get("year")
    if year is not None:
        try:
            year = int(year)
            if not (1900 <= year <= 2030):
                logger.debug("Year out of range, clearing: %s", year)
                result["year"] = None
            else:
                result["year"] = year
        except (ValueError, TypeError):
            result["year"] = None

    # Validate title length
    title = result.get("title", "") or ""
    if len(title) < 5:
        result["title"] = ""

    # Validate authors is a list of strings
    authors = result.get("authors", [])
    if not isinstance(authors, list):
        result["authors"] = []
    else:
        result["authors"] = [a for a in authors if isinstance(a, str) and a.strip()]

    # Clean volume
    volume = result.get("volume", "") or ""
    if volume and not re.match(r'^\d+', str(volume)):
        result["volume"] = ""

    return result


# ---------------------------------------------------------------------------
# Hints builder
# ---------------------------------------------------------------------------

def _build_hints_section(regex_results: dict) -> str:
    """Build the hints section for the LLM prompt from regex results."""
    hints = []
    if regex_results.get("doi"):
        hints.append(f"- A regex pre-scan found this possible DOI: {regex_results['doi']}. Verify it.")
    if regex_results.get("year"):
        hints.append(f"- A regex pre-scan found this possible year: {regex_results['year']}. Verify it.")
    if regex_results.get("title"):
        hints.append(f"- A regex pre-scan found this possible title: {regex_results['title']}. Verify it.")
    if regex_results.get("journal"):
        hints.append(f"- A regex pre-scan found this possible journal: {regex_results['journal']}. Verify it.")
    if regex_results.get("volume"):
        hints.append(f"- A regex pre-scan found this possible volume: {regex_results['volume']}. Verify it.")
    if regex_results.get("pages"):
        hints.append(f"- A regex pre-scan found this possible pages: {regex_results['pages']}. Verify it.")

    if hints:
        return "\nPre-extracted hints (verify these against the text):\n" + "\n".join(hints) + "\n"
    return ""


# ---------------------------------------------------------------------------
# Phase 2 + 3: LLM + CrossRef orchestrator
# ---------------------------------------------------------------------------

async def extract_metadata(
    page_texts: list[str],
    llm: LLMClient,
) -> PaperMetadata:
    """Extract metadata via three-phase pipeline.

    Phase 1: Regex pre-extraction for high-confidence seeds.
    Phase 2: LLM extraction with regex hints.
    Phase 3: CrossRef verification/enrichment.
    Merge: CrossRef > LLM > Regex. Validate final result.
    """
    # Phase 1 — Regex
    regex_results = _regex_pre_extract(page_texts)
    logger.info("Regex pre-extraction: %s", regex_results)

    # Phase 2 — LLM with hints
    text = _build_text(page_texts)
    hints_section = _build_hints_section(regex_results)
    prompt = EXTRACT_METADATA_PROMPT.format(text=text[:6000], hints_section=hints_section)
    response = await llm.chat_simple(prompt)
    parsed = extract_json(response)

    if not isinstance(parsed, dict):
        logger.warning("Failed to parse metadata from LLM")
        parsed = {}

    # Clean LLM DOI
    llm_doi = _clean_doi(parsed.get("doi", "") or "")
    llm_results = {
        "title": parsed.get("title", "") or "",
        "authors": parsed.get("authors", []) if isinstance(parsed.get("authors"), list) else [],
        "journal": parsed.get("journal", "") or "",
        "year": parsed.get("year"),
        "doi": llm_doi,
        "volume": parsed.get("volume", "") or "",
        "pages": parsed.get("pages", "") or "",
    }

    # Phase 3 — CrossRef verification/enrichment
    # Pick best DOI: prefer LLM (more contextual), fall back to regex
    doi = llm_results["doi"] or regex_results.get("doi", "")
    title = llm_results["title"] or regex_results.get("title", "")

    crossref_data = None
    if doi:
        crossref_data = await verify_and_enrich_doi(doi)
        if crossref_data:
            # Cross-validate: reject if CrossRef title doesn't match our extracted title
            cr_title = crossref_data.get("title", "")
            if title and cr_title:
                sim = _title_similarity(title.lower(), cr_title.lower())
                if sim < 0.4:
                    logger.info(
                        "CrossRef DOI %s rejected: title mismatch (sim=%.2f) '%s' vs '%s'",
                        doi, sim, title[:60], cr_title[:60],
                    )
                    crossref_data = None
                else:
                    logger.info("CrossRef verified DOI: %s (title sim=%.2f)", doi, sim)
            else:
                logger.info("CrossRef verified DOI: %s (no title to cross-validate)", doi)
    if not crossref_data and title:
        crossref_data = await search_by_title(title)
        if crossref_data:
            logger.info("CrossRef found match by title search")

    # Merge: CrossRef > LLM > Regex
    # CrossRef is authoritative for structured fields; title prefers LLM/regex
    merged = {}

    # Title: prefer LLM/regex (CrossRef can have HTML entities)
    merged["title"] = llm_results["title"] or regex_results.get("title", "")
    if not merged["title"] and crossref_data:
        merged["title"] = crossref_data.get("title", "")

    # Authors: CrossRef > LLM (CrossRef has structured author data)
    if crossref_data and crossref_data.get("authors"):
        merged["authors"] = crossref_data["authors"]
    else:
        merged["authors"] = llm_results["authors"]

    # Journal: CrossRef > LLM > Regex
    merged["journal"] = (
        (crossref_data or {}).get("journal", "")
        or llm_results["journal"]
        or regex_results.get("journal", "")
    )

    # Year: CrossRef > LLM > Regex
    merged["year"] = (
        (crossref_data or {}).get("year")
        or llm_results["year"]
        or regex_results.get("year")
    )

    # DOI: CrossRef > LLM > Regex
    merged["doi"] = _clean_doi(
        (crossref_data or {}).get("doi", "")
        or llm_results["doi"]
        or regex_results.get("doi", "")
    )

    # Volume: CrossRef > LLM > Regex
    merged["volume"] = (
        (crossref_data or {}).get("volume", "")
        or llm_results["volume"]
        or regex_results.get("volume", "")
    )

    # Pages: CrossRef > LLM > Regex
    merged["pages"] = (
        (crossref_data or {}).get("pages", "")
        or llm_results["pages"]
        or regex_results.get("pages", "")
    )

    # Validate
    validated = _validate_metadata(merged)

    return PaperMetadata(
        title=validated.get("title", ""),
        authors=validated.get("authors", []),
        journal=validated.get("journal", ""),
        year=validated.get("year"),
        doi=validated.get("doi", ""),
        volume=validated.get("volume", ""),
        pages=validated.get("pages", ""),
    )

"""Async CrossRef API client for metadata verification and enrichment."""

import logging
import re
import unicodedata
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

_CROSSREF_BASE = "https://api.crossref.org"
_TIMEOUT = 10.0
_HEADERS = {
    "User-Agent": "polyreader/0.2 (https://github.com/Chenghao-Wu/polyextract)",
    "Accept": "application/json",
}


def _parse_crossref_response(data: dict) -> dict:
    """Normalize CrossRef work item to a flat metadata dict."""
    msg = data.get("message", data)

    # Title
    titles = msg.get("title", [])
    title = titles[0] if titles else ""

    # Authors
    authors = []
    for a in msg.get("author", []):
        given = a.get("given", "")
        family = a.get("family", "")
        if given and family:
            authors.append(f"{given} {family}")
        elif family:
            authors.append(family)

    # Journal
    container = msg.get("container-title", [])
    journal = container[0] if container else ""

    # Year — prefer published-print, then published-online, then issued
    year = None
    for date_key in ("published-print", "published-online", "issued"):
        date_parts = msg.get(date_key, {}).get("date-parts", [[]])
        if date_parts and date_parts[0] and date_parts[0][0]:
            year = int(date_parts[0][0])
            break

    # DOI
    doi = msg.get("DOI", "")

    # Volume and pages
    volume = msg.get("volume", "") or ""
    pages = msg.get("page", "") or ""

    return {
        "title": title,
        "authors": authors,
        "journal": journal,
        "year": year,
        "doi": doi,
        "volume": volume,
        "pages": pages,
    }


async def verify_and_enrich_doi(doi: str) -> Optional[dict]:
    """Look up a DOI on CrossRef and return normalized metadata, or None."""
    if not doi:
        return None
    url = f"{_CROSSREF_BASE}/works/{doi}"
    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT, headers=_HEADERS) as client:
            resp = await client.get(url)
            if resp.status_code != 200:
                logger.info("CrossRef DOI lookup returned %d for %s", resp.status_code, doi)
                return None
            return _parse_crossref_response(resp.json())
    except Exception as e:
        logger.warning("CrossRef DOI lookup failed for %s: %s", doi, e)
        return None


_OCR_REPLACEMENTS = {
    "€": "ε",
    "\u2018": "'",  # left single smart quote
    "\u2019": "'",  # right single smart quote
    "\u201c": '"',  # left double smart quote
    "\u201d": '"',  # right double smart quote
    "\u2013": "-",  # en dash
    "\u2014": "-",  # em dash
    "ﬁ": "fi",
    "ﬂ": "fl",
    "ﬀ": "ff",
    "ﬃ": "ffi",
    "ﬄ": "ffl",
}


def _clean_title_for_search(title: str) -> str:
    """Replace common OCR artifacts and strip non-ASCII for CrossRef search."""
    for bad, good in _OCR_REPLACEMENTS.items():
        title = title.replace(bad, good)
    # Strip remaining non-ASCII
    title = unicodedata.normalize("NFKD", title)
    title = title.encode("ascii", "ignore").decode("ascii")
    # Collapse whitespace
    title = re.sub(r"\s+", " ", title).strip()
    return title


async def search_by_title(title: str) -> Optional[dict]:
    """Search CrossRef by title and return best match, or None."""
    if not title or len(title) < 10:
        return None
    cleaned_title = _clean_title_for_search(title)
    if not cleaned_title or len(cleaned_title) < 10:
        return None
    url = f"{_CROSSREF_BASE}/works"
    params = {"query.title": cleaned_title, "rows": 5, "select": "DOI,title,author,container-title,published-print,published-online,issued,volume,page"}
    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT, headers=_HEADERS) as client:
            resp = await client.get(url, params=params)
            if resp.status_code != 200:
                logger.info("CrossRef title search returned %d", resp.status_code)
                return None
            items = resp.json().get("message", {}).get("items", [])
            if not items:
                logger.info("CrossRef title search returned no items for: %s", cleaned_title)
                return None
            # Pick best match: compare normalized titles
            query_lower = cleaned_title.lower().strip()
            best_item = None
            best_sim = 0.0
            for item in items:
                cr_titles = item.get("title", [])
                for cr_title in cr_titles:
                    sim = _title_similarity(query_lower, cr_title.lower().strip())
                    if sim > 0.8:
                        logger.info("CrossRef title match (sim=%.2f): %s", sim, cr_title)
                        return _parse_crossref_response(item)
                    if sim > best_sim:
                        best_sim = sim
                        best_item = item
            # Only return best result if similarity > 0.5
            if best_item and best_sim > 0.5:
                logger.info("CrossRef best title match (sim=%.2f)", best_sim)
                return _parse_crossref_response(best_item)
            logger.info("CrossRef title search: no good match (best sim=%.2f) for: %s", best_sim, cleaned_title)
            return None
    except Exception as e:
        logger.warning("CrossRef title search failed: %s", e)
        return None


def _title_similarity(a: str, b: str) -> float:
    """Word-overlap Jaccard similarity, ignoring punctuation and short words."""
    # Strip punctuation
    a_clean = re.sub(r"[^\w\s]", "", a)
    b_clean = re.sub(r"[^\w\s]", "", b)
    # Filter words <= 2 chars (articles like "a", "of", "in")
    words_a = {w for w in a_clean.lower().split() if len(w) > 2}
    words_b = {w for w in b_clean.lower().split() if len(w) > 2}
    if not words_a or not words_b:
        return 0.0
    intersection = words_a & words_b
    union = words_a | words_b
    return len(intersection) / len(union)

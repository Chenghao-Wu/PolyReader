"""Provenance factory and merge helpers."""

from __future__ import annotations

from polyreader.schema.polymer import Provenance


def make_provenance(
    *,
    source_type: str = "",
    source_location: str = "",
    page_numbers: list[int] | None = None,
    source_document: str = "",
) -> Provenance:
    """Create a Provenance object with sensible defaults."""
    return Provenance(
        source_type=source_type,
        source_location=source_location,
        page_numbers=page_numbers or [],
        source_document=source_document,
    )


def merge_provenances(provenances: list[Provenance]) -> Provenance:
    """Merge multiple provenance records into one.

    Takes the first record as the base, merges page numbers,
    and combines source locations.
    """
    if not provenances:
        return Provenance()
    if len(provenances) == 1:
        return provenances[0].model_copy()

    base = provenances[0].model_copy()

    # Merge page numbers and source locations from all sources
    all_pages = set()
    locations = []
    documents = []
    for p in provenances:
        all_pages.update(p.page_numbers)
        if p.source_location:
            locations.append(p.source_location)
        if p.source_document:
            documents.append(p.source_document)

    base.page_numbers = sorted(all_pages)
    if len(locations) > 1:
        base.source_location = "; ".join(dict.fromkeys(locations))  # deduplicate, preserve order
    # Merge source_document: join unique values with ";"
    unique_docs = list(dict.fromkeys(documents))
    if len(unique_docs) > 1:
        base.source_document = "; ".join(unique_docs)
    elif unique_docs:
        base.source_document = unique_docs[0]

    return base

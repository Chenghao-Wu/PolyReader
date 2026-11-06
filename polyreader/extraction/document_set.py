"""DocumentSet: wraps multiple PaperIndex objects for multi-document extraction.

Enables processing of main text + supplementary information PDFs in a single
extraction run. The LLM sees all document names/structures and decides what
to read from where via the ``doc`` parameter on tool calls.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Iterator

from polyreader.indexing import PaperIndex
from polyreader.llm import LLMClient

logger = logging.getLogger(__name__)


def _sanitize_label(name: str) -> str:
    """Convert a document name to a short, safe label string."""
    # Take first 40 chars, replace non-alphanumeric with underscore, collapse runs
    label = re.sub(r"[^a-zA-Z0-9]+", "_", name[:40]).strip("_").lower()
    return label or "doc"


class UnknownDocumentError(LookupError):
    """A tool call named a document label that is not in the set."""

    def __init__(self, label: str, available: list[str]):
        self.label = label
        self.available = list(available)
        super().__init__(
            f"Unknown document '{label}'. Available documents: "
            + ", ".join(f'"{a}"' for a in self.available)
        )


class DocumentSet:
    """Wraps multiple PaperIndex objects for multi-document extraction.

    Documents are keyed by short labels derived from each PaperIndex.doc_name.
    Tool dispatch reads the ``doc`` parameter from LLM arguments and routes
    to the correct PaperIndex via :meth:`resolve`.
    """

    def __init__(self, documents: dict[str, PaperIndex]) -> None:
        if not documents:
            raise ValueError("DocumentSet requires at least one document.")
        self._documents: dict[str, PaperIndex] = dict(documents)
        self._labels: list[str] = list(self._documents.keys())
        logger.info(f"  [document_set] Created with {len(self._documents)} document(s): {self._labels}")

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_index(cls, index: PaperIndex) -> DocumentSet:
        """Wrap a single PaperIndex. Label derived from doc_name."""
        label = _sanitize_label(index.doc_name) if index.doc_name else "doc_1"
        return cls({label: index})

    @classmethod
    def from_indices(cls, *indices: PaperIndex) -> DocumentSet:
        """Wrap multiple PaperIndex objects. Labels auto-derived from doc_name."""
        if not indices:
            raise ValueError("At least one PaperIndex is required.")

        documents: dict[str, PaperIndex] = {}
        used_labels: dict[str, int] = {}

        for idx in indices:
            base = _sanitize_label(idx.doc_name) if idx.doc_name else "doc"
            if not base:
                base = "doc"

            # Handle collisions
            if base in used_labels:
                used_labels[base] += 1
                label = f"{base}_{used_labels[base]}"
                logger.warning(f"  [document_set] Label collision for '{base}', using '{label}' instead")
            else:
                used_labels[base] = 1
                label = base

            documents[label] = idx

        return cls(documents)

    @classmethod
    async def from_indices_auto(cls, *indices: PaperIndex, llm: LLMClient) -> DocumentSet:
        """Wrap multiple PaperIndex objects, auto-detecting main vs. supplementary.

        Uses LLM classification to determine which document is the main paper
        and which are supplementary, regardless of argument order.
        """
        from .doc_classifier import sort_documents

        if not indices:
            raise ValueError("At least one PaperIndex is required.")

        main_index, si_list, classifications = await sort_documents(list(indices), llm)

        documents: dict[str, PaperIndex] = {}

        # Main document first (becomes .primary)
        main_label = _sanitize_label(main_index.doc_name) if main_index.doc_name else "main"
        documents[main_label] = main_index

        # Supplementary documents
        for i, si_index in enumerate(si_list):
            if len(si_list) == 1:
                label = "si"
            else:
                label = f"si_{i + 1}"
            documents[label] = si_index

        return cls(documents)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def primary(self) -> PaperIndex:
        """First document (for metadata/DOI access)."""
        return self._documents[self._labels[0]]

    def labels(self) -> list[str]:
        """Return all document labels in insertion order."""
        return list(self._labels)

    def has_multiple_documents(self) -> bool:
        """True when the set contains more than one document."""
        return len(self._documents) > 1

    # ------------------------------------------------------------------
    # Routing
    # ------------------------------------------------------------------

    def _match_label(self, label: str) -> str | None:
        """Find the label for *label*: exact, case-insensitive, or by doc name."""
        if label in self._documents:
            return label
        label_lower = label.strip().lower()
        for key in self._documents:
            if key.lower() == label_lower:
                logger.info(f"  [document_set] Resolved '{label}' via case-insensitive match to '{key}'")
                return key
        # The LLM sometimes passes the file name instead of the label.
        sanitized = _sanitize_label(label)
        for key, index in self._documents.items():
            name = index.doc_name or ""
            if name and (
                name.lower() == label_lower
                or Path(name).stem.lower() == label_lower
                or _sanitize_label(name) == sanitized
            ):
                logger.info(f"  [document_set] Resolved '{label}' via document name to '{key}'")
                return key
        return None

    def resolve(self, label: str | None, strict: bool = False) -> PaperIndex:
        """Route to a PaperIndex by label.

        Falls back to primary when *label* is ``None`` or empty. An unknown
        label also falls back to primary, unless ``strict`` is True, in which
        case :class:`UnknownDocumentError` (listing the available labels) is
        raised instead of silently reading the wrong document.
        """
        if label is not None and not isinstance(label, str):
            label = str(label)
        if not label or not label.strip():
            if len(self._documents) > 1:
                logger.debug("  [document_set] resolve() called with empty label, falling back to primary")
            return self.primary
        key = self._match_label(label)
        if key is not None:
            return self._documents[key]
        if strict:
            raise UnknownDocumentError(label, self._labels)
        logger.warning(f"  [document_set] Unknown label '{label}', falling back to primary")
        return self.primary

    # ------------------------------------------------------------------
    # Structure helpers
    # ------------------------------------------------------------------

    def get_combined_structure(self) -> dict[str, dict]:
        """Return all documents' trees keyed by label.

        Each value contains ``doc_name``, ``pages``, and ``structure``.
        For single-doc sets the format is identical to what the LLM already
        expects (a flat list) but wrapped in a keyed dict.
        """
        result: dict[str, dict] = {}
        for label, index in self._documents.items():
            result[label] = {
                "doc_name": index.doc_name,
                "pages": len(index.page_texts) if index.page_texts else 0,
                "structure": index.get_document_structure(),
            }
        logger.info(
            "  [document_set] Combined structure: "
            + ", ".join(f"'{label}' ({info['pages']}p)" for label, info in result.items())
        )
        return result

    def ensure_pages_loaded(self) -> None:
        """Lazy-load page texts for all documents."""
        logger.info(f"  [document_set] Loading pages for {len(self._documents)} document(s)")
        for index in self._documents.values():
            index._lazy_load_pages()
        logger.info("  [document_set] Pages loaded for all documents")

    # ------------------------------------------------------------------
    # Iteration
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._documents)

    def __iter__(self) -> Iterator[tuple[str, PaperIndex]]:
        for label in self._labels:
            yield label, self._documents[label]

    def __getitem__(self, label: str) -> PaperIndex:
        return self._documents[label]

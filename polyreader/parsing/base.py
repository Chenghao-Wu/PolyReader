"""Common interface for PDF parser backends.

Each backend converts one PDF into per-page Markdown texts plus extracted
images saved under ``images_dir``. Backends never expose their native output
formats to PaperIndex/Polyextract — they return a :class:`BackendOutput` that
``parse_pdf()`` turns into the shared ``ParseResult``.

Image references inside page texts must be written as ``![](<filename>)``
where ``<filename>`` is relative to ``images_dir`` (flat, no sub-directories),
so ``PaperIndex.get_page_image()`` can serve them for every backend.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


class ParserError(RuntimeError):
    """A PDF backend failed to convert a document.

    ``diagnostics`` carries backend-specific details (exit code, stderr tail,
    command line, …) so failures can be investigated without re-running.
    """

    def __init__(self, message: str, diagnostics: dict[str, Any] | None = None):
        super().__init__(message)
        self.diagnostics = diagnostics or {}


class ParserUnavailableError(ParserError):
    """The selected backend is not installed or cannot be found."""


@dataclass
class BackendOutput:
    """What a backend returns: page texts, image metadata, and parser info."""

    page_texts: list[str]
    images: list = field(default_factory=list)  # list[ImageMeta]
    # Name/version/configuration of the backend that produced this output.
    info: dict[str, Any] = field(default_factory=dict)


class PdfParserBackend:
    """Base class for PDF parser backends."""

    name: str = ""

    def version(self) -> str:
        """Return the installed backend version, or "" if unknown."""
        return ""

    def check_available(self) -> None:
        """Raise ParserUnavailableError if the backend cannot run."""

    def convert(self, pdf_path: str, images_dir: str) -> BackendOutput:
        """Convert ``pdf_path`` synchronously (called from an executor)."""
        raise NotImplementedError

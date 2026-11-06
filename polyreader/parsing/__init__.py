"""PDF parsing: Marker (default) or MinerU (optional) → per-page Markdown + images."""

from .registry import AVAILABLE_BACKENDS, DEFAULT_BACKEND, get_backend
from .base import BackendOutput, ParserError, ParserUnavailableError, PdfParserBackend
from .pdf_parser import ImageMeta, ParseResult, parse_pdf

__all__ = [
    "AVAILABLE_BACKENDS",
    "BackendOutput",
    "DEFAULT_BACKEND",
    "ImageMeta",
    "ParseResult",
    "ParserError",
    "ParserUnavailableError",
    "PdfParserBackend",
    "get_backend",
    "parse_pdf",
]

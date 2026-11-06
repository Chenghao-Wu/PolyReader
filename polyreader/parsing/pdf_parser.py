"""PDF parsing: per-page Markdown + images via a selectable backend.

Marker is the default backend; MinerU is an optional alternative (see
``polyreader.parsing``). Every backend returns the same ``ParseResult``.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .registry import DEFAULT_BACKEND, get_backend
# Re-exported for backward compatibility (these used to live here).
from .marker_parser import (  # noqa: F401
    _clean_page_text,
    _convert_with_marker,
    _get_artifact_dict,
    _save_marker_images,
    _split_paginated_text,
)

logger = logging.getLogger(__name__)


@dataclass
class ImageMeta:
    """Metadata for an extracted image."""
    filename: str
    page: int
    index: int
    width: int
    height: int
    x: float
    y: float
    bbox_width: float
    bbox_height: float
    size_bytes: int
    format: str
    colorspace: str


@dataclass
class ParseResult:
    """Result from PDF parsing."""
    markdown: str
    page_texts: list[str]
    images: list[ImageMeta] = field(default_factory=list)
    images_dir: Optional[str] = None
    mode_used: str = "marker"
    pdf_path: str = ""
    # Backend name, version and settings (recorded in the saved index).
    parser_info: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

async def parse_pdf(
    pdf_path: str,
    output_dir: str = None,
    backend: str = DEFAULT_BACKEND,
    **backend_options,
) -> ParseResult:
    """Parse PDF: extract per-page text and images.

    Args:
        pdf_path: Path to PDF file (never modified).
        output_dir: Directory for output files (images).
        backend: ``"marker"`` (default) or ``"mineru"``.
        **backend_options: Backend settings, e.g. ``tier="standard"`` for MinerU.

    Returns:
        ParseResult with markdown, page_texts, images

    Raises:
        FileNotFoundError: the PDF does not exist.
        ParserUnavailableError: the selected backend is not installed.
        ParserError: the backend failed (``.diagnostics`` holds details).
    """
    parser = get_backend(backend, **backend_options)

    pdf = Path(pdf_path).resolve()
    if not pdf.is_file():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    pdf_path = str(pdf)

    parser.check_available()

    if output_dir is None:
        output_dir = str(pdf.parent / f"{pdf.stem}_output")
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    images_dir = str(Path(output_dir) / "images")

    loop = asyncio.get_running_loop()
    out = await loop.run_in_executor(None, parser.convert, pdf_path, images_dir)

    page_texts = out.page_texts
    markdown = "\n\n".join(page_texts)

    return ParseResult(
        markdown=markdown,
        page_texts=page_texts,
        images=out.images,
        images_dir=images_dir,
        mode_used=parser.name,
        pdf_path=pdf_path,
        parser_info=out.info or {"backend": parser.name},
    )

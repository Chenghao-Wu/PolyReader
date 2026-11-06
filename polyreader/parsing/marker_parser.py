"""Marker backend (default): structured Markdown + images via marker-pdf."""

from __future__ import annotations

import logging
import re
import threading
from pathlib import Path

from .base import BackendOutput, ParserUnavailableError, PdfParserBackend

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Marker model singleton (thread-safe lazy init)
# ---------------------------------------------------------------------------

_artifact_dict = None
_artifact_lock = threading.Lock()

# Marker is not safe to run concurrently in one process (two parses in
# parallel threads crash with heap corruption), so conversions are serialized.
_convert_lock = threading.Lock()


def _get_artifact_dict() -> dict:
    global _artifact_dict
    if _artifact_dict is None:
        with _artifact_lock:
            if _artifact_dict is None:
                from marker.models import create_model_dict
                _artifact_dict = create_model_dict()
    return _artifact_dict


# ---------------------------------------------------------------------------
# Marker conversion helpers
# ---------------------------------------------------------------------------

def _clean_page_text(text: str) -> str:
    """Remove noise artifacts from Marker-parsed page text."""
    text = re.sub(r'<span[^>]*></span>', '', text)
    text = re.sub(r'■\s*', '', text)
    return text.strip()


def _split_paginated_text(text: str) -> list[str]:
    """Split marker paginated output into per-page texts.

    Marker's pagination format (when paginate_output=True):
        \\n\\n{PAGE_ID}------------------------------------------------\\n\\n
    where PAGE_ID is 0-indexed and separator is 48 dashes by default.
    """
    pattern = r"\n\n\{(\d+)\}-{20,}\n\n"
    parts = re.split(pattern, text)

    if len(parts) == 1:
        # No pagination markers found — single page
        return [parts[0].strip()]

    # parts looks like: [text_before_first_marker, page_id_0, text_0, page_id_1, text_1, ...]
    # The first element is text before the first page marker (usually empty or whitespace).
    # Then alternating: page_id, page_text
    pages: dict[int, str] = {}
    # Start from index 1 (skip leading text), step by 2
    for i in range(1, len(parts), 2):
        page_id = int(parts[i])
        page_text = parts[i + 1].strip() if i + 1 < len(parts) else ""
        pages[page_id] = page_text

    # Return ordered by page_id
    if not pages:
        return [text.strip()]
    max_page = max(pages.keys())
    return [pages.get(i, "") for i in range(max_page + 1)]


def _save_marker_images(images: dict, output_dir: str) -> list:
    """Save marker PIL images to disk and build ImageMeta list.

    Marker returns {filename: PIL.Image} where filenames follow the pattern:
        _page_{page_id}_{BlockType}_{block_id}.{ext}
    Example: _page_0_Figure_3.png (page_id is 0-indexed)
    """
    from .pdf_parser import ImageMeta

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    filename_pattern = re.compile(r"_page_(\d+)_(\w+)_(\d+)\.(\w+)")
    extracted: list[ImageMeta] = []
    page_counters: dict[int, int] = {}  # per-page image counter

    for filename, pil_image in images.items():
        match = filename_pattern.match(filename)
        if match:
            page_id = int(match.group(1))  # 0-indexed
            ext = match.group(4)
        else:
            # Fallback: try to extract page from filename, default to page 0
            page_id = 0
            ext = Path(filename).suffix.lstrip(".") or "png"

        page_num = page_id + 1  # Convert to 1-indexed

        # Per-page counter
        page_counters[page_num] = page_counters.get(page_num, 0) + 1
        index = page_counters[page_num]

        img_path = out / filename
        pil_image.save(str(img_path))

        size_bytes = img_path.stat().st_size
        width, height = pil_image.size

        extracted.append(ImageMeta(
            filename=filename,
            page=page_num,
            index=index,
            width=width,
            height=height,
            x=0.0,
            y=0.0,
            bbox_width=0.0,
            bbox_height=0.0,
            size_bytes=size_bytes,
            format=ext,
            colorspace="RGB",
        ))

    return extracted


def _convert_with_marker(pdf_path: str, images_dir: str) -> tuple[list[str], list]:
    """Run marker PDF conversion (sync, intended to run in executor).

    Returns (page_texts, image_metas).
    """
    from marker.converters.pdf import PdfConverter
    from marker.output import text_from_rendered

    artifact_dict = _get_artifact_dict()

    config = {
        "paginate_output": True,
        "use_llm": False,
    }

    converter = PdfConverter(
        artifact_dict=artifact_dict,
        config=config,
    )

    rendered = converter(pdf_path)
    text, _, images = text_from_rendered(rendered)

    page_texts = [_clean_page_text(p) for p in _split_paginated_text(text)]
    image_metas = _save_marker_images(images, images_dir)

    logger.info(f"Marker extracted {len(page_texts)} pages, {len(image_metas)} images from {pdf_path}")
    return page_texts, image_metas


class MarkerBackend(PdfParserBackend):
    name = "marker"

    def version(self) -> str:
        try:
            from importlib.metadata import version
            return version("marker-pdf")
        except Exception:
            return ""

    def check_available(self) -> None:
        try:
            import marker  # noqa: F401
        except ImportError as e:
            raise ParserUnavailableError(
                "The 'marker' PDF backend requires marker-pdf. "
                "Install it with: pip install 'marker-pdf==1.10.2'"
            ) from e

    def convert(self, pdf_path: str, images_dir: str) -> BackendOutput:
        # Resolve through pdf_parser so callers that patch
        # polyreader.parsing.pdf_parser._convert_with_marker keep working.
        from . import pdf_parser

        with _convert_lock:
            page_texts, images = pdf_parser._convert_with_marker(pdf_path, images_dir)
        return BackendOutput(
            page_texts=page_texts,
            images=images,
            info={"backend": self.name, "version": self.version()},
        )

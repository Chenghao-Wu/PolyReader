"""Reuse a benchmark parse for indexing, without re-running the PDF parser.

The downstream comparison indexes each document from the *same* parse that
was benchmarked:

    from saved_parse import use_saved_parse
    with use_saved_parse("~/polyreader/bench/outputs/mineru_standard/P2_polym_star_raft/polymers-15-02002"):
        index = asyncio.run(polyreader.process_document(pdf_path, ...))   # parse step is skipped
        polyreader.save_index(index, "out/<name>.json")   # inside the block: images live in a temp dir

``use_saved_parse`` patches ``polyreader.indexing.paper_index.parse_pdf`` so ``process_document``
receives a ``ParseResult`` built from ``page_contents.json`` + ``images/``.
Images are copied to a temporary directory first, because ``save_index`` moves
the images directory into the index folder.
"""

from __future__ import annotations

import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path

from common import load_page_texts, read_json


def load_parse_result(doc_dir: str | Path, images_dir: str | Path | None = None):
    """Build a ParseResult from a saved benchmark output directory."""
    from polyreader.parsing.pdf_parser import ImageMeta, ParseResult

    d = Path(doc_dir).expanduser()
    meta = read_json(d / "parse_meta.json") if (d / "parse_meta.json").exists() else {}
    page_texts = load_page_texts(d)
    src_images = d / "images"
    images: list = []
    for name in meta.get("images", []):
        f = src_images / name
        if f.is_file():
            images.append(ImageMeta(filename=name, page=0, index=0, width=0, height=0, x=0.0, y=0.0,
                                    bbox_width=0.0, bbox_height=0.0, size_bytes=f.stat().st_size,
                                    format=f.suffix.lstrip("."), colorspace=""))
    return ParseResult(
        markdown="\n\n".join(page_texts),
        page_texts=page_texts,
        images=images,
        images_dir=str(images_dir or src_images),
        mode_used=meta.get("backend", ""),
        pdf_path=meta.get("pdf", ""),
        parser_info=meta.get("parser_info", {}),
    )


@contextmanager
def use_saved_parse(doc_dir: str | Path):
    """Make polyreader.process_document() use the saved parse in ``doc_dir``."""
    from polyreader.indexing import paper_index

    d = Path(doc_dir).expanduser()
    tmp = Path(tempfile.mkdtemp(prefix="saved_parse_"))
    images_copy = tmp / "images"
    if (d / "images").is_dir():
        shutil.copytree(d / "images", images_copy)
    else:
        images_copy.mkdir()
    result = load_parse_result(d, images_dir=images_copy)

    async def _fake_parse_pdf(pdf_path, *args, **kwargs):
        result.pdf_path = str(Path(pdf_path).resolve())
        return result

    original = paper_index.parse_pdf
    paper_index.parse_pdf = _fake_parse_pdf
    try:
        yield result
    finally:
        paper_index.parse_pdf = original
        shutil.rmtree(tmp, ignore_errors=True)

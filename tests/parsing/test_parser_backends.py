"""PDF parser backend selection and the Marker adapter.

The real-Marker test runs only with POLYREADER_RUN_MARKER=1 (needs the
marker models; a GPU makes it fast).
"""

from __future__ import annotations

import os
import sys

import pytest

from polyreader.parsing import pdf_parser
from polyreader.parsing import ParserUnavailableError, get_backend
from polyreader.parsing.pdf_parser import ImageMeta, parse_pdf


def _tiny_pdf(path, pages=2):
    fitz = pytest.importorskip("fitz")
    doc = fitz.open()
    for i in range(pages):
        page = doc.new_page()
        page.insert_text((72, 72), f"Poly(L-lactide) sample P{i + 1}: Mn = 12 300 g/mol, Tg = 55 C")
    doc.save(str(path))
    doc.close()
    return str(path)


def test_unknown_backend():
    with pytest.raises(ValueError, match="Available: marker, mineru"):
        get_backend("pymupdf")


def test_marker_takes_no_options():
    with pytest.raises(ValueError):
        get_backend("marker", tier="standard")


@pytest.mark.asyncio
async def test_missing_pdf(tmp_path):
    with pytest.raises(FileNotFoundError):
        await parse_pdf(str(tmp_path / "missing.pdf"))


@pytest.mark.asyncio
async def test_default_backend_is_marker(tmp_path, monkeypatch):
    pdf = _tiny_pdf(tmp_path / "doc.pdf")
    calls = {}

    def fake_convert(pdf_path, images_dir):
        calls["args"] = (pdf_path, images_dir)
        meta = ImageMeta("_page_0_Figure_1.png", 1, 1, 1, 1, 0, 0, 0, 0, 10, "png", "RGB")
        return ["page one", "page two"], [meta]

    monkeypatch.setitem(sys.modules, "marker", type(sys)("marker"))  # pretend installed
    monkeypatch.setattr(pdf_parser, "_convert_with_marker", fake_convert)
    result = await parse_pdf(pdf, output_dir=str(tmp_path / "out"))
    assert result.mode_used == "marker"
    assert result.page_texts == ["page one", "page two"]
    assert result.markdown == "page one\n\npage two"
    assert result.parser_info["backend"] == "marker"
    assert calls["args"][1] == str(tmp_path / "out" / "images")


@pytest.mark.asyncio
async def test_marker_missing_gives_install_error(tmp_path, monkeypatch):
    pdf = _tiny_pdf(tmp_path / "doc.pdf")
    monkeypatch.setitem(sys.modules, "marker", None)  # import marker -> ImportError
    with pytest.raises(ParserUnavailableError, match="marker-pdf"):
        await parse_pdf(pdf, output_dir=str(tmp_path / "out"))


@pytest.mark.asyncio
async def test_mineru_selected_but_missing_is_an_error_not_a_fallback(tmp_path, monkeypatch):
    pdf = _tiny_pdf(tmp_path / "doc.pdf")
    monkeypatch.delenv("POLYREADER_MINERU_EXECUTABLE", raising=False)
    monkeypatch.delenv("PAPERINDEX_MINERU_EXECUTABLE", raising=False)
    monkeypatch.setenv("PATH", str(tmp_path))  # no mineru-kit anywhere
    called = []
    monkeypatch.setattr(pdf_parser, "_convert_with_marker", lambda *a: called.append(a))
    with pytest.raises(ParserUnavailableError):
        await parse_pdf(pdf, output_dir=str(tmp_path / "out"), backend="mineru")
    assert called == []  # Marker was not used silently


@pytest.mark.skipif(os.environ.get("POLYREADER_RUN_MARKER") != "1",
                    reason="set POLYREADER_RUN_MARKER=1 to run Marker for real")
@pytest.mark.asyncio
async def test_real_marker_page_mapping(tmp_path):
    pdf = _tiny_pdf(tmp_path / "doc.pdf", pages=3)
    result = await parse_pdf(pdf, output_dir=str(tmp_path / "out"))
    assert len(result.page_texts) == 3
    assert "P2" in result.page_texts[1] and "P3" in result.page_texts[2]
    assert result.parser_info["version"]


@pytest.mark.asyncio
async def test_marker_conversions_are_serialized(tmp_path, monkeypatch):
    """Concurrent parse_pdf calls must never run two Marker conversions at once."""
    import asyncio
    import threading
    import time

    pdfs = [_tiny_pdf(tmp_path / f"doc{i}.pdf") for i in range(3)]
    active, peak, lock = [0], [0], threading.Lock()

    def fake_convert(pdf_path, images_dir):
        with lock:
            active[0] += 1
            peak[0] = max(peak[0], active[0])
        time.sleep(0.05)
        with lock:
            active[0] -= 1
        return ["text"], []

    monkeypatch.setitem(sys.modules, "marker", type(sys)("marker"))
    monkeypatch.setattr(pdf_parser, "_convert_with_marker", fake_convert)
    await asyncio.gather(*(parse_pdf(p, output_dir=str(tmp_path / f"out{i}")) for i, p in enumerate(pdfs)))
    assert peak[0] == 1

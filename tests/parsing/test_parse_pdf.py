"""Quick integration test for parse_pdf — run with: python test_parse_pdf.py"""

import asyncio
import tempfile
from pathlib import Path

from polyreader.parsing.pdf_parser import parse_pdf

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PDF_PATH = str(
    PROJECT_ROOT
    / "reference"
    / "10.1021_ma202814c"
    / "stereocomplexation-of-poly(l-lactide)-and-random-copolymer-poly(d-lactide-co-ε-caprolactone)-to-enhance-melt-stability.pdf"
)


async def main():
    print("=" * 60)
    print("Test 1: parse_pdf (Marker extraction)")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        result = await parse_pdf(PDF_PATH, output_dir=tmpdir)

        print(f"  Pages extracted : {len(result.page_texts)}")
        print(f"  Images extracted: {len(result.images)}")
        print(f"  mode_used       : {result.mode_used}")
        print(f"  images_dir exists: {Path(result.images_dir).is_dir()}")
        print(f"  Page 1 text (first 200 chars):")
        print(f"    {result.page_texts[2]!r}")

        assert len(result.page_texts) > 0, "No pages extracted"
        assert len(result.images) > 0, "No images extracted"
        assert result.mode_used == "marker"
        assert Path(result.images_dir).is_dir(), "images_dir not created"

        print("\n  All assertions passed.\n")

    print("\n" + "=" * 60)
    print("All tests passed.")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())

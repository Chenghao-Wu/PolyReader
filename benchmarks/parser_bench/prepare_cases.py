"""Build the benchmark case manifest (bench_root/cases.json).

Each sub-directory of ``bench_root/papers`` is one case. Every PDF in it is a
document to parse, except ``*_source.pdf`` files, which are kept only as text
references: for a case containing ``X_source.pdf`` an image-only ``X_scanned.pdf``
is generated (pages rasterized, no text layer) to emulate a scanned paper, and
the source's text layer is used as its reference.

Roles: filenames containing "suppl", "supporting" or "_si" are SI, else main.
SI documents longer than ``--max-si-pages`` are benchmarked on a trimmed copy
of their first pages, written to ``bench_root/derived/<case>/`` (the original
stays untouched in papers/ and is recorded as ``trimmed_from``).

    python prepare_cases.py --bench-root ~/polyreader/bench [--dpi 200] [--max-si-pages 25]
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from common import write_json

_SI_PATTERN = re.compile(r"suppl|supporting|(^|[_\-.])si([_\-.]|$)", re.IGNORECASE)


def rasterize(src: Path, dst: Path, dpi: int) -> None:
    """Write an image-only copy of ``src`` (one JPEG per page, no text layer)."""
    import pymupdf as fitz

    zoom = dpi / 72
    with fitz.open(str(src)) as doc, fitz.open() as out:
        for page in doc:
            pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
            new = out.new_page(width=page.rect.width, height=page.rect.height)
            new.insert_image(new.rect, stream=pix.tobytes("jpeg", jpg_quality=90))
        out.save(str(dst), deflate=True)
    with fitz.open(str(dst)) as check:
        assert all(not p.get_text().strip() for p in check), "scanned PDF still has a text layer"


def trim(src: Path, dst: Path, pages: int) -> None:
    """Write the first ``pages`` pages of ``src`` to ``dst``."""
    import pymupdf as fitz

    dst.parent.mkdir(parents=True, exist_ok=True)
    with fitz.open(str(src)) as doc, fitz.open() as out:
        out.insert_pdf(doc, from_page=0, to_page=pages - 1)
        out.save(str(dst), garbage=3, deflate=True)


def page_count(pdf: Path) -> int:
    import pymupdf as fitz

    with fitz.open(str(pdf)) as d:
        return d.page_count


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bench-root", required=True, type=Path)
    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument("--max-si-pages", type=int, default=25,
                    help="Benchmark only the first N pages of longer SI documents (0 = no limit)")
    args = ap.parse_args()

    root = args.bench_root.expanduser().resolve()
    cases = []
    for case_dir in sorted(p for p in (root / "papers").iterdir() if p.is_dir()):
        docs = []
        for src in sorted(case_dir.glob("*_source.pdf")):
            scanned = src.with_name(src.name.replace("_source.pdf", "_scanned.pdf"))
            if not scanned.exists():
                rasterize(src, scanned, args.dpi)
                print(f"rasterized {src.name} -> {scanned.name} @ {args.dpi} dpi")
        for pdf in sorted(case_dir.glob("*.pdf")):
            if pdf.name.endswith("_source.pdf"):
                continue
            ref = pdf
            if pdf.name.endswith("_scanned.pdf"):
                ref = pdf.with_name(pdf.name.replace("_scanned.pdf", "_source.pdf"))
            role = "si" if _SI_PATTERN.search(pdf.stem) else "main"
            entry = {"role": role, "scanned": pdf.name.endswith("_scanned.pdf"), "pages": page_count(pdf)}
            if role == "si" and args.max_si_pages and entry["pages"] > args.max_si_pages:
                n = args.max_si_pages
                trimmed = root / "derived" / case_dir.name / f"{pdf.stem}_p1-{n}.pdf"
                if not trimmed.exists():
                    trim(pdf, trimmed, n)
                    print(f"trimmed {pdf.name} ({entry['pages']} pages) -> {trimmed.name}")
                entry.update({"trimmed_from": str(pdf.relative_to(root)), "original_pages": entry["pages"],
                              "pages": n})
                pdf = ref = trimmed
            docs.append({"pdf": str(pdf.relative_to(root)), "reference_pdf": str(ref.relative_to(root)), **entry})
        docs.sort(key=lambda d: (d["role"] != "main", d["pdf"]))
        cases.append({"case": case_dir.name, "docs": docs})

    write_json(root / "cases.json", {"cases": cases})
    for c in cases:
        print(c["case"], [(Path(d["pdf"]).name, d["role"]) for d in c["docs"]])


if __name__ == "__main__":
    main()

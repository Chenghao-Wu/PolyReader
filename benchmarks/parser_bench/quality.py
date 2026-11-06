"""Automatic document-reconstruction metrics for saved parser outputs.

Reference = the PyMuPDF text layer of the born-digital PDF (for a scanned case,
the text layer of its ``*_source.pdf``). The reference has its own flaws
(running headers, raw column order, flattened tables), so these numbers are a
screening signal; table cell accuracy must still be checked by eye using the
table dumps written to ``<bench_root>/reports/tables/``.

Metrics per document:
  page_count_match        parsed pages == PDF pages
  token_recall/precision  multiset word overlap vs. reference (whole doc, and per page)
  page_mapping_accuracy   share of pages whose text best matches reference page i
                          (vs. i-1 / i+1) — detects shifted or merged pages
  order_score             share of shared words appearing in reference order
                          (difflib matching blocks / shared words), per page, averaged
  headings, tables, table_cells, equations_display/inline, images, broken_image_refs
  symbol_retention        polymer/chemistry notation (Đ, Mn, Mw, Tg, °C, Greek, -co-, …)

    python quality.py --bench-root ~/polyreader/bench [--configs marker mineru_standard]
"""

from __future__ import annotations

import argparse
import difflib
import html
import re
import unicodedata
from collections import Counter
from html.parser import HTMLParser
from pathlib import Path

from common import load_manifest, doc_key, load_page_texts, read_json, write_json

# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

_DASHES = dict.fromkeys(map(ord, "\u2010\u2011\u2012\u2013\u2014\u2212\u2043"), "-")
# Look-alike glyphs used for degree signs and the dispersity symbol (before NFKC,
# which would turn the masculine ordinal into "o").
_LOOKALIKES = {ord("º"): "°", ord("˚"): "°", ord("◦"): "°", ord("∘"): "°",
               ord("Ð"): "Đ", ord("Ɖ"): "Đ"}

_LATEX_SYMBOLS = {
    "alpha": "α", "beta": "β", "gamma": "γ", "delta": "δ", "epsilon": "ε", "varepsilon": "ε",
    "zeta": "ζ", "eta": "η", "theta": "θ", "kappa": "κ", "lambda": "λ", "mu": "μ", "nu": "ν",
    "xi": "ξ", "pi": "π", "rho": "ρ", "sigma": "σ", "tau": "τ", "phi": "φ", "varphi": "φ",
    "chi": "χ", "psi": "ψ", "omega": "ω", "Delta": "Δ", "Gamma": "Γ", "Theta": "Θ",
    "Lambda": "Λ", "Sigma": "Σ", "Phi": "Φ", "Psi": "Ψ", "Omega": "Ω", "DJ": "Đ", "Dj": "Đ",
    "circ": "°", "degree": "°", "times": "×", "pm": "±", "cdot": "·", "approx": "≈",
    "leq": "≤", "geq": "≥", "le": "≤", "ge": "≥", "sim": "~", "%": "%", "rightarrow": "→",
    "to": "→", "infty": "∞", "partial": "∂", "langle": "⟨", "rangle": "⟩",
}

_IMG_RE = re.compile(r"!\[[^\]]*\]\(([^)\s]+)\)")
_DISPLAY_MATH_RE = re.compile(r"\$\$(.+?)\$\$|\\\[(.+?)\\\]", re.S)
_INLINE_MATH_RE = re.compile(r"(?<![\\$])\$(?!\$)([^$\n]+?)(?<!\\)\$|\\\((.+?)\\\)", re.S)
_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s+\S", re.M)
_PIPE_SEP_RE = re.compile(r"^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?\s*$")


def latex_to_text(s: str) -> str:
    """Flatten simple LaTeX into plain text: \\mathrm{M}_{n} -> Mn, ^{\\circ}C -> °C."""
    s = re.sub(r"\\mathcal\s*\{?\s*D\s*\}?", "Đ", s)  # common LaTeX spelling of dispersity
    s = re.sub(r"\\(?:mathrm|text|textrm|mathit|mathbf|textbf|textit|operatorname|boldsymbol|bm|mathsf|mathcal|rm|it|bf)\b\s*", "", s)
    s = re.sub(r"\\([A-Za-z]+)", lambda m: _LATEX_SYMBOLS.get(m.group(1), " "), s)
    s = re.sub(r"\\[,;:! ]", " ", s)
    s = s.replace("\\%", "%").replace("\\_", "_").replace("\\&", "&")
    s = re.sub(r"[_^]", "", s)
    s = s.replace("{", "").replace("}", "").replace("~", " ")
    return s


class _TextExtractor(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag not in ("sub", "sup", "i", "b", "em", "strong", "span"):
            self.parts.append(" ")

    def handle_endtag(self, tag):
        if tag not in ("sub", "sup", "i", "b", "em", "strong", "span"):
            self.parts.append(" ")

    def handle_data(self, data):
        self.parts.append(data)


def markdown_to_text(md: str) -> str:
    """Parsed Markdown/HTML/LaTeX -> plain text comparable to the PDF text layer."""
    s = _IMG_RE.sub(" ", md)
    s = _DISPLAY_MATH_RE.sub(lambda m: " " + latex_to_text(m.group(1) or m.group(2) or "") + " ", s)
    s = _INLINE_MATH_RE.sub(lambda m: latex_to_text(m.group(1) or m.group(2) or ""), s)
    if "<" in s:
        p = _TextExtractor()
        p.feed(s)
        p.close()
        s = "".join(p.parts)
    s = html.unescape(s)
    s = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", s)  # markdown links -> text
    s = re.sub(r"^\s{0,3}#{1,6}\s*", "", s, flags=re.M)
    s = re.sub(r"[*_`]+", "", s)  # emphasis markers: "-*b*-" -> "-b-", "*M*<sub>n</sub>" -> "Mn"
    s = re.sub(r"[|>]+", " ", s)
    s = re.sub(r"^\s*:?-{3,}:?\s*$", " ", s, flags=re.M)
    return normalize(s)


def normalize(s: str) -> str:
    s = unicodedata.normalize("NFKC", s.translate(_LOOKALIKES)).translate(_DASHES)
    s = s.replace("\u00ad", "")  # soft hyphen
    return s


def reference_pages(pdf: Path) -> list[str]:
    import pymupdf as fitz

    pages = []
    with fitz.open(str(pdf)) as doc:
        for page in doc:
            t = normalize(page.get_text("text"))
            t = re.sub(r"(\w)-\n(\w)", r"\1\2", t)  # undo line-break hyphenation
            pages.append(t)
    return pages


def tokens(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


# ---------------------------------------------------------------------------
# Structure counters
# ---------------------------------------------------------------------------

class _TableParser(HTMLParser):
    """Collect rows/cells of every <table> (colspan/rowspan noted in the cell text)."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.tables: list[list[list[str]]] = []
        self._depth = 0
        self._row: list[str] | None = None
        self._cell: list[str] | None = None
        self._span = ""

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == "table":
            self._depth += 1
            if self._depth == 1:
                self.tables.append([])
        elif tag == "tr" and self._depth:
            self._row = []
        elif tag in ("td", "th") and self._depth:
            self._cell = []
            spans = [f"{k}={a[k]}" for k in ("colspan", "rowspan") if a.get(k) and a.get(k) != "1"]
            self._span = f" [{' '.join(spans)}]" if spans else ""
        elif tag == "br" and self._cell is not None:
            self._cell.append(" ")

    def handle_endtag(self, tag):
        if tag == "table":
            self._depth = max(0, self._depth - 1)
        elif tag in ("td", "th") and self._cell is not None and self._row is not None:
            self._row.append(re.sub(r"\s+", " ", "".join(self._cell)).strip() + self._span)
            self._cell = None
        elif tag == "tr" and self._row is not None and self.tables:
            self.tables[-1].append(self._row)
            self._row = None

    def handle_data(self, data):
        if self._cell is not None:
            self._cell.append(data)


def extract_tables(md: str) -> list[list[list[str]]]:
    """All tables in a page: HTML tables and Markdown pipe tables, as rows of cells."""
    tables: list[list[list[str]]] = []
    if "<table" in md.lower():
        p = _TableParser()
        p.feed(md)
        p.close()
        tables.extend(t for t in p.tables if t)
    lines = md.splitlines()
    i = 0
    while i < len(lines):
        if i + 1 < len(lines) and "|" in lines[i] and _PIPE_SEP_RE.match(lines[i + 1]):
            rows = [lines[i]]
            j = i + 2
            while j < len(lines) and lines[j].strip().startswith("|"):
                rows.append(lines[j])
                j += 1
            tables.append([[c.strip() for c in r.strip().strip("|").split("|")] for r in rows])
            i = j
        else:
            i += 1
    return tables


SYMBOLS: list[tuple[str, str]] = [
    ("Đ (dispersity)", r"Đ"),
    ("Mn", r"\bMn\b"),
    ("Mw", r"\bMw\b"),
    ("Tg", r"\bTg\b"),
    ("Tm", r"\bTm\b"),
    ("°C", r"°\s?C"),
    ("g/mol", r"\bg\s*(?:/|·)?\s*mol\b"),
    ("mol%", r"mol\s?%"),
    ("-co-", r"-co-"),
    ("-b-", r"-b-"),
    ("×10", r"×\s?10"),
    ("±", r"±"),
    ("Sn(Oct)2", r"Sn\s*\(\s*Oct\s*\)\s*2"),
    ("α", r"α"), ("β", r"β"), ("δ", r"δ"), ("ε", r"ε"), ("χ", r"χ"),
    ("μ", r"μ"), ("λ", r"λ"), ("θ", r"θ"), ("η", r"η"), ("σ", r"σ"), ("Δ", r"Δ"),
]


def symbol_counts(text: str) -> dict[str, int]:
    return {name: len(re.findall(rx, text)) for name, rx in SYMBOLS}


# ---------------------------------------------------------------------------
# Per-document metrics
# ---------------------------------------------------------------------------

def _overlap(a: Counter, b: Counter) -> int:
    return sum((a & b).values())


def _order_score(ref_tok: list[str], par_tok: list[str]) -> float | None:
    shared = _overlap(Counter(ref_tok), Counter(par_tok))
    if shared < 20:
        return None
    sm = difflib.SequenceMatcher(None, ref_tok, par_tok, autojunk=False)
    matched = sum(b.size for b in sm.get_matching_blocks())
    return matched / shared


def doc_metrics(doc_dir: Path, ref_pdf: Path, pdf_pages: int) -> tuple[dict, list[dict]]:
    pages_md = load_page_texts(doc_dir)
    ref = reference_pages(ref_pdf)
    images_dir = doc_dir / "images"

    par_text = [markdown_to_text(p) for p in pages_md]
    ref_tok = [tokens(t) for t in ref]
    par_tok = [tokens(t) for t in par_text]
    ref_c = [Counter(t) for t in ref_tok]
    par_c = [Counter(t) for t in par_tok]

    per_page = []
    mapped_ok = mapped_n = 0
    order_scores = []
    for i in range(max(len(ref), len(pages_md))):
        r = ref_c[i] if i < len(ref) else Counter()
        p = par_c[i] if i < len(pages_md) else Counter()
        inter = _overlap(r, p)
        rec = inter / sum(r.values()) if r else None
        prec = inter / sum(p.values()) if p else None
        best = None
        if i < len(pages_md) and sum(p.values()) >= 30 and i < len(ref):
            cands = {j: _overlap(p, ref_c[j]) for j in (i - 1, i, i + 1) if 0 <= j < len(ref)}
            best = max(cands, key=cands.get)
            mapped_n += 1
            mapped_ok += int(best == i)
        order = _order_score(ref_tok[i], par_tok[i]) if i < len(ref) and i < len(pages_md) else None
        if order is not None:
            order_scores.append(order)
        per_page.append({"page": i + 1, "recall": _r(rec), "precision": _r(prec),
                         "best_ref_page": None if best is None else best + 1, "order_score": _r(order)})

    R, P = sum(ref_c, Counter()), sum(par_c, Counter())
    inter = _overlap(R, P)

    md_all = "\n\n".join(pages_md)
    tables_all = [t for p in pages_md for t in extract_tables(p)]
    img_refs = _IMG_RE.findall(md_all)
    broken = [x for x in img_refs if not ((images_dir / x).is_file() or (doc_dir / x).is_file())]
    display_eq = len(_DISPLAY_MATH_RE.findall(md_all)) + md_all.count('<math display="block"')
    inline_eq = len(_INLINE_MATH_RE.findall(_DISPLAY_MATH_RE.sub(" ", md_all))) + \
        max(0, md_all.count("<math") - md_all.count('<math display="block"'))

    sym_ref = symbol_counts("\n".join(ref))
    sym_par = symbol_counts("\n".join(par_text))
    kept = sum(min(sym_par[k], v) for k, v in sym_ref.items())
    total = sum(sym_ref.values())

    metrics = {
        "pdf_pages": pdf_pages,
        "parsed_pages": len(pages_md),
        "page_count_match": len(pages_md) == pdf_pages,
        "empty_pages": sum(1 for p in pages_md if not p.strip()),
        "ref_tokens": sum(R.values()),
        "parsed_tokens": sum(P.values()),
        "token_recall": _r(inter / sum(R.values()) if R else None),
        "token_precision": _r(inter / sum(P.values()) if P else None),
        "page_mapping_accuracy": _r(mapped_ok / mapped_n if mapped_n else None),
        "order_score": _r(sum(order_scores) / len(order_scores) if order_scores else None),
        "headings": len(_HEADING_RE.findall(md_all)),
        "tables": len(tables_all),
        "table_cells": sum(len(row) for t in tables_all for row in t),
        "equations_display": display_eq,
        "equations_inline": inline_eq,
        "image_refs": len(img_refs),
        "image_files": len([f for f in images_dir.iterdir() if f.is_file()]) if images_dir.is_dir() else 0,
        "broken_image_refs": len(broken),
        "symbol_retention": _r(kept / total if total else None),
        "symbols_ref": sym_ref,
        "symbols_parsed": sym_par,
    }
    return metrics, per_page


def _r(x, nd: int = 4):
    return None if x is None else round(x, nd)


# ---------------------------------------------------------------------------
# Table dump for manual inspection
# ---------------------------------------------------------------------------

def render_table(rows: list[list[str]]) -> str:
    return "\n".join("| " + " | ".join(c.replace("|", "\\|") for c in r) + " |" for r in rows)


def write_table_report(bench_root: Path, configs: list[str], cases: list[dict]) -> list[Path]:
    written = []
    for case in cases:
        for doc in case["docs"]:
            key = doc_key(doc)
            lines = [f"# Tables — {case['case']} / {key}", ""]
            for cfg in configs:
                d = bench_root / "outputs" / cfg / case["case"] / key
                if not (d / "page_contents.json").exists():
                    continue
                lines.append(f"## {cfg}")
                n = 0
                for pno, page in enumerate(load_page_texts(d), start=1):
                    for t in extract_tables(page):
                        n += 1
                        lines += [f"### {cfg} — page {pno}, table {n} ({len(t)} rows)", "", render_table(t), ""]
                if not n:
                    lines += ["(no tables found)", ""]
            out = bench_root / "reports" / "tables" / case["case"] / f"{key}.md"
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text("\n".join(lines), encoding="utf-8")
            written.append(out)
    return written


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bench-root", required=True, type=Path)
    ap.add_argument("--configs", nargs="*", help="Output configs to score (default: all under outputs/)")
    args = ap.parse_args()

    root = args.bench_root.expanduser().resolve()
    cases = load_manifest(root)
    configs = args.configs or sorted(p.name for p in (root / "outputs").iterdir() if p.is_dir())

    results: dict = {"configs": {}}
    for cfg in configs:
        docs = []
        for case in cases:
            for doc in case["docs"]:
                key = doc_key(doc)
                d = root / "outputs" / cfg / case["case"] / key
                if not (d / "page_contents.json").exists():
                    docs.append({"case": case["case"], "doc": key, "missing": True})
                    continue
                meta = read_json(d / "parse_meta.json") if (d / "parse_meta.json").exists() else {}
                pdf_pages = meta.get("pdf_pages") or 0
                if not pdf_pages:
                    from common import pdf_page_count
                    pdf_pages = pdf_page_count(root / doc["pdf"])
                m, per_page = doc_metrics(d, root / doc["reference_pdf"], pdf_pages)
                docs.append({"case": case["case"], "doc": key, "role": doc["role"],
                             "scanned": doc.get("scanned", False), **m, "per_page": per_page})
                print(f"{cfg:18s} {case['case']}/{key}: recall={m['token_recall']} prec={m['token_precision']} "
                      f"map={m['page_mapping_accuracy']} order={m['order_score']} tables={m['tables']} "
                      f"eq={m['equations_display']}/{m['equations_inline']} sym={m['symbol_retention']}")
        results["configs"][cfg] = {"docs": docs}
    write_json(root / "results" / "quality.json", results)
    written = write_table_report(root, configs, cases)
    print(f"wrote {root / 'results' / 'quality.json'} and {len(written)} table dumps under reports/tables/")


if __name__ == "__main__":
    main()

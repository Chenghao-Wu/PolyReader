"""Tests for the optional MinerU PDF backend.

MinerU itself is not needed: a fake ``mineru-kit`` script replays trimmed real
MinerU 4.1.4 outputs (``fixtures/mineru``) into the zip layout the real CLI
writes, and simulates its failure modes (teardown segfault, hang, no output).
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path

import pytest

from polyreader.parsing import ParserError, ParserUnavailableError, get_backend
from polyreader.parsing.mineru_parser import (
    ENV_EXECUTABLE,
    MinerUBackend,
    _join_continuation,
    _PageRenderer,
)

fitz = pytest.importorskip("fitz")

FIXTURES = Path(__file__).parent / "fixtures" / "mineru"

FAKE_MINERU = r'''#!{python}
"""Fake mineru-kit: replays a fixture dir as MinerU's zip output."""
import json, os, re, signal, subprocess, sys, time, zipfile
from pathlib import Path

args = sys.argv[1:]
log = os.environ.get("FAKE_MINERU_LOG")
if log:
    with open(log, "a") as f:
        f.write(json.dumps({{"args": args, "vk": os.environ.get("GGML_VK_VISIBLE_DEVICES")}}) + "\n")
if args[:1] == ["--version"]:
    print("MinerU version: 9.9.9\nPython version: 3.12")
    sys.exit(0)

mode = os.environ.get("FAKE_MINERU_MODE", "ok")
pdf = Path(args[1])
out = Path(args[args.index("-o") + 1])
if mode == "timeout":
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
    Path(os.environ["FAKE_MINERU_CHILD_PID"]).write_text(str(child.pid))
    time.sleep(120)
if mode == "fail":
    print("Error: Failed to parse document: PDFium data format error", file=sys.stderr)
    sys.exit(1)
if mode == "empty":
    sys.exit(0)

fixture = Path(os.environ["FAKE_MINERU_FIXTURE"])
tiny = Path(os.environ["FAKE_MINERU_TINY"]).read_bytes()
out.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(out / (pdf.stem + ".zip"), "w") as z:
    names = set()
    for f in sorted(fixture.iterdir()):
        if not f.is_file():
            continue
        if mode == "nojson" and f.suffix == ".json":
            continue
        if mode == "middle_only" and f.name == "structured_content.json":
            continue
        text = f.read_text(encoding="utf-8")
        names.update(re.findall(r"images/[A-Za-z0-9_.-]+", text))
        z.writestr("markdown.md" if f.suffix == ".md" else f.name, text)
    for n in sorted(names):
        z.writestr(n, tiny)
    if mode == "zipslip":
        z.writestr("../escaped.txt", "nope")
print("Parsed 1 input(s).")
sys.stdout.flush()
if mode == "segfault":
    os.kill(os.getpid(), signal.SIGSEGV)
sys.exit(0)
'''


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_pdf(path: Path, pages: int) -> Path:
    doc = fitz.open()
    for i in range(pages):
        page = doc.new_page()
        page.insert_text((72, 72), f"page {i + 1}")
    doc.save(str(path))
    doc.close()
    return path


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def fake(tmp_path, monkeypatch):
    """Install the fake mineru-kit; returns a helper to configure a run."""
    exe = tmp_path / "bin" / "mineru-kit"
    exe.parent.mkdir()
    exe.write_text(FAKE_MINERU.format(python=sys.executable))
    exe.chmod(0o755)
    log = tmp_path / "fake_calls.jsonl"
    monkeypatch.setenv("FAKE_MINERU_LOG", str(log))
    monkeypatch.setenv("FAKE_MINERU_TINY", str(FIXTURES / "tiny.jpg"))
    monkeypatch.delenv(ENV_EXECUTABLE, raising=False)

    class Fake:
        path = str(exe)
        work = tmp_path / "work"

        def configure(self, mode="ok", fixture="si"):
            monkeypatch.setenv("FAKE_MINERU_MODE", mode)
            src = fixture if isinstance(fixture, Path) else FIXTURES / fixture
            monkeypatch.setenv("FAKE_MINERU_FIXTURE", str(src))

        def calls(self):
            if not log.exists():
                return []
            return [json.loads(line) for line in log.read_text().splitlines()]

        def backend(self, **opts):
            self.work.mkdir(exist_ok=True)
            opts.setdefault("executable", self.path)
            opts.setdefault("work_dir", str(self.work))
            return MinerUBackend(**opts)

    return Fake()


def _convert(backend, pdf: Path, out: Path):
    return backend.convert(str(pdf), str(out / "images"))


# ---------------------------------------------------------------------------
# Page and image mapping
# ---------------------------------------------------------------------------


def test_pages_map_one_to_one_and_match_mineru_markdown(fake, tmp_path):
    fake.configure("ok", "si")
    pdf = _make_pdf(tmp_path / "si.pdf", 6)
    out = _convert(fake.backend(), pdf, tmp_path / "out")

    assert len(out.page_texts) == 6
    assert out.page_texts[0].startswith("# Supporting Information for")
    assert out.page_texts[5].startswith("## References")
    # Per-page texts joined == MinerU's own merged markdown (images flattened).
    expected = (FIXTURES / "si" / "markdown.md").read_text().replace("](images/", "](").strip()
    assert "\n\n".join(t for t in out.page_texts if t) == expected
    assert out.info["backend"] == "mineru"
    assert out.info["version"] == "4.1.4"  # from the output's producer metadata
    assert out.info["tier"] == "standard"
    assert out.info["pdf_pages"] == 6


def test_images_are_flat_referenced_and_on_correct_pages(fake, tmp_path):
    fake.configure("ok", "si")
    pdf = _make_pdf(tmp_path / "si.pdf", 6)
    out_dir = tmp_path / "out"
    out = _convert(fake.backend(), pdf, out_dir)

    assert len(out.images) == 7
    for meta in out.images:
        assert "/" not in meta.filename
        assert (out_dir / "images" / meta.filename).is_file()
        # MinerU names images page_<0-based page>_<kind>_<n>
        assert meta.page == int(re.match(r"page_(\d+)_", meta.filename).group(1)) + 1
        assert f"![]({meta.filename})" in out.page_texts[meta.page - 1]
        assert meta.size_bytes > 0
    by_page = {}
    for meta in out.images:
        by_page.setdefault(meta.page, []).append(meta.index)
    assert all(sorted(v) == list(range(1, len(v) + 1)) for v in by_page.values())
    assert not re.search(r"!\[\]\(images/", "\n".join(out.page_texts))


def test_page_image_retrievable_through_paper_index(fake, tmp_path):
    from polyreader.indexing import PaperIndex

    fake.configure("ok", "si")
    pdf = _make_pdf(tmp_path / "si.pdf", 6)
    out_dir = tmp_path / "out"
    out = _convert(fake.backend(), pdf, out_dir)

    index = PaperIndex(images_dir=str(out_dir / "images"), page_texts=out.page_texts)
    name = re.search(r"!\[\]\(([^)]+)\)", out.page_texts[2]).group(1)
    result = index.get_page_image(img_path=name)
    assert result["success"] is True
    assert result["image_base64"]


def test_tables_equations_footnotes_and_blank_pages(fake, tmp_path):
    fake.configure("ok", "p5_excerpt")
    pdf = _make_pdf(tmp_path / "paper.pdf", 12)
    out = _convert(fake.backend(), pdf, tmp_path / "out")

    texts = out.page_texts
    assert len(texts) == 12
    assert all(texts[i] == "" for i in (3, 4, 5, 7, 8, 9, 10, 11))
    # Page furniture (headers) dropped, page footnotes kept.
    assert "PCCP" not in texts[0] and "ROYAL SOCIETY OF CHEMISTRY" not in texts[0]
    assert "Natural Resources Canada" in texts[0]
    # A paragraph continued from the previous page stays on its own page.
    assert texts[1].startswith("solubility is the calculation of the Flory")
    # Display equations as LaTeX, not as the equation crop image.
    assert "$$\n\\chi = \\frac {\\Delta \\omega_ {1 2}}{R T}\\tag{1}\n$$" in texts[1]
    assert "equation" not in "".join(m.filename for m in out.images)
    # Table: caption above, table body, then footnote.
    page6 = texts[6]
    cap, table, note = page6.index("Table 1 $R_{g}$"), page6.index("| Polymer |"), page6.index("a Taken from ref.")
    assert cap < table < note
    assert "| PIB | 0.0137 | 0.595 |" in page6
    # Chart: image reference followed by its caption.
    assert page6.index("![](page_6_chart_7.jpg)") < page6.index("Fig. 5")
    assert [m.filename for m in out.images] == ["page_6_chart_7.jpg"]
    assert out.images[0].page == 7
    assert out.images[0].bbox_width > 0  # bbox converted to PDF points


def test_middle_json_fallback(fake, tmp_path):
    fake.configure("middle_only", "p5_excerpt")
    pdf = _make_pdf(tmp_path / "paper.pdf", 12)
    out = _convert(fake.backend(), pdf, tmp_path / "out")

    assert out.info["structure_source"] == "middle_json.json"
    assert len(out.page_texts) == 12
    assert "\\tag{1}\n$$" in out.page_texts[1]
    page6 = out.page_texts[6]
    assert page6.index("Table 1") < page6.index("<table>") < page6.index("a Taken from ref.")
    assert "![](page_6_chart_7.jpg)" in page6


def test_renderer_merges_continuations_and_orders_side_captions():
    refs = []
    renderer = _PageRenderer(lambda src, block: refs.append(src) or Path(src).name)
    text = renderer.render([
        {"type": "header", "content": "Journal"},
        {"type": "text", "content": "The polymer-"},
        {"type": "page_footnote", "content": "* footnote"},
        {"type": "chart", "bbox": [0.3, 0.36, 0.6, 0.5], "content": "", "image_source": "images/page_0_chart_1.jpg",
         "captions": [{"bbox": [0.28, 0.366, 0.3, 0.378], "content": "(a)"}]},
        {"type": "text", "content": "ization was fast.", "continues_prev": True},
        {"type": "paragraph_title", "level": 2, "content": "Results"},
    ])
    assert text == ("The polymerization was fast.\n\n* footnote\n\n(a)\n\n"
                    "![](page_0_chart_1.jpg)\n\n## Results")
    assert refs == ["images/page_0_chart_1.jpg"]


@pytest.mark.parametrize("prev,nxt,joined", [
    ("polymer-", "ization", "polymerization"),
    ("GLGE-", "difficult", "GLGE-difficult"),
    ("co-", "polymers", "co-polymers"),
    ("pages 10–", "12", "pages 10–12"),
    ("word", "next", "word next"),
    ("word", ", next", "word, next"),
])
def test_join_continuation(prev, nxt, joined):
    assert _join_continuation(prev, nxt) == joined


# ---------------------------------------------------------------------------
# Process handling and failures
# ---------------------------------------------------------------------------


def test_segfault_after_writing_output_is_success(fake, tmp_path):
    fake.configure("segfault", "si")
    pdf = _make_pdf(tmp_path / "si.pdf", 6)
    out = _convert(fake.backend(), pdf, tmp_path / "out")
    assert len(out.page_texts) == 6
    assert out.info["exit_code"] != 0


def _process_gone(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    stat = Path(f"/proc/{pid}/stat")
    return stat.exists() and stat.read_text().split(") ")[-1].startswith("Z")


def test_timeout_kills_process_group(fake, tmp_path, monkeypatch):
    child_pid_file = tmp_path / "child.pid"
    monkeypatch.setenv("FAKE_MINERU_CHILD_PID", str(child_pid_file))
    fake.configure("timeout", "si")
    pdf = _make_pdf(tmp_path / "si.pdf", 6)
    start = time.monotonic()
    # 8 s leaves a cold, loaded machine time to start the fake and its child.
    with pytest.raises(ParserError, match="timed out") as exc:
        _convert(fake.backend(timeout=8), pdf, tmp_path / "out")
    assert time.monotonic() - start < 40
    assert exc.value.diagnostics["timeout_s"] == 8
    assert exc.value.diagnostics["command"][1] == "parse"
    assert child_pid_file.exists(), "fake mineru-kit was killed before it started its worker"
    child = int(child_pid_file.read_text())
    deadline = time.monotonic() + 5
    while not _process_gone(child) and time.monotonic() < deadline:
        time.sleep(0.1)
    assert _process_gone(child), "MinerU worker process survived the timeout"
    assert list(fake.work.iterdir()) == []


def test_missing_executable_is_unavailable(tmp_path, monkeypatch):
    monkeypatch.delenv(ENV_EXECUTABLE, raising=False)
    with pytest.raises(ParserUnavailableError, match="MinerU executable not found"):
        MinerUBackend(executable=str(tmp_path / "nope" / "mineru-kit")).check_available()
    monkeypatch.setenv(ENV_EXECUTABLE, str(tmp_path / "also-missing"))
    with pytest.raises(ParserUnavailableError):
        MinerUBackend().check_available()
    monkeypatch.delenv(ENV_EXECUTABLE)
    monkeypatch.setenv("PATH", str(tmp_path))
    with pytest.raises(ParserUnavailableError, match="Install MinerU"):
        MinerUBackend().check_available()


def test_executable_from_env_var(fake, tmp_path, monkeypatch):
    fake.configure("ok", "si")
    monkeypatch.setenv(ENV_EXECUTABLE, fake.path)
    backend = MinerUBackend(work_dir=None)
    backend.check_available()
    assert backend.version() == "9.9.9"


def test_parse_pdf_reports_unavailable_backend(tmp_path, monkeypatch):
    from polyreader.parsing.pdf_parser import parse_pdf

    monkeypatch.delenv(ENV_EXECUTABLE, raising=False)
    pdf = _make_pdf(tmp_path / "x.pdf", 1)
    with pytest.raises(ParserUnavailableError):
        asyncio.run(parse_pdf(str(pdf), str(tmp_path / "out"), backend="mineru",
                              executable=str(tmp_path / "missing")))


@pytest.mark.parametrize("mode,match", [
    ("empty", "produced no output"),
    ("fail", "produced no output"),
    ("nojson", "no structured_content.json or middle_json.json"),
    ("zipslip", "unsafe path"),
])
def test_bad_runs_raise_with_diagnostics(fake, tmp_path, mode, match):
    fake.configure(mode, "si")
    pdf = _make_pdf(tmp_path / "si.pdf", 6)
    with pytest.raises(ParserError, match=match) as exc:
        _convert(fake.backend(), pdf, tmp_path / "out")
    diag = exc.value.diagnostics
    assert diag["command"][1:2] == ["parse"]
    assert "exit_code" in diag and "stderr_tail" in diag
    if mode == "fail":
        assert diag["exit_code"] == 1
        assert "PDFium" in diag["stderr_tail"]
    assert list(fake.work.iterdir()) == []  # temp dir removed on failure
    assert not (tmp_path / "out" / "images").exists() or not any((tmp_path / "out" / "images").iterdir())


def test_page_index_outside_pdf_is_an_error(fake, tmp_path):
    bad = tmp_path / "bad_fixture"
    bad.mkdir()
    data = json.loads((FIXTURES / "si" / "structured_content.json").read_text())
    data["pages"][-1]["page_idx"] = 40
    (bad / "structured_content.json").write_text(json.dumps(data))
    fake.configure("ok", bad)
    pdf = _make_pdf(tmp_path / "si.pdf", 6)
    with pytest.raises(ParserError, match="outside the PDF"):
        _convert(fake.backend(), pdf, tmp_path / "out")


def test_original_pdf_untouched_and_cli_arguments(fake, tmp_path):
    fake.configure("ok", "si")
    pdf = _make_pdf(tmp_path / "my.paper.v2.pdf", 6)  # dots in the stem are fine
    before = _sha(pdf)
    backend = fake.backend(tier="advanced", extra_env={"GGML_VK_VISIBLE_DEVICES": 3})
    out = _convert(backend, pdf, tmp_path / "out")

    assert _sha(pdf) == before
    assert sorted(p.name for p in pdf.parent.iterdir() if p.suffix == ".pdf") == ["my.paper.v2.pdf"]
    call = [c for c in fake.calls() if c["args"][0] == "parse"][-1]
    args = call["args"]
    assert args[1] != str(pdf) and args[1].endswith("document.pdf")
    assert args[args.index("-f") + 1] == "zip"
    assert args[args.index("--tier") + 1] == "advanced"
    assert "--remote" not in args and "--remote-url" not in args
    assert call["vk"] == "3"
    assert out.info["tier"] == "advanced"
    assert list(fake.work.iterdir()) == []  # temp dir removed on success


def test_keep_raw_output(fake, tmp_path):
    fake.configure("ok", "si")
    pdf = _make_pdf(tmp_path / "si.pdf", 6)
    out_dir = tmp_path / "out"
    out = _convert(fake.backend(keep_raw_output=True), pdf, out_dir)
    raw = out_dir / "mineru_raw"
    assert (raw / "structured_content.json").is_file()
    assert out.info["raw_output_dir"] == str(raw)


def test_page_count_mismatch_is_reported(fake, tmp_path):
    fake.configure("ok", "si")
    pdf = _make_pdf(tmp_path / "si.pdf", 8)  # MinerU metadata says 6
    out = _convert(fake.backend(), pdf, tmp_path / "out")
    assert len(out.page_texts) == 8
    assert out.page_texts[6] == out.page_texts[7] == ""
    assert any("reports 6 pages" in w for w in out.info["warnings"])


# ---------------------------------------------------------------------------
# Options and registry
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("opts", [{"tier": "ultra"}, {"timeout": 0}, {"timeout": "soon"}, {"extra_env": ["A=1"]}])
def test_invalid_options(opts):
    with pytest.raises(ValueError):
        MinerUBackend(**opts)


def test_registry_and_parse_pdf(fake, tmp_path):
    from polyreader.parsing.pdf_parser import parse_pdf

    backend = get_backend("mineru", tier="basic", executable=fake.path)
    assert isinstance(backend, MinerUBackend) and backend.tier == "basic"

    fake.configure("ok", "si")
    pdf = _make_pdf(tmp_path / "si.pdf", 6)
    result = asyncio.run(parse_pdf(str(pdf), str(tmp_path / "out"), backend="mineru",
                                   executable=fake.path, work_dir=str(tmp_path)))
    assert result.mode_used == "mineru"
    assert result.parser_info["backend"] == "mineru"
    assert len(result.page_texts) == 6
    assert result.images_dir == str(tmp_path / "out" / "images")
    assert len(result.images) == 7

"""MinerU backend (optional): runs the installed ``mineru-kit`` CLI in a subprocess.

MinerU is not a dependency of polyreader. It usually lives in its own
environment, so the executable can be given explicitly (``executable=`` option
or ``POLYREADER_MINERU_EXECUTABLE``); otherwise ``mineru-kit`` is looked up on
PATH. Each conversion copies the PDF into a private temp dir, runs::

    mineru-kit parse <copy> -o <tmp>/out -f zip --tier <tier>

and maps the zip's ``structured_content.json`` (per-page blocks with absolute
0-based ``page_idx``) to one Markdown string per PDF page. Images are copied
flat into ``images_dir`` and referenced as ``![](<filename>)``.

Success is judged by the produced files, not the exit code: mineru-kit 4.x
can segfault during GPU teardown after writing every output.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import signal
import subprocess
import tempfile
import time
import zipfile
from pathlib import Path
from typing import Any, Callable

from .base import BackendOutput, ParserError, ParserUnavailableError, PdfParserBackend

logger = logging.getLogger(__name__)

ENV_EXECUTABLE = "POLYREADER_MINERU_EXECUTABLE"
LEGACY_ENV_EXECUTABLE = "PAPERINDEX_MINERU_EXECUTABLE"  # pre-polyreader name, still honoured
DEFAULT_EXECUTABLE = "mineru-kit"
DEFAULT_TIER = "standard"
DEFAULT_TIMEOUT = 1800
TIERS = ("flash", "basic", "standard", "advanced")

INSTALL_HINT = (
    "Install MinerU (tested with 4.1.4: pip install 'mineru[full]>=4.1,<5') in its own "
    "environment and point polyreader at it, e.g. "
    "POLYREADER_MINERU_EXECUTABLE=~/miniconda3/envs/mineru/bin/mineru-kit, or pass "
    "executable=... in the 'mineru' config section."
)

# Page furniture that MinerU's own Markdown renderer drops.
_AUXILIARY_TYPES = {"header", "footer", "page_number", "aside_text"}
_TITLE_TYPES = {"doc_title", "title", "paragraph_title"}
_VISUAL_TYPES = {"image", "chart", "table", "code"}
_TAIL_CHARS = 4000


def _tail(text: str | bytes | None) -> str:
    if text is None:
        return ""
    if isinstance(text, bytes):
        text = text.decode("utf-8", errors="replace")
    return text[-_TAIL_CHARS:]


def _pdf_page_count(pdf_path: str) -> int | None:
    """Page count of the original PDF, independent of MinerU (None if unknown)."""
    try:
        import fitz  # PyMuPDF (polyreader dependency)

        with fitz.open(pdf_path) as doc:
            return doc.page_count
    except Exception:
        pass
    try:
        import pypdfium2

        return len(pypdfium2.PdfDocument(pdf_path))
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Loading MinerU's structured output
# ---------------------------------------------------------------------------

def _flatten_spans(content: Any) -> str:
    """Render middle_json inline spans (text / equation_inline / links) to Markdown."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        inner = _flatten_spans(content.get("content"))
        if content.get("type") == "equation_inline":
            return f"${inner}$"
        if "bold" in (content.get("styles") or []) and inner.strip():
            return f"**{inner}**"
        return inner
    if isinstance(content, list):
        return "".join(_flatten_spans(c) for c in content)
    return str(content)


def _normalize_middle_block(block: dict) -> dict:
    """Convert a middle_json block into the structured_content block shape."""
    kind = block.get("type", "")
    out: dict[str, Any] = {"type": kind, "bbox": block.get("bbox")}
    for key in ("level", "continues_prev"):
        if key in block:
            out[key] = block[key]
    children = block.get("content")
    if kind in _VISUAL_TYPES and isinstance(children, list):
        out.update(content="", captions=[], footnotes=[])
        for child in children:
            if not isinstance(child, dict):
                continue
            ctype = child.get("type", "")
            note = {"bbox": child.get("bbox"), "content": _flatten_spans(child.get("content"))}
            if ctype.endswith("_caption"):
                out["captions"].append(note)
            elif ctype.endswith("_footnote"):
                out["footnotes"].append(note)
            else:  # *_body
                out["content"] = _flatten_spans(child.get("content"))
                if child.get("image_path"):
                    out["image_source"] = child["image_path"]
                if child.get("bbox"):
                    out["bbox"] = child["bbox"]
        return out
    if kind in ("list", "index") and isinstance(children, list):
        out["content"] = "\n".join(f"- {_flatten_spans(c).strip()}" for c in children)
    else:
        out["content"] = _flatten_spans(children)
    if block.get("image_path"):
        out["image_source"] = block["image_path"]
    return out


def _load_structure(raw_dir: Path) -> tuple[list[dict], dict, dict, str]:
    """Return (pages, metadata, extensions, json_name) from the unpacked zip.

    Prefers structured_content.json (blocks already rendered to inline
    Markdown); falls back to middle_json.json (inline span lists).
    """
    structured = raw_dir / "structured_content.json"
    middle = raw_dir / "middle_json.json"
    if structured.is_file():
        data = json.loads(structured.read_text(encoding="utf-8"))
        pages = data.get("pages") or []
        name = structured.name
    elif middle.is_file():
        data = json.loads(middle.read_text(encoding="utf-8"))
        pages = [
            {"page_idx": p.get("page_idx"),
             "blocks": [_normalize_middle_block(b) for b in p.get("blocks") or [] if isinstance(b, dict)]}
            for p in data.get("pages") or [] if isinstance(p, dict)
        ]
        name = middle.name
    else:
        raise ParserError(
            "MinerU output has no structured_content.json or middle_json.json; "
            "cannot map content to PDF pages."
        )
    if not isinstance(data, dict) or not isinstance(pages, list):
        raise ParserError(f"Unexpected {name} layout (no 'pages' list)")
    return pages, data.get("metadata") or {}, data.get("extensions") or {}, name


# ---------------------------------------------------------------------------
# Page rendering
# ---------------------------------------------------------------------------

def _is_cjk(ch: str) -> bool:
    return "　" <= ch <= "鿿" or "＀" <= ch <= "￯"


_LINE_END_HYPHEN_RE = re.compile(r"([A-Za-z]+)[-­‐‑⁃]$")
_COMPOUND_PREFIXES = {"open", "non", "self", "cross", "anti", "pre", "post", "co", "semi", "multi", "quasi"}


def _join_continuation(prev: str, nxt: str) -> str:
    """Join a paragraph split across columns (MinerU ``continues_prev``).

    Follows MinerU's line-boundary rules: drop a word-break hyphen before a
    lowercase continuation (but keep acronym/compound hyphens), no space for
    CJK text or digit ranges, otherwise a single space.
    """
    prev, nxt = prev.rstrip(), nxt.lstrip()
    if not prev or not nxt:
        return prev or nxt
    hyphen = _LINE_END_HYPHEN_RE.search(prev)
    if hyphen:
        word = hyphen.group(1)
        keep = (len(word) > 1 and word.isupper()) or (prev.endswith("-") and word.lower() in _COMPOUND_PREFIXES)
        return prev + nxt if keep or not nxt[0].islower() else prev[:-1] + nxt
    if re.search(r"\d[-–]$", prev) and nxt[0].isdigit():
        return prev + nxt
    if _is_cjk(prev[-1]) or _is_cjk(nxt[0]) or nxt[0] in ".,;:!?)]}”’" or prev[-1] in "([{“‘":
        return prev + nxt
    return f"{prev} {nxt}"


def _bbox(item: Any) -> tuple[float, float, float, float] | None:
    try:
        x0, y0, x1, y1 = (float(v) for v in item["bbox"])
        return x0, y0, x1, y1
    except (TypeError, KeyError, ValueError):
        return None


def _reads_before(note: Any, body: tuple | None) -> bool:
    """True if a caption/footnote precedes the body in reading order (above or to its left)."""
    box = _bbox(note)
    if box is None or body is None:
        return False
    x0, y0, x1, y1 = box
    above = (y0 + y1) / 2 < body[1]
    left_side = x1 <= body[0] and y0 < body[3]
    return above or left_side


def _text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else _flatten_spans(value).strip()


class _PageRenderer:
    """Renders one page's blocks to Markdown, mirroring MinerU's markdown.md rules.

    ``image_ref(source, block)`` returns the flat filename to reference, or
    None when the image is missing (the reference is then dropped).
    """

    def __init__(self, image_ref: Callable[[str, dict], str | None]):
        self._image_ref = image_ref

    def _image_md(self, source: Any, block: dict) -> str:
        if not source or not isinstance(source, str):
            return ""
        name = self._image_ref(source, block)
        return f"![]({name})" if name else ""

    def _visual(self, block: dict) -> str:
        kind = block.get("type", "")
        content = _text(block.get("content"))
        image = block.get("image_source")
        if kind in ("image", "chart"):
            body = "\n\n".join(p for p in (self._image_md(image, block), content) if p)
        else:
            # Recognised table/code text wins; fall back to the crop if recognition failed.
            body = content or self._image_md(image, block)

        # Captions/footnotes keep their reading position relative to the body
        # (MinerU renders children in original order; the block bbox is the body bbox).
        body_box = _bbox(block)
        before, after, notes = [], [], []
        for key, trailing in (("captions", after), ("footnotes", notes)):
            for item in block.get(key) or []:
                text = _text(item.get("content")) if isinstance(item, dict) else _text(item)
                if text:
                    box = _bbox(item) or (0.0, 0.0, 0.0, 0.0)
                    (before if _reads_before(item, body_box) else trailing).append(((box[1], box[0]), text))
        parts = [t for _, t in sorted(before, key=lambda p: p[0])] + ([body] if body else [])
        parts += [t for _, t in after] + [t for _, t in notes]
        return "\n\n".join(parts)

    def render(self, blocks: list) -> str:
        chunks: list[str] = []
        # Indices of the paragraph a ``continues_prev`` block joins. Like MinerU, body text
        # continues across any block on the page; references only across page furniture.
        last_text: int | None = None
        last_ref: int | None = None
        for block in blocks:
            if not isinstance(block, dict):
                continue
            kind = block.get("type", "")
            if kind in _AUXILIARY_TYPES:
                continue
            text = _text(block.get("content"))

            if kind in ("text", "ref_text"):
                target = last_text if kind == "text" else last_ref
                if block.get("continues_prev") and target is not None:
                    chunks[target] = _join_continuation(chunks[target], text)
                elif text:
                    chunks.append(text)
                    if kind == "text":
                        last_text = len(chunks) - 1
                    else:
                        last_ref = len(chunks) - 1
                continue
            if kind != "page_footnote":
                last_ref = None

            if kind in _TITLE_TYPES:
                try:
                    level = min(max(int(block.get("level") or 1), 1), 6)
                except (TypeError, ValueError):
                    level = 1
                rendered = f"{'#' * level} {text}" if text else ""
            elif kind in ("equation", "interline_equation"):
                rendered = f"$$\n{text}\n$$" if text else self._image_md(block.get("image_source"), block)
            elif kind in _VISUAL_TYPES:
                rendered = self._visual(block)
            else:
                # page_footnote, list, index, abstract and unknown text-like blocks
                rendered = text or self._image_md(block.get("image_source"), block)
            if rendered:
                chunks.append(rendered)
        return "\n\n".join(chunks)


# ---------------------------------------------------------------------------
# Helpers for files
# ---------------------------------------------------------------------------

def _safe_extract(zf: zipfile.ZipFile, dest: Path) -> None:
    """Extract a zip, refusing members that would land outside ``dest``."""
    dest = dest.resolve()
    for member in zf.infolist():
        target = (dest / member.filename).resolve()
        if target != dest and dest not in target.parents:
            raise ParserError(f"MinerU zip contains an unsafe path: {member.filename!r}")
    zf.extractall(dest)


def _flat_name(source: str, used: dict[str, str]) -> str:
    """Flat, filesystem-safe filename for an image source, unique within a document."""
    parts = [p for p in Path(source).parts if p not in ("", ".", "..", "/")]
    base = re.sub(r"[^A-Za-z0-9._-]", "_", parts[-1] if parts else "image").lstrip(".") or "image"
    name = base
    if used.get(name, source) != source:
        prefix = re.sub(r"[^A-Za-z0-9._-]", "_", "_".join(parts[:-1])) or "img"
        name, n = f"{prefix}_{base}", 2
        while used.get(name, source) != source:
            name, n = f"{prefix}_{n}_{base}", n + 1
    used[name] = source
    return name


# ---------------------------------------------------------------------------
# Backend
# ---------------------------------------------------------------------------

class MinerUBackend(PdfParserBackend):
    """Optional PDF backend that shells out to ``mineru-kit``.

    Options (all optional): ``tier`` (flash|basic|standard|advanced),
    ``executable`` (path or name of mineru-kit), ``timeout`` (seconds),
    ``extra_env`` (env vars for the MinerU process, e.g. GPU/Vulkan knobs),
    ``keep_raw_output`` (copy MinerU's raw files next to ``images_dir``),
    ``work_dir`` (parent directory for temporary files).
    """

    name = "mineru"

    def __init__(
        self,
        tier: str = DEFAULT_TIER,
        executable: str | None = None,
        timeout: float = DEFAULT_TIMEOUT,
        extra_env: dict[str, Any] | None = None,
        keep_raw_output: bool = False,
        work_dir: str | None = None,
    ):
        tier = (tier or DEFAULT_TIER).strip().lower()
        if tier not in TIERS:
            raise ValueError(f"Unknown MinerU tier '{tier}'. Choose one of: {', '.join(TIERS)}")
        try:
            timeout = float(timeout)
        except (TypeError, ValueError):
            raise ValueError(f"MinerU timeout must be a number of seconds, got {timeout!r}") from None
        if timeout <= 0:
            raise ValueError("MinerU timeout must be positive")
        if extra_env is not None and not isinstance(extra_env, dict):
            raise ValueError("MinerU extra_env must be a mapping of environment variables")
        self.tier = tier
        self.executable = executable
        self.timeout = timeout
        self.extra_env = {str(k): str(v) for k, v in (extra_env or {}).items()}
        self.keep_raw_output = bool(keep_raw_output)
        self.work_dir = work_dir
        self._version: str | None = None

    # -- availability -------------------------------------------------------

    def _wanted_executable(self) -> str:
        return os.path.expanduser(str(self.executable or os.environ.get(ENV_EXECUTABLE)
                                      or os.environ.get(LEGACY_ENV_EXECUTABLE) or DEFAULT_EXECUTABLE))

    def _resolve_executable(self) -> str | None:
        exe = self._wanted_executable()
        if os.sep in exe or (os.altsep and os.altsep in exe):
            return exe if os.path.isfile(exe) and os.access(exe, os.X_OK) else None
        return shutil.which(exe)

    def check_available(self) -> None:
        if self._resolve_executable() is None:
            wanted = self._wanted_executable()
            raise ParserUnavailableError(
                f"MinerU executable not found: {wanted!r}. {INSTALL_HINT}",
                diagnostics={"executable": wanted},
            )

    def version(self) -> str:
        if self._version is None:
            self._version = ""
            exe = self._resolve_executable()
            if exe:
                try:
                    proc = subprocess.run(
                        [exe, "--version"], capture_output=True, text=True,
                        timeout=120, env=self._env(), stdin=subprocess.DEVNULL,
                    )
                    m = re.search(r"MinerU version:\s*([\w.+-]+)", proc.stdout + proc.stderr)
                    self._version = m.group(1) if m else ""
                except Exception as e:  # noqa: BLE001 -- version is informational only
                    logger.debug(f"mineru-kit --version failed: {e}")
        return self._version

    def _env(self) -> dict[str, str]:
        return {**os.environ, **self.extra_env}

    # -- subprocess ---------------------------------------------------------

    def _run(self, cmd: list[str], cwd: Path) -> tuple[int, str, str, float]:
        start = time.monotonic()
        proc = subprocess.Popen(
            cmd, cwd=str(cwd), env=self._env(),
            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            start_new_session=True,  # own process group, so a timeout also kills workers
        )
        try:
            out, err = proc.communicate(timeout=self.timeout)
        except subprocess.TimeoutExpired:
            self._kill_group(proc)
            out, err = proc.communicate()
            raise ParserError(
                f"MinerU timed out after {self.timeout:g}s",
                diagnostics={
                    "command": cmd, "timeout_s": self.timeout,
                    "elapsed_s": round(time.monotonic() - start, 2),
                    "stdout_tail": _tail(out), "stderr_tail": _tail(err),
                },
            ) from None
        except BaseException:
            self._kill_group(proc)
            proc.communicate()
            raise
        return proc.returncode, _tail(out), _tail(err), time.monotonic() - start

    @staticmethod
    def _kill_group(proc: subprocess.Popen) -> None:
        try:
            if hasattr(os, "killpg"):
                os.killpg(proc.pid, signal.SIGKILL)
            else:  # pragma: no cover - non-POSIX
                proc.kill()
        except (ProcessLookupError, PermissionError):
            pass

    # -- conversion ---------------------------------------------------------

    def convert(self, pdf_path: str, images_dir: str) -> BackendOutput:
        self.check_available()
        exe = self._resolve_executable()
        tmp = Path(tempfile.mkdtemp(prefix="polyreader_mineru_", dir=self.work_dir))
        try:
            # Work on a copy with a fixed, dot-free name: mineru-kit names outputs after
            # the input stem, and treats an -o path that has a suffix as an output *file*.
            in_dir, out_dir, raw_dir = tmp / "in", tmp / "out", tmp / "raw"
            in_dir.mkdir()
            out_dir.mkdir()
            pdf_copy = in_dir / "document.pdf"
            shutil.copyfile(pdf_path, pdf_copy)

            cmd = [exe, "parse", str(pdf_copy), "-o", str(out_dir), "-f", "zip", "--tier", self.tier]
            logger.info(f"MinerU ({self.tier}) converting {pdf_path}")
            exit_code, out_tail, err_tail, elapsed = self._run(cmd, tmp)
            diagnostics = {
                "command": cmd, "exit_code": exit_code, "elapsed_s": round(elapsed, 2),
                "stdout_tail": out_tail, "stderr_tail": err_tail, "pdf_path": pdf_path,
            }

            zips = sorted(p for p in out_dir.rglob("*.zip") if p.is_file())
            if not zips:
                raise ParserError(
                    f"MinerU produced no output for {pdf_path} (exit code {exit_code})",
                    diagnostics=diagnostics,
                )
            if exit_code != 0:
                logger.info(f"mineru-kit exited with code {exit_code} after writing its output")
            try:
                with zipfile.ZipFile(zips[0]) as zf:
                    bad = zf.testzip()
                    if bad:
                        raise zipfile.BadZipFile(f"corrupt member {bad}")
                    _safe_extract(zf, raw_dir)
                pages, metadata, extensions, json_name = _load_structure(raw_dir)
                output = self._map_output(pages, metadata, extensions, raw_dir, Path(images_dir), str(pdf_copy))
            except ParserError as e:
                e.diagnostics = {**diagnostics, **e.diagnostics}
                raise
            except (zipfile.BadZipFile, OSError, ValueError) as e:
                raise ParserError(f"MinerU output is unreadable: {e}", diagnostics=diagnostics) from e

            output.info.update(exit_code=exit_code, elapsed_s=round(elapsed, 2), structure_source=json_name)
            if self.keep_raw_output:
                keep = Path(images_dir).parent / "mineru_raw"
                shutil.rmtree(keep, ignore_errors=True)
                shutil.copytree(raw_dir, keep)
                output.info["raw_output_dir"] = str(keep)
            logger.info(
                f"MinerU extracted {len(output.page_texts)} pages, {len(output.images)} images "
                f"from {pdf_path} in {elapsed:.1f}s"
            )
            return output
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def _map_output(
        self,
        pages: list,
        metadata: dict,
        extensions: dict,
        raw_dir: Path,
        images_dir: Path,
        pdf_copy: str,
    ) -> BackendOutput:
        from .pdf_parser import ImageMeta

        warnings: list[str] = []
        meta_count = (metadata.get("document") or {}).get("page_count")
        pdf_count = _pdf_page_count(pdf_copy)
        if pdf_count and meta_count and pdf_count != meta_count:
            warnings.append(f"MinerU reports {meta_count} pages, the PDF has {pdf_count}")

        by_idx: dict[int, list] = {}
        for page in pages:
            try:
                idx = int(page.get("page_idx"))
            except (AttributeError, TypeError, ValueError):
                raise ParserError(f"MinerU page without a valid page_idx: {str(page)[:200]}") from None
            by_idx.setdefault(idx, []).extend(page.get("blocks") or [])
        n_pages = pdf_count or meta_count or (max(by_idx, default=-1) + 1)
        out_of_range = sorted(i for i in by_idx if not 0 <= i < n_pages)
        if out_of_range:
            raise ParserError(
                f"MinerU page indices {out_of_range} fall outside the PDF's {n_pages} pages",
                diagnostics={"pdf_pages": pdf_count, "mineru_page_count": meta_count},
            )

        # Physical page sizes (points) to express image bboxes like other backends.
        sizes = {
            p.get("page_idx"): (p.get("width_pt"), p.get("height_pt"))
            for p in ((extensions.get("docvortex_layout") or {}).get("pages") or [])
            if isinstance(p, dict)
        }

        raw_root = raw_dir.resolve()
        used_names: dict[str, str] = {}
        refs: dict[str, tuple[str, int, dict]] = {}  # flat name -> (source, page_idx, block)
        missing: list[str] = []
        page_idx = 0

        def image_ref(source: str, block: dict) -> str | None:
            src = (raw_dir / source).resolve()
            if raw_root not in src.parents or not src.is_file():
                missing.append(source)
                return None
            name = _flat_name(source, used_names)
            refs.setdefault(name, (source, page_idx, block))
            return name

        renderer = _PageRenderer(image_ref)
        page_texts: list[str] = []
        for page_idx in range(n_pages):
            page_texts.append(renderer.render(by_idx.get(page_idx, [])))

        images = []
        if refs:
            images_dir.mkdir(parents=True, exist_ok=True)
        per_page: dict[int, int] = {}
        for name, (source, idx, block) in refs.items():
            dst = images_dir / name
            shutil.copyfile(raw_dir / source, dst)
            page = idx + 1
            per_page[page] = per_page.get(page, 0) + 1
            width = height = 0
            try:
                from PIL import Image

                with Image.open(dst) as im:
                    width, height = im.size
            except Exception:
                pass
            x = y = bw = bh = 0.0
            w_pt, h_pt = sizes.get(idx, (None, None))
            bbox = block.get("bbox") if isinstance(block, dict) else None
            if w_pt and h_pt and isinstance(bbox, (list, tuple)) and len(bbox) == 4:
                x, y = bbox[0] * w_pt, bbox[1] * h_pt
                bw, bh = (bbox[2] - bbox[0]) * w_pt, (bbox[3] - bbox[1]) * h_pt
            images.append(ImageMeta(
                filename=name, page=page, index=per_page[page],
                width=width, height=height, x=x, y=y, bbox_width=bw, bbox_height=bh,
                size_bytes=dst.stat().st_size, format=dst.suffix.lstrip(".").lower(),
                colorspace="RGB",
            ))
        if missing:
            warnings.append(f"{len(missing)} image reference(s) missing from MinerU output: {missing[:5]}")

        info: dict[str, Any] = {
            "backend": self.name,
            "version": (metadata.get("producer") or {}).get("version") or self.version(),
            "tier": self.tier,
            "pdf_pages": n_pages,
        }
        if warnings:
            info["warnings"] = warnings
            for w in warnings:
                logger.warning(f"MinerU: {w}")
        return BackendOutput(page_texts=page_texts, images=images, info=info)

"""PaperIndex: reasoning-based, vectorless index of a scientific paper.

Public API (re-exported by ``polyreader.indexing`` and ``polyreader``):
    process_document(pdf_path, **kwargs) -> PaperIndex
    print_tree(tree)
    save_index(index, path)
    load_index(path) -> PaperIndex

PaperIndex methods:
    get_page_content(doc, pages) -> dict   # structured page content (1-indexed)
    get_page_image(doc, img_path) -> dict  # retrieve extracted image by path
    get_document_structure(doc) -> list[dict]  # tree without text fields
"""

import asyncio
import json
import logging
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path

from .config import ConfigLoader
from polyreader.llm import LLMClient, create_llm_client
from polyreader.llm.token_counter import count_tokens
from polyreader.parsing.pdf_parser import parse_pdf
from .tree_builder import build_tree
from .node_enrichment import enrich_nodes, generate_doc_description
from .metadata.metadata_extractor import extract_metadata, PaperMetadata
from polyreader.utils.tree_utils import (
    print_tree as _print_tree,
    write_node_id,
    remove_fields,
    add_node_text,
)
from polyreader.utils.image_utils import image_to_base64
from polyreader.utils.json_utils import write_json_atomic

logger = logging.getLogger(__name__)

__all__ = [
    "IndexLoadError",
    "PaperIndex",
    "process_document",
    "print_tree",
    "save_index",
    "load_index",
    "verify_index",
]

# Page requests spanning more pages than this are clamped, not expanded.
_MAX_EXPANDED_PAGES = 10_000

# Extensions served by get_page_image().
_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".tif", ".tiff"}


class IndexLoadError(ValueError):
    """An index file (tree.json / page_contents.json) is missing or corrupted."""


def _parse_page_ranges(pages_str: str) -> list[tuple[int, int]]:
    """Parse a page specification into (start, end) ranges without expanding them.

    Accepts "5", "1-3", "1,3,5", "1-3,7,9-11"; tolerates spaces, en/em dashes
    and reversed ranges ("7-5").
    """
    text = re.sub(r"[\u2010-\u2015\u2212~]", "-", str(pages_str))
    ranges = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part.lstrip("-"):
            start_s, end_s = part.split("-", 1)
            start, end = int(start_s.strip()), int(end_s.strip())
            if start > end:
                start, end = end, start
        else:
            start = end = int(part)
        ranges.append((start, end))
    if not ranges:
        raise ValueError(f"no pages in '{pages_str}'")
    return ranges


def _parse_page_range(pages_str: str) -> list[int]:
    """Parse a page range string into a sorted list of unique page numbers.

    Supports: "5", "1-3", "1,3,5", "1-3,7,9-11".
    """
    pages = set()
    for start, end in _parse_page_ranges(pages_str):
        pages.update(range(start, end + 1))
    return sorted(pages)


def _format_ranges(ranges: list[tuple[int, int]]) -> str:
    return ",".join(f"{s}-{e}" if s != e else str(s) for s, e in ranges)


def _format_page_range(pages: list[int]) -> str:
    """Convert a sorted list of page numbers to a compact range string.

    [1,2,3] -> "1-3", [1,3,5] -> "1,3,5", [1,2,3,7,9,10,11] -> "1-3,7,9-11".
    """
    if not pages:
        return ""
    groups = []
    start = prev = pages[0]
    for p in pages[1:]:
        if p == prev + 1:
            prev = p
        else:
            groups.append((start, prev))
            start = prev = p
    groups.append((start, prev))
    return ",".join(f"{s}-{e}" if s != e else str(s) for s, e in groups)


@dataclass
class PaperIndex:
    """Complete paper index with tree structure, metadata, and parse info."""
    tree: list[dict] = field(default_factory=list)
    metadata: PaperMetadata = field(default_factory=PaperMetadata)
    pdf_path: str = ""
    images_dir: str = ""
    mode_used: str = ""
    doc_name: str = ""
    doc_description: str = ""
    page_texts: list[str] = field(default_factory=list)
    _pages_file: str = ""
    # PDF parser backend name, version and settings used to build the index.
    parser_info: dict = field(default_factory=dict)

    def _lazy_load_pages(self):
        """Load page_contents.json on demand if page_texts is empty.

        Raises:
            IndexLoadError: the page file exists but is corrupted.
        """
        if not self.page_texts and self._pages_file:
            pages_path = Path(self._pages_file)
            if pages_path.exists():
                self.page_texts = _read_page_texts(pages_path)

    def get_page_content(self, doc: str = None, pages=None, doc_name: str = None) -> dict:
        """Retrieve page content from a paper index.

        Args:
            doc: Document label (for multi-doc routing via LLM tool interface).
            pages: Page range — str ("1-3"), int (1), or list[int] ([1,2,3]).
            doc_name: Document name (legacy, defaults to self.doc_name).

        Returns:
            Dict with success, content array, approximate_tokens, metadata, and next_steps.
        """
        doc_name = doc or doc_name or self.doc_name

        try:
            self._lazy_load_pages()
        except IndexLoadError as e:
            return {"success": False, "error": str(e)}
        if not self.page_texts:
            return {"success": False, "error": "No page texts available. Re-process the document to populate page_texts."}

        if pages is None or (isinstance(pages, str) and not pages.strip()):
            return {"success": False, "error": "No pages specified."}

        # Normalise pages to (start, end) ranges — never expanded beyond the
        # document, so a request like "1-100000000" stays cheap.
        try:
            if isinstance(pages, bool):
                raise TypeError("expected a page number or range")
            if isinstance(pages, str):
                ranges = _parse_page_ranges(pages)
            elif isinstance(pages, (int, float)):
                ranges = [(int(pages), int(pages))]
            else:
                ranges = [(int(p), int(p)) for p in pages]
                if not ranges:
                    raise ValueError("empty page list")
        except (ValueError, TypeError) as e:
            return {"success": False, "error": f"Invalid page specification: {e}"}

        total = len(self.page_texts)
        span = sum(end - start + 1 for start, end in ranges)
        if span <= _MAX_EXPANDED_PAGES:
            page_list = sorted({p for start, end in ranges for p in range(start, end + 1)})
            requested_str = _format_page_range(page_list)
            valid_pages = [p for p in page_list if 1 <= p <= total]
            skipped_str = _format_page_range(sorted(set(page_list) - set(valid_pages)))
        else:
            # Absurdly large request: clamp to the document instead of expanding.
            requested_str = _format_ranges(sorted(set(ranges)))
            valid_set: set[int] = set()
            skipped_ranges: list[tuple[int, int]] = []
            for start, end in ranges:
                lo, hi = max(start, 1), min(end, total)
                if lo <= hi:
                    valid_set.update(range(lo, hi + 1))
                if start < 1:
                    skipped_ranges.append((start, min(end, 0)))
                if end > total:
                    skipped_ranges.append((max(start, total + 1), end))
            valid_pages = sorted(valid_set)
            skipped_str = _format_ranges(skipped_ranges)

        if not valid_pages:
            return {
                "success": False,
                "error": f"All requested pages out of range (1-{total}).",
            }

        content = [{"page": p, "text": self.page_texts[p - 1]} for p in valid_pages]
        returned_str = _format_page_range(valid_pages)

        summary = f"Successfully retrieved content for {len(valid_pages)} page(s)."
        if skipped_str:
            summary += f" Pages {skipped_str} out of range (1-{total})."

        combined_text = "\n".join(item["text"] for item in content)

        return {
            "success": True,
            "requested_pages": requested_str,
            "returned_pages": returned_str,
            "total_pages": total,
            "approximate_tokens": count_tokens(combined_text),
            "doc_name": doc_name,
            "content": content,
            "next_steps": {
                "options": [
                    "Use get_document_structure(doc) to review the full document outline",
                    "Use get_page_content(doc, pages) to read additional pages",
                    "Use get_page_image(doc, img_path) to view figures/tables — find img_path values in page text as ![](path)",
                ],
                "summary": summary,
            },
        }

    def get_page_image(self, doc: str = None, img_path: str = None) -> dict:
        """Retrieve a specific extracted image by path.

        Args:
            doc: Document label (for multi-doc routing).
            img_path: Image filename from Marker extraction (e.g. '_page_16_Picture_1.jpeg').
                      Found in page text as ![](img_path).
        """
        if not img_path or not isinstance(img_path, str):
            return {"success": False, "error": "No img_path specified. Look for ![](img_path) references in page text."}

        if not self.images_dir:
            return {"success": False, "error": "No images directory available."}

        full_path = self._resolve_image_path(img_path)
        if full_path is None:
            return {"success": False, "error": f"Image not found: {img_path}"}

        try:
            image_b64 = image_to_base64(str(full_path))
        except OSError as e:
            return {"success": False, "error": f"Image could not be read: {img_path} ({type(e).__name__})"}
        return {
            "success": True,
            "img_path": img_path,
            "image_base64": image_b64,
            "format": full_path.suffix.lstrip("."),
            "next_steps": {
                "options": [
                    "Use get_page_content(doc, pages) to read the text around this image",
                    "Use get_page_image(doc, img_path) to view another extracted image",
                    "Use get_document_structure(doc) to review the full document outline",
                ],
                "summary": f"Retrieved extracted image: {img_path}",
            },
        }

    def _resolve_image_path(self, img_path: str) -> Path | None:
        """Map an LLM-supplied image reference to a file inside images_dir.

        Accepts a bare filename, a path relative to images_dir, or the full
        ``![](name)`` markdown reference. Anything resolving outside
        images_dir (``..``, absolute paths, symlinks) or that is not an image
        file is rejected. Falls back to the bare filename when the reference
        carries an extra directory prefix.
        """
        name = img_path.strip()
        match = re.fullmatch(r"!\[[^\]]*\]\(([^)]*)\)", name)
        if match:
            name = match.group(1).strip()
        name = name.strip("'\"<> ")
        if not name or "\x00" in name:
            return None

        try:
            base = Path(self.images_dir).resolve()
        except (OSError, ValueError):
            return None

        candidates = [name]
        bare = Path(name).name
        if bare and bare != name:
            candidates.append(bare)
        for candidate in candidates:
            try:
                path = (base / candidate).resolve()
            except (OSError, ValueError, RuntimeError):
                continue
            if path == base or not path.is_relative_to(base):
                logger.warning(f"Rejected image path outside the images directory: {img_path!r}")
                continue
            if path.suffix.lower() in _IMAGE_EXTENSIONS and path.is_file():
                return path
        return None

    def get_document_structure(self, doc: str = None) -> list[dict]:
        """Return the tree structure without text fields.

        Args:
            doc: Document label (unused, for consistent tool interface).

        Returns:
            Deep copy of the tree with ``text`` fields stripped,
            keeping titles, summaries, node_ids, and pages.
        """
        return remove_fields(self.tree)


def _read_page_texts(pages_path: Path) -> list[str]:
    """Read page texts from a page_contents.json file."""
    try:
        with open(pages_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return [p["text"] for p in data["page_contents"]]
    except (OSError, ValueError, KeyError, TypeError) as e:
        raise IndexLoadError(
            f"Corrupted page contents file {pages_path} ({type(e).__name__}: {e}). "
            "Re-index the document."
        ) from e


async def process_document(
    pdf_path: str,
    api_key: str = None,
    base_url: str = None,
    model: str = None,
    parse_output_dir: str = None,
    **kwargs,
) -> PaperIndex:
    """Index a scientific paper: parse PDF, build tree, enrich nodes.

    Pipeline: parse → build_tree (with appear_start) → write_node_id →
    enrich_nodes(page_texts) → generate_doc_description → extract_metadata

    Args:
        pdf_path: Path to the PDF file
        api_key: LLM API key (or set OPENROUTER_API_KEY / DEEPSEEK_API_KEY)
        base_url: LLM API base URL
        model: LLM model name
        parse_output_dir: Scratch directory for parser output (images);
            defaults to ``<pdf dir>/<pdf stem>_output``.
        **kwargs: Additional config overrides (see polyreader/indexing/config.yaml),
            e.g. ``pdf_parser_backend="mineru"``, ``mineru={"tier": "basic"}``.

    Returns:
        PaperIndex object with tree, metadata, and doc description
    """
    # Build config
    user_opt = {}
    if model:
        user_opt["model"] = model
    if base_url:
        user_opt["base_url"] = base_url
    user_opt.update(kwargs)

    opt = ConfigLoader().load(user_opt)
    if api_key:
        opt.api_key = api_key
    if not hasattr(opt, "api_key") or not opt.api_key:
        raise ValueError(
            "API key required. Set OPENROUTER_API_KEY (OpenRouter) or "
            "DEEPSEEK_API_KEY (DeepSeek API) env var, or pass api_key parameter."
        )

    # Create LLM client
    llm = create_llm_client(
        api_key=opt.api_key,
        model=opt.model,
        base_url=opt.base_url,
        temperature=opt.temperature,
        timeout=getattr(opt, "request_timeout", None),
        max_concurrent=opt.max_concurrent_api_calls,
        include_extra_body=getattr(opt, "include_extra_body", True),
        reasoning_effort=getattr(opt, "reasoning_effort", None),
    )

    # Step 1: Parse PDF
    backend = getattr(opt, "pdf_parser_backend", "marker") or "marker"
    backend_options = {}
    if backend == "mineru":
        backend_options = {k: v for k, v in (getattr(opt, "mineru", None) or {}).items() if v is not None}
    print(f"Parsing PDF ({backend}): {pdf_path}")
    parse_result = await parse_pdf(
        pdf_path, output_dir=parse_output_dir, backend=backend, **backend_options,
    )
    print(f"  Pages: {len(parse_result.page_texts)}")
    print(f"  Images: {len(parse_result.images)}")

    # Step 2: Build tree (NO text on nodes)
    print("Building section tree...")
    tree = await build_tree(
        parse_result.page_texts,
        llm,
        max_token_num=opt.max_token_num_each_node,
        max_page_num=opt.max_page_num_each_node,
    )

    # Step 3: Add node IDs
    if opt.if_add_node_id == "yes":
        write_node_id(tree)

    # Step 4: Generate summaries (pass page_texts directly, no add_node_text needed)
    if opt.if_add_node_summary == "yes":
        print("Enriching nodes...")
        await enrich_nodes(
            tree,
            llm,
            page_texts=parse_result.page_texts,
            summary_token_threshold=opt.summary_token_threshold,
            max_concurrent=opt.max_concurrent_api_calls,
        )

    # Step 5: Add node text only if explicitly requested (rare)
    if opt.if_add_node_text == "yes":
        add_node_text(tree, parse_result.page_texts)

    # Step 7: Generate doc description
    doc_description = ""
    if opt.if_add_doc_description == "yes":
        print("Generating document description...")
        doc_description = await generate_doc_description(tree, llm)

    # Step 8: Extract metadata
    print("Extracting metadata...")
    metadata = await extract_metadata(parse_result.page_texts, llm)

    doc_name = Path(pdf_path).name
    print(f"Done! Paper: {doc_name}")

    return PaperIndex(
        tree=tree,
        metadata=metadata,
        pdf_path=str(Path(pdf_path).resolve()),
        images_dir=parse_result.images_dir or "",
        mode_used=parse_result.mode_used,
        doc_name=doc_name,
        doc_description=doc_description,
        page_texts=parse_result.page_texts,
        parser_info=dict(parse_result.parser_info or {"backend": parse_result.mode_used}),
    )


def print_tree(index_or_tree):
    """Print the tree structure."""
    if isinstance(index_or_tree, PaperIndex):
        _print_tree(index_or_tree.tree)
    else:
        _print_tree(index_or_tree)


def _move_images(src: Path, dst: Path) -> None:
    """Move an image directory to ``dst``, replacing it without data loss.

    The new images are moved next to ``dst`` first; the old directory is only
    removed once the move succeeded.
    """
    staging = dst.with_name(dst.name + ".incoming")
    previous = dst.with_name(dst.name + ".previous")
    for leftover in (staging, previous):
        if leftover.exists():
            shutil.rmtree(leftover)
    shutil.move(str(src), str(staging))
    # Two renames instead of delete-then-rename: dst is never missing for longer
    # than a rename, and the old images are deleted only after the swap.
    if dst.exists():
        dst.rename(previous)
    staging.rename(dst)
    if previous.exists():
        shutil.rmtree(previous, ignore_errors=True)


def save_index(index: PaperIndex, path: str):
    """Save index as two files: tree.json + page_contents.json, plus images.

    Given output path like ``output/paper.json``, produces:
    - ``output/paper_tree.json`` — lightweight tree + metadata
    - ``output/paper_page_contents.json`` — parsed page texts
    - ``output/images_paper/`` — extracted images (moved from the parser output)

    Files are written atomically; page contents are written before the tree,
    so an existing ``*_tree.json`` always has its page file complete.
    """
    p = Path(path)
    stem = p.stem  # e.g. "paper" from "paper.json"
    out_dir = p.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    tree_path = out_dir / f"{stem}_tree.json"
    pages_path = out_dir / f"{stem}_page_contents.json"
    images_dest = out_dir / f"images_{stem}"

    # Move images from the parser's temp output to the save directory
    if index.images_dir and Path(index.images_dir).is_dir():
        src = Path(index.images_dir).resolve()
        dst = images_dest.resolve()
        if src != dst:
            _move_images(src, dst)
            # Clean up empty parent dir left behind by the temp extraction
            src_parent = src.parent
            if src_parent.exists() and not any(src_parent.iterdir()):
                src_parent.rmdir()
            index.images_dir = str(dst)
        # else: already in the right place, nothing to do

    # Serialize metadata to dict
    meta = index.metadata
    metadata_dict = {
        "doi": meta.doi or "",
        "title": meta.title or "",
        "authors": meta.authors or [],
        "journal": meta.journal or "",
        "year": meta.year,
        "volume": meta.volume or "",
        "pages": meta.pages or "",
    }

    # Write page_contents.json — parsed page texts
    pages_data = {
        "total_pages": len(index.page_texts),
        "page_contents": [
            {"page": i + 1, "text": text}
            for i, text in enumerate(index.page_texts)
        ],
    }
    write_json_atomic(pages_path, pages_data)
    print(f"Page contents saved to {pages_path}")

    # Write tree.json — lightweight retrieval index
    tree_data = {
        "version": 2,
        "doc_name": index.doc_name,
        "doc_description": index.doc_description,
        "pdf_path": index.pdf_path,
        "images_dir": f"images_{stem}/",
        "total_pages": len(index.page_texts),
        "pages_file": f"{stem}_page_contents.json",
        "metadata": metadata_dict,
        "tree": index.tree,
        "next_steps": {
            "options": [
                "Use get_page_content(doc, pages) to read specific pages identified by start_index/end_index in the tree",
                "Use get_page_image(doc, img_path) to view figures/tables — find img_path in page text as ![](path)",
                "Use tree node summaries and prefix_summary to reason about which sections are relevant",
            ],
            "summary": "Document structure retrieved successfully.",
        },
    }
    parser_info = index.parser_info or ({"backend": index.mode_used} if index.mode_used else {})
    if parser_info:
        tree_data["parser"] = parser_info
    write_json_atomic(tree_path, tree_data)
    print(f"Tree saved to {tree_path}")


def _read_tree_file(tree_path: Path) -> dict:
    if not tree_path.is_file():
        raise FileNotFoundError(f"Index file not found: {tree_path}")
    try:
        with open(tree_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError) as e:
        raise IndexLoadError(
            f"Corrupted index file {tree_path} ({type(e).__name__}: {e}). Re-index the document."
        ) from e
    if not isinstance(data, dict):
        raise IndexLoadError(f"Corrupted index file {tree_path}: expected a JSON object.")
    return data


def load_index(path: str, pdf_path: str = None) -> PaperIndex:
    """Load index from a tree.json file (version 2 format).

    Page texts are NOT loaded immediately — they are lazily loaded from
    the companion page_contents.json on first access via get_page_content()
    or query wrappers.

    Args:
        path: Path to the *_tree.json file.
        pdf_path: Optional override for the stored pdf_path (useful if the
            PDF has been moved since indexing).

    Raises:
        FileNotFoundError: the tree file does not exist.
        IndexLoadError: the tree file is not valid JSON / not an index.
        ValueError: unsupported index version.
    """
    tree_path = Path(path)
    data = _read_tree_file(tree_path)

    if data.get("version") != 2:
        raise ValueError(
            f"Unsupported index format (version={data.get('version')}). "
            "Only version 2 (split tree/page_contents) is supported. "
            "Please re-index the document."
        )

    # Resolve pages_file path relative to the tree.json directory
    pages_file = data.get("pages_file", "")
    if pages_file:
        pages_file = str(tree_path.parent / pages_file)

    # Resolve images_dir relative to tree.json directory
    images_dir = data.get("images_dir", "")
    if images_dir:
        images_dir = str(tree_path.parent / images_dir)

    # Reconstruct metadata from stored dict (backward-compatible with old files)
    meta_dict = data.get("metadata", {})
    if meta_dict:
        metadata = PaperMetadata(
            doi=meta_dict.get("doi", ""),
            title=meta_dict.get("title", ""),
            authors=meta_dict.get("authors", []),
            journal=meta_dict.get("journal", ""),
            year=meta_dict.get("year"),
            volume=meta_dict.get("volume", ""),
            pages=meta_dict.get("pages", ""),
        )
    else:
        metadata = PaperMetadata()

    parser_info = data.get("parser") if isinstance(data.get("parser"), dict) else {}

    return PaperIndex(
        tree=data.get("tree", []),
        metadata=metadata,
        pdf_path=pdf_path or data.get("pdf_path", ""),
        images_dir=images_dir,
        mode_used=parser_info.get("backend", ""),
        doc_name=data.get("doc_name", ""),
        doc_description=data.get("doc_description", ""),
        page_texts=[],  # lazy — loaded on demand
        _pages_file=pages_file,
        parser_info=parser_info,
    )


def verify_index(path: str) -> list[str]:
    """Check that a saved index is complete and readable.

    Returns a list of problems (empty when the index can be used as-is):
    missing/corrupted tree file, unsupported version, missing/corrupted page
    file or a page count that disagrees with the tree.
    """
    tree_path = Path(path)
    try:
        data = _read_tree_file(tree_path)
    except (FileNotFoundError, IndexLoadError) as e:
        return [str(e)]
    problems = []
    if data.get("version") != 2:
        problems.append(f"unsupported index version {data.get('version')!r}")
    if not data.get("tree"):
        problems.append("index has an empty section tree")
    pages_file = data.get("pages_file")
    if not pages_file:
        problems.append("index has no pages_file entry")
        return problems
    pages_path = tree_path.parent / pages_file
    if not pages_path.is_file():
        problems.append(f"page contents file missing: {pages_path}")
        return problems
    try:
        texts = _read_page_texts(pages_path)
    except IndexLoadError as e:
        problems.append(str(e))
        return problems
    expected = data.get("total_pages")
    if isinstance(expected, int) and expected != len(texts):
        problems.append(
            f"page count mismatch: tree says {expected}, page file has {len(texts)}"
        )
    return problems

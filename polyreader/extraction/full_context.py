"""Full-context extraction lane: read the paper once, extract many samples.

For papers that fit comfortably in the model's context window (the common
case: a research article plus SI is 10-40k tokens), navigating pages per
sample with tools re-reads the same pages many times. This lane instead

1. packs all page texts of the document set into one deterministic prefix
   (identical for every call of a paper, so the provider's prompt cache
   serves it after the first call),
2. runs entity discovery as a single call that also returns evidence
   pointers (document, pages, table/figure) for every sample,
3. extracts the samples of an entity in small batches, in parallel,
4. post-processes every sample exactly like the per-sample loop; samples a
   batch misses or gets wrong fall back to a single-sample retry or to the
   existing tool-based extraction.

The prompt content (triage gate, ontology, schema, validation rules) is the
one used by the tool-based lane; only how the paper is delivered differs.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from pydantic import ValidationError

from polyreader.llm.client import LLMClient
from polyreader.llm.token_counter import count_tokens
from polyreader.utils.json_utils import extract_json

from .config import ExtractionConfig
from .document_set import DocumentSet, UnknownDocumentError
from .entity_discovery import (
    DISCOVERY_PROMPT_CORE,
    ENTITY_DISCOVERY_TASK,
    _build_retry_message,
    parse_entity_answer,
    validate_entity_map,
)
from .extraction_loop import EntityExtractionOutcome, _finalize_sample, postprocess_raw_sample
from .prompts import _guided_prompt_core
from .tools import _image_data_url, supports_vision

logger = logging.getLogger(__name__)

MODES = ("agentic", "full_context", "auto")
MAX_IMAGES_PER_BATCH = 6


# ---------------------------------------------------------------------------
# Packing and mode selection
# ---------------------------------------------------------------------------

CONTEXT_HEADER = """\
You are given the complete text of a scientific paper{si_note}, converted to \
Markdown page by page. Every page starts with a marker <<doc="LABEL" page=N>>. \
Use these document labels and page numbers for provenance: page_numbers are the \
page numbers N of the pages the data comes from, and source_document is the \
document label{primary_note}. Images appear as ![](file) next to their \
captions; tables are given as Markdown or HTML.
"""


def pack_document_set(doc_set: DocumentSet) -> str:
    """Render all documents of the set as one deterministic text block."""
    doc_set.ensure_pages_loaded()
    multi = doc_set.has_multiple_documents()
    primary_label = doc_set.labels()[0]
    parts = [CONTEXT_HEADER.format(
        si_note=" together with its supplementary documents" if multi else "",
        primary_note=f' (leave it empty for the main document "{primary_label}")',
    )]
    for label, index in doc_set:
        pages = index.page_texts or []
        meta = index.metadata
        title = getattr(meta, "title", "") or ""
        parts.append(
            f'\n=== Document "{label}": {index.doc_name} ({len(pages)} pages)'
            + (f" — {title}" if title and label == primary_label else "")
            + " ==="
        )
        for i, text in enumerate(pages, start=1):
            parts.append(f'<<doc="{label}" page={i}>>\n{text.strip()}\n')
    return "\n".join(parts)


def choose_extraction_mode(config: ExtractionConfig, packed: str | None) -> tuple[str, int | None]:
    """Resolve ``config.extraction_mode`` to "agentic" or "full_context"."""
    mode = (config.extraction_mode or "agentic").lower()
    if mode not in MODES:
        raise ValueError(f"extraction_mode must be one of {MODES}, got {config.extraction_mode!r}")
    if mode == "agentic":
        return "agentic", None
    tokens = count_tokens(packed or "")
    if mode == "auto" and tokens > config.full_context_budget:
        logger.info(
            f"  [full_context] paper is {tokens} tokens > budget {config.full_context_budget}: "
            "using tool-based extraction"
        )
        return "agentic", tokens
    return "full_context", tokens


# ---------------------------------------------------------------------------
# Discovery with evidence pointers
# ---------------------------------------------------------------------------

EVIDENCE_SECTION = """\
## 4a. Evidence Pointers

The complete paper is provided above, so read it directly — no tools are \
available or needed. In addition to the fields of section 4, give every entity \
an "evidence" object that maps each of its sample_ids to where that sample's \
data is reported:

```json
"evidence": {
    "<sample_id>": [
        {"doc": "<document label>", "pages": [<page numbers>],
         "kind": "table | text | figure | scheme",
         "ref": "<e.g. Table 1, row S3 / Figure 4b / Section 2.3>",
         "images": ["<image file names from ![](...) needed to read figure-only values or structures>"]}
    ]
}
```

List every location (characterization tables, synthesis procedure, figures whose \
values are not given in text). Only list images when the value or structure can \
only be read from the image.

Sample IDs must be the labels the paper itself uses (e.g. "S3", "Run 2", "Entry 5", \
"PS40-b-PI132-b-PS23"); never invent abbreviations. When the same label appears in \
different tables, qualify it with the table, e.g. "Table 2, Run 1".
"""


def build_full_context_discovery_prompt(doc_set: DocumentSet) -> str:
    index = doc_set.primary
    authors = ", ".join(index.metadata.authors) if index.metadata.authors else "Unknown"
    docs = "\n".join(
        f'  "{label}": {idx.doc_name} ({len(idx.page_texts or [])} pages)' for label, idx in doc_set
    )
    return (
        DISCOVERY_PROMPT_CORE
        + EVIDENCE_SECTION
        + f"\n## 6. Paper Metadata\n\nPaper: {index.doc_name}\nAuthors: {authors}\n"
        + f"Documents (page numbers are per document):\n{docs}\n\n"
        + "## 7. Task\n\n"
        + ENTITY_DISCOVERY_TASK
        + "\n\nScan every document provided above. Return only the JSON object."
    )


def _normalize_evidence(entity_map: dict[str, Any]) -> None:
    """Coerce each entity's evidence to {sample_id: [pointer, ...]}."""
    for info in entity_map.get("entities", {}).values():
        if not isinstance(info, dict):
            continue
        raw = info.get("evidence")
        evidence: dict[str, list[dict]] = {}
        if isinstance(raw, dict):
            for sid, pointers in raw.items():
                if isinstance(pointers, dict):
                    pointers = [pointers]
                if isinstance(pointers, list):
                    evidence[str(sid)] = [p for p in pointers if isinstance(p, dict)]
        info["evidence"] = evidence


async def discover_entities_full_context(
    doc_set: DocumentSet,
    llm: LLMClient,
    paper_context: str,
    max_retries: int = 2,
) -> dict[str, Any]:
    """Entity discovery from the packed paper (no tools), with evidence pointers."""
    prompt = build_full_context_discovery_prompt(doc_set)
    error_history: list[dict] = []
    entity_map: dict[str, Any] = {"entities": {}, "sample_list": []}
    unparseable = False
    for attempt in range(max_retries):
        messages = [
            {"role": "system", "content": paper_context},
            {"role": "user", "content": prompt},
        ]
        if error_history:
            messages.append({"role": "user", "content": _build_retry_message(error_history)})
        answer = await llm.chat(messages, json_mode=True)
        parsed = extract_json(answer, quiet=True)
        if not isinstance(parsed, dict) or not parsed:
            unparseable = True
            error_history.append({"attempt": attempt + 1, "errors": ["Answer was not a JSON object"]})
            continue
        unparseable = False
        entity_map = parse_entity_answer(answer)
        _normalize_evidence(entity_map)
        is_valid, issues = validate_entity_map(entity_map)
        if is_valid:
            logger.info(
                f"  [full_context] discovery succeeded (attempt {attempt + 1}): "
                f"{len(entity_map.get('entities', {}))} entities"
            )
            return entity_map
        # Same policy as tool-based discovery: retry with the issues, then
        # proceed with what we have.
        logger.warning(f"  [full_context] discovery issues (attempt {attempt + 1}): {issues}")
        error_history.append({"attempt": attempt + 1, "errors": issues})
    if unparseable:
        raise ValueError("Full-context discovery did not return a JSON entity map")
    return entity_map


# ---------------------------------------------------------------------------
# Batch extraction
# ---------------------------------------------------------------------------

BATCH_EXTRACTION_TASK = """\
Extract ALL data for each of these samples of polymer entity "{entity_id}": {targets}.

Each sample is a distinct polymer batch. Extract data for every listed sample, \
and ONLY for the listed samples.

Follow this 5-step workflow for each sample:

## Step 1: SCAN & TRIAGE
- Confirm {entity_id} is synthesized in this paper (not commercial/cited)
- Identify its full name, topology, and constituent monomers with their SMILES
- If {entity_id} is a blend, identify its base polymer components

## Step 2: MASTER TABLE
Find ALL characterization data for each sample across all tables and documents.
Molecular weight (Mn, Mw, PDI) → put in `molecular_weight` field, NOT `properties`.
Thermal, mechanical, structural, rheological, and other values → put in `properties` array.
Check every table — data is often split across multiple tables.
Convert kDa to g/mol by multiplying by 1000.
Match data to each sample by its ID label. Never copy values between samples
— each sample is a different polymer batch with its own characterization.
Put EVERY other value reported for a sample into its `properties` array: every
table column (e.g. Mn by NMR, theoretical Mn, conversion, yield, composition)
and every value stated in the text or readable from a figure. Each sample
object must be as complete as if it were the only sample requested — do not
shorten later objects or leave `properties` empty when values are reported.

## Step 3: SYNTHESIS
Find polymerization conditions: method, catalyst, initiator, co-catalyst,
solvent, temperature, time, monomer feed ratio, atmosphere, etc.
These may be shared across multiple samples of this entity type.

## Step 4: PROCESSING HISTORY
Trace post-synthesis steps: precipitation, drying, annealing, film casting,
compression molding, electrospinning, etc. Include conditions (solvent, temp, time).

## Step 5: SELF-CHECK
Apply these 4 validation gates to every sample before outputting:
(a) Dispersity: Mw ≥ Mn, PDI ≈ Mw/Mn ±5%
(b) Thermal hierarchy: Tg < Tm < Td (when all present)
(c) Missing PDI: calculate if Mn and Mw available, set pdi_derived=true
(d) Entity triage: verify synthesized AND characterized in this paper

Return a JSON object {{"samples": [...]}} with exactly one object per listed \
sample, in the listed order. Each object follows the schema in section 4 and has \
"sample_id" set to the listed ID. Use null for missing values. If a listed sample \
was NOT synthesized and characterized in this paper, return \
{{"sample_id": "<id>", "skipped": true, "reason": "explanation"}} for it."""


def _targeting_section(entity_info: dict[str, Any], targets: list[str]) -> str:
    siblings = [sid for sid in entity_info.get("sample_ids", []) if sid not in targets]
    return f"""
## 2a. Sample Targeting

You are extracting data for these samples: {", ".join(f"**{t}**" for t in targets)}

This entity contains distinct polymer batches, each with its own
characterization data. Synthesis conditions may be shared across samples —
characterization data (molecular weight, thermal, mechanical properties) is
unique per sample: match values to each sample by its label.

Other samples of this entity (do NOT extract): {", ".join(siblings) if siblings else "(none)"}
"""


def _evidence_section(evidence: dict[str, list[dict]], targets: list[str]) -> str:
    lines = []
    for t in targets:
        for p in evidence.get(t, []):
            pages = p.get("pages")
            pages_str = ",".join(str(x) for x in pages) if isinstance(pages, list) else str(pages or "")
            lines.append(
                f'- {t}: doc="{p.get("doc", "")}" pages={pages_str} '
                f'{p.get("kind", "")} {p.get("ref", "")}'.rstrip()
            )
    if not lines:
        return ""
    return (
        "\n## 6. Where the Data Is\n\nLocations found during entity discovery "
        "(check the whole paper anyway — this list may be incomplete):\n" + "\n".join(lines) + "\n"
    )


def build_batch_prompt(
    entity_id: str,
    entity_info: dict[str, Any],
    entity_map: dict[str, Any],
    targets: list[str],
    base_samples_context: dict[str, Any] | None = None,
    error_history: list[dict[str, Any]] | None = None,
) -> str:
    info_for_prompt = {k: v for k, v in entity_info.items() if k != "evidence"}
    map_for_prompt = {
        k: ({kk: vv for kk, vv in v.items() if kk != "evidence"} if isinstance(v, dict) else v)
        for k, v in entity_map.items()
    }
    core = _guided_prompt_core(
        entity_id, info_for_prompt, map_for_prompt, base_samples_context,
        error_history, _targeting_section(entity_info, targets),
    )
    evidence = _evidence_section(entity_info.get("evidence") or {}, targets)
    task = BATCH_EXTRACTION_TASK.format(entity_id=entity_id, targets=", ".join(f'"{t}"' for t in targets))
    return core + evidence + "\n## 7. Task\n\n" + task


def _norm(text: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(text or "").lower())


def _assign_items(items: list[dict], targets: list[str]) -> dict[str, dict]:
    """Map returned sample objects to target IDs (exact ID, then order)."""
    assigned: dict[str, dict] = {}
    remaining = list(items)
    for t in targets:
        for item in remaining:
            if _norm(item.get("sample_id")) == _norm(t):
                assigned[t] = item
                remaining.remove(item)
                break
    unassigned = [t for t in targets if t not in assigned]
    if unassigned and len(unassigned) == len(remaining):
        # Same count left over: the model renamed the IDs; keep the order.
        for t, item in zip(unassigned, remaining):
            assigned[t] = item
    return assigned


def _parse_items(answer: str) -> list[dict]:
    parsed = extract_json(answer, quiet=True)
    if isinstance(parsed, dict):
        items = parsed.get("samples")
        if items is None and parsed.get("sample_id") is not None:
            items = [parsed]
    elif isinstance(parsed, list):
        items = parsed
    else:
        items = None
    return [i for i in (items or []) if isinstance(i, dict)]


_IMAGE_REF = re.compile(r"!\[[^\]]*\]\(([^)\s]+)\)")
_CAPTION_START = re.compile(r"^\W*(supplementary\s+)?(figure|fig\.|scheme|chart)\s*s?\d", re.IGNORECASE)
# Figures whose captions name a characterization method often carry values
# (e.g. Tg labels on DSC curves) that the text does not repeat.
_CHARACTERIZATION = re.compile(
    r"\b(DSC|TGA|DMA|GPC|SEC|MALDI|SAXS|WAXS|XRD|thermogram|chromatogra|stress|strain|tensile|"
    r"rheolog|glass transition|melting|T_?g\b|T_?m\b|T_?d\b)", re.IGNORECASE)


_VALUE_PLOTS = re.compile(
    r"\b(DSC|TGA|DMA|thermogram|stress|strain|tensile|rheolog|SAXS|WAXS|XRD|"
    r"glass transition|melting)", re.IGNORECASE)


@dataclass
class Figure:
    doc: str
    page: int
    image: str
    caption: str


def figure_inventory(doc_set: DocumentSet) -> list[Figure]:
    """Images with the caption text that follows (or precedes) them on the page."""
    doc_set.ensure_pages_loaded()
    figures = []
    for label, index in doc_set:
        for page_no, text in enumerate(index.page_texts or [], start=1):
            lines = (text or "").splitlines()
            for i, line in enumerate(lines):
                for name in _IMAGE_REF.findall(line):
                    figures.append(Figure(label, page_no, name, _caption_near(lines, i)[:500]))
    return figures


def _caption_near(lines: list[str], i: int) -> str:
    """Caption after the image (Marker/MinerU order), else just before it;
    the search never crosses another image reference."""
    for line in lines[i + 1:i + 6]:
        if _IMAGE_REF.search(line):
            break
        if _CAPTION_START.match(line.strip()):
            return line.strip()
    for line in reversed(lines[max(0, i - 3):i]):
        if _IMAGE_REF.search(line):
            break
        if _CAPTION_START.match(line.strip()):
            return line.strip()
    return ""


def _relevant_figures(figures: list[Figure], entity_id: str, entity_info: dict[str, Any],
                      targets: list[str]) -> list[Figure]:
    """Characterization figures whose caption mentions the entity, a target or its table."""
    names = {entity_id, entity_info.get("full_name", ""), *entity_info.get("aliases", []), *targets}
    for pointers in (entity_info.get("evidence") or {}).values():
        for p in pointers:
            ref = str(p.get("ref", ""))
            m = re.search(r"(supplementary\s+)?table\s*s?\d+", ref, re.IGNORECASE)
            if m:
                names.add(m.group(0))
    names = {n.strip().lower() for n in names if isinstance(n, str) and len(n.strip()) >= 2}
    target_names = {t.strip().lower() for t in targets if len(t.strip()) >= 2}
    ranked = []
    for order, f in enumerate(figures):
        caption = f.caption.lower()
        if not (f.caption and _CHARACTERIZATION.search(f.caption) and any(n in caption for n in names)):
            continue
        # Thermal/mechanical plots carry values the text often omits; molar-mass
        # traces and spectra rarely do. Captions naming a target come first.
        value_plot = bool(_VALUE_PLOTS.search(f.caption))
        names_target = any(t in caption for t in target_names)
        ranked.append((not value_plot, not names_target, order, f))
    return [f for *_, f in sorted(ranked, key=lambda r: r[:3])]


def _collect_images(doc_set: DocumentSet, evidence: dict[str, list[dict]],
                    targets: list[str], extra: list[Figure] = ()) -> list[tuple[str, str]]:
    """(name, data URL) of evidence images plus relevant characterization
    figures, resolved inside each document's images dir."""
    wanted: list[tuple[str | None, str]] = []
    for t in targets:
        for p in evidence.get(t, []):
            for name in p.get("images") or []:
                if isinstance(name, str):
                    wanted.append((p.get("doc"), name))
    wanted.extend((f.doc, f.image) for f in extra)

    seen, images = set(), []
    for doc_label, name in wanted:
        if name in seen or len(images) >= MAX_IMAGES_PER_BATCH:
            continue
        seen.add(name)
        try:
            index = doc_set.resolve(doc_label, strict=doc_set.has_multiple_documents())
        except UnknownDocumentError:
            index = doc_set.primary
        result = index.get_page_image(img_path=name)
        if result.get("success"):
            images.append((name, _image_data_url(result["image_base64"], result.get("format"))))
    return images


@dataclass
class BatchStats:
    batches: int = 0
    batch_samples: int = 0
    single_retries: int = 0
    agentic_fallbacks: int = 0
    failed_batches: int = 0
    reasons: list[str] = field(default_factory=list)

    def as_trace(self) -> dict:
        return {
            "batches": self.batches,
            "batch_samples": self.batch_samples,
            "single_sample_retries": self.single_retries,
            "agentic_fallbacks": self.agentic_fallbacks,
            "failed_batches": self.failed_batches,
            "fallback_reasons": self.reasons[:50],
        }


Result = tuple[Any, EntityExtractionOutcome, Any, list[str]]


async def _call_batch(llm: LLMClient, paper_context: str, prompt: str,
                      images: list[tuple[str, str]]) -> str:
    if images:
        content: Any = [{"type": "text", "text": prompt}]
        for name, url in images:
            content.append({"type": "text", "text": f"Image {name}:"})
            content.append({"type": "image_url", "image_url": {"url": url}})
    else:
        content = prompt
    messages = [{"role": "system", "content": paper_context}, {"role": "user", "content": content}]
    return await llm.chat(messages, json_mode=True)


async def extract_batch(
    tasks: list,
    entity_map: dict[str, Any],
    doc_set: DocumentSet,
    llm: LLMClient,
    config: ExtractionConfig,
    paper_context: str,
    base_ctx: dict[str, Any] | None,
    semaphore,
    agentic_fallback: Callable[[Any, dict | None], Awaitable[Result]],
    stats: BatchStats,
) -> list[Result]:
    """Extract one batch of tasks (same entity) from the packed paper."""
    entity_id = tasks[0].entity_id
    entity_info = tasks[0].entity_info
    targets = [t.target_sample_id or t.entity_id for t in tasks]
    evidence = entity_info.get("evidence") or {}
    images = []
    if supports_vision(config):
        figures = _relevant_figures(figure_inventory(doc_set), entity_id, entity_info, targets)
        images = _collect_images(doc_set, evidence, targets, figures)
    prompt = build_batch_prompt(entity_id, entity_info, entity_map, targets, base_ctx)
    stats.batches += 1
    stats.batch_samples += len(tasks)

    try:
        async with semaphore:
            answer = await _call_batch(llm, paper_context, prompt, images)
        returned = _parse_items(answer)
        items = _assign_items(returned, targets)
        logger.info(
            f"  [full_context] batch {entity_id} {targets}: {len(returned)} objects returned, "
            f"{len(items)} matched" + (f", {len(images)} images attached" if images else "")
        )
    except Exception as e:
        logger.warning(f"  [full_context] batch {targets} failed ({type(e).__name__}: {e}); "
                       "falling back to tool-based extraction")
        stats.failed_batches += 1
        items = {}

    async def finish(task, target) -> Result:
        item = items.get(target)
        if item is None:
            stats.agentic_fallbacks += 1
            stats.reasons.append(f"{target}: missing from batch output")
            logger.warning(f"  [full_context] {target}: missing from batch output; tool-based extraction")
            return await agentic_fallback(task, base_ctx)
        if item.get("skipped"):
            logger.info(f"  [full_context] {target} skipped: {item.get('reason', '')}")
            return (task, EntityExtractionOutcome.SKIPPED, None, [])
        try:
            sample, flags, critical = postprocess_raw_sample(item)
        except ValidationError as e:
            stats.agentic_fallbacks += 1
            stats.reasons.append(f"{target}: schema validation failed ({e.error_count()} errors)")
            logger.warning(f"  [full_context] {target}: schema validation failed; tool-based extraction")
            return await agentic_fallback(task, base_ctx)
        if not sample.sample_id:
            sample.sample_id = target if task.target_sample_id else ""
        if critical:
            # One cheap retry on the cached paper, with the issue as feedback.
            stats.single_retries += 1
            retried = await _retry_single(
                task, target, entity_map, llm, paper_context, base_ctx, images, semaphore,
                [{"attempt": 1, "errors": [f"Critical validation issue: {c.message}" for c in critical]}],
            )
            if retried is not None:
                sample, flags = retried
        _finalize_sample(sample, flags, task.entity_info)
        return (task, EntityExtractionOutcome.SUCCESS, sample, flags)

    return list(await asyncio.gather(*(finish(t, tg) for t, tg in zip(tasks, targets))))


async def _retry_single(task, target, entity_map, llm, paper_context, base_ctx, images,
                        semaphore, error_history):
    """Single-sample full-context retry; returns (sample, flags) if it is clean."""
    prompt = build_batch_prompt(task.entity_id, task.entity_info, entity_map, [target],
                                base_ctx, error_history)
    try:
        async with semaphore:
            answer = await _call_batch(llm, paper_context, prompt, images)
        item = _assign_items(_parse_items(answer), [target]).get(target)
        if not item or item.get("skipped"):
            return None
        sample, flags, critical = postprocess_raw_sample(item)
    except Exception as e:
        logger.warning(f"  [full_context] retry of {target} failed: {type(e).__name__}: {e}")
        return None
    if critical:
        return None  # keep the first (flagged) result, as the per-sample loop does
    if not sample.sample_id:
        sample.sample_id = target if task.target_sample_id else ""
    return sample, flags


def group_tasks(tasks: list, batch_size: int) -> list[list]:
    """Group tasks by entity (keeping order) and split into batches."""
    by_entity: dict[str, list] = {}
    for t in tasks:
        by_entity.setdefault(t.entity_id, []).append(t)
    batches = []
    size = max(1, batch_size)
    for group in by_entity.values():
        for i in range(0, len(group), size):
            batches.append(group[i:i + size])
    return batches


def describe_entity_map(entity_map: dict[str, Any]) -> str:
    """Compact JSON of the entity map without evidence (for logging)."""
    return json.dumps({k: v.get("sample_ids", []) for k, v in entity_map.items() if isinstance(v, dict)})

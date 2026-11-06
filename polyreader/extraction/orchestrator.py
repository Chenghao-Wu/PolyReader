"""Main extraction pipeline orchestrator.

Two-stage architecture: entity discovery followed by parallel per-entity guided
extraction with built-in validation loops. No reconciliation agent needed.
"""

from __future__ import annotations

import asyncio
import dataclasses
import logging
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from polyreader.indexing import PaperIndex
from polyreader.llm import LLMClient, create_llm_client
from polyreader.utils.json_utils import write_json_atomic

from .config import ExtractionConfig
from .document_set import DocumentSet
from .extraction_loop import (
    EntityExtractionOutcome,
    extract_single_entity,
)
from polyreader.schema.polymer import (
    ExtractionResult,
    PaperMetadataRef,
    PolymerSample,
    TIER_GOLD, TIER_SILVER, TIER_BRONZE, TIER_REJECT,
    TYPE_BASE, TYPE_DERIVED_BLEND, TYPE_COMPOSITE,
)
from .entity_discovery import discover_entities
from .full_context import (
    BatchStats,
    choose_extraction_mode,
    discover_entities_full_context,
    extract_batch,
    group_tasks,
    pack_document_set,
)
from ._transport import is_transport_error

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Extraction task (sample-level granularity)
# ---------------------------------------------------------------------------


@dataclass
class ExtractionTask:
    """A single extraction task, potentially targeting one sample within a multi-sample entity."""

    entity_id: str
    entity_info: dict
    target_sample_id: str | None = None  # None for single-sample entities
    is_multi_sample: bool = False  # True when entity was expanded


def _expand_entity_tasks(
    entity_ids: list[str],
    entity_map: dict[str, Any],
) -> list[ExtractionTask]:
    """Expand entities with multiple sample_ids into per-sample extraction tasks.

    Single-sample entities produce one task with target_sample_id=None (original behavior).
    Multi-sample entities produce N tasks, one per sample_id.
    """
    tasks = []
    for eid in entity_ids:
        if eid not in entity_map:
            continue
        info = entity_map[eid]
        sample_ids = info.get("sample_ids", [])

        if len(sample_ids) <= 1:
            tasks.append(ExtractionTask(
                entity_id=eid,
                entity_info=info,
                target_sample_id=sample_ids[0] if sample_ids else None,
                is_multi_sample=False,
            ))
        else:
            for sid in sample_ids:
                tasks.append(ExtractionTask(
                    entity_id=eid,
                    entity_info=info,
                    target_sample_id=sid,
                    is_multi_sample=True,
                ))
    return tasks


# ---------------------------------------------------------------------------
# Entity type -> topology mapping
# ---------------------------------------------------------------------------

_ENTITY_TYPE_TO_TOPOLOGY: dict[str, str] = {
    "homopolymer": "homopolymer",
    "linear": "linear",
    "star": "star",
    "branched": "branched",
    "cyclic": "cyclic",
    "network": "network",
    "blend": "blend",
    "composite": "composite",
    "block": "block",
    "random": "random",
    "graft": "graft",
    "dendrimer": "dendrimer",
    "alternating": "alternating",
    "gradient": "gradient",
    "copolymer": "random",  # composition type, not architecture
}


# ---------------------------------------------------------------------------
# Utility functions (kept from old orchestrator for backward compatibility)
# ---------------------------------------------------------------------------

def _enrich_from_entity_map(
    samples: list[dict[str, Any]],
    entity_map: dict[str, Any],
) -> None:
    """Fill missing topology, constituents, and abbreviation from entity_map.

    This is the sole enrichment point for entity-map data.
    Mutates samples in-place. Never overwrites existing data.
    """
    for sample in samples:
        polymer_id = sample.get("polymer_id", "")
        sample_id = sample.get("sample_id", "")

        # Find matching entity
        matched_info: dict | None = None
        matched_abbrev: str = ""
        for abbrev, info in entity_map.items():
            if not isinstance(info, dict):
                continue
            if (polymer_id and polymer_id == info.get("full_name", "")) or \
               (polymer_id and polymer_id == abbrev) or \
               (sample_id and sample_id in info.get("sample_ids", [])):
                matched_info = info
                matched_abbrev = abbrev
                break

        if matched_info is None:
            continue

        # Fill topology if missing
        if not sample.get("topology"):
            raw_type = matched_info.get("type", "")
            if raw_type:
                sample["topology"] = _ENTITY_TYPE_TO_TOPOLOGY.get(
                    raw_type.lower(), raw_type
                )

        # Fill abbreviation if missing
        if not sample.get("abbreviation") and matched_abbrev:
            sample["abbreviation"] = matched_abbrev

        # Fill constituents if missing or unnamed
        constituents = sample.get("constituents", [])
        has_named = any(c.get("name") for c in constituents) if constituents else False
        if not has_named and matched_info.get("full_name"):
            sample["constituents"] = [
                {"name": matched_info["full_name"], "role": "polymer"}
            ]


def unique_name(base: str, used: set[str]) -> str:
    """Return ``base``, or ``base_2``, ``base_3``, ... if already in ``used``.

    Adds the result to ``used``. Keeps distinct samples whose IDs sanitize to
    the same file name from overwriting each other.
    """
    name, n = base, 1
    while name.lower() in used:
        n += 1
        name = f"{base}_{n}"
    used.add(name.lower())
    return name


def save_samples(
    samples: list[PolymerSample],
    output_dir: str,
    doc_name: str,
) -> Path:
    """Save sample JSON files into a 'samples' subfolder of output_dir.

    Files are written atomically.
    """
    samples_dir = Path(output_dir) / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)
    used: set[str] = set()
    for sample in samples:
        safe_id = sample.sample_id or sample.abbreviation or "unknown"
        safe_id = "".join(c if c.isalnum() or c in "-_" else "_" for c in safe_id)
        filepath = samples_dir / f"{unique_name(f'{doc_name}_{safe_id}', used)}.json"
        write_json_atomic(filepath, sample.clean_dump())
        logger.info(f"  Saved sample JSON: {filepath}")
    return samples_dir


def _cross_entity_reconcile(
    samples: list[PolymerSample],
    entity_map: dict[str, Any],
) -> list[PolymerSample]:
    """Deterministic cross-entity reconciliation.

    With per-entity extraction, reconciliation is trivial:
    1. Deduplicate by sample_id (prefer the one whose abbreviation matches entity)
    2. Resolve blend component references (ensure polymer_ref points to existing sample)

    No LLM calls needed.
    """
    if not samples:
        return samples

    # Deduplicate by sample_id
    seen: dict[str, PolymerSample] = {}
    for sample in samples:
        sid = sample.sample_id
        if not sid:
            # Keep samples without sample_id (they're unique)
            seen[f"_no_id_{id(sample)}"] = sample
            continue

        if sid in seen:
            # Prefer the one whose abbreviation matches the entity
            existing = seen[sid]
            existing_abbrev = existing.abbreviation
            new_abbrev = sample.abbreviation

            # Check which one matches an entity abbreviation
            if new_abbrev in entity_map and existing_abbrev not in entity_map:
                seen[sid] = sample
            # Otherwise keep existing
        else:
            seen[sid] = sample

    deduped = list(seen.values())

    # Resolve blend component references
    sample_ids = {s.sample_id for s in deduped if s.sample_id}
    abbreviations = {s.abbreviation for s in deduped if s.abbreviation}

    for sample in deduped:
        for bc in sample.blend_components:
            ref = bc.polymer_ref
            if ref and ref not in sample_ids and ref not in abbreviations:
                # Reference doesn't exist — this is a soft warning
                logger.warning(
                    f"  Sample {sample.sample_id}: blend component reference "
                    f"'{ref}' not found in extracted samples"
                )

    return deduped


# ---------------------------------------------------------------------------
# Main extraction pipeline
# ---------------------------------------------------------------------------

async def extract(
    index: PaperIndex | DocumentSet | list[PaperIndex],
    config: ExtractionConfig | None = None,
) -> ExtractionResult:
    """Run the per-entity guided extraction pipeline.

    Steps:
    1. Entity discovery (identify all polymers and sample IDs)
    2. Wave 1: Extract base entities in parallel
    3. Wave 2: Extract derived entities with base context
    4. Cross-entity reconciliation (deterministic, no LLM)
    5. Save individual JSONs + combined result

    Args:
        index: A loaded PaperIndex, a DocumentSet, or a list of PaperIndex objects.
               Lists are auto-wrapped in DocumentSet.from_indices().
        config: Optional configuration. Uses defaults if not provided.

    Returns:
        ExtractionResult with all extracted PolymerSample records.
    """
    if config is None:
        config = ExtractionConfig()
    else:
        # Work on a copy: per-run adjustments must not leak into the caller's config.
        config = dataclasses.replace(config)

    # Normalize input to DocumentSet
    if isinstance(index, DocumentSet):
        doc_set = index
    elif isinstance(index, list):
        doc_set = DocumentSet.from_indices(*index)
    else:
        doc_set = DocumentSet.from_index(index)

    # Log document set summary
    logger.info(
        f"  [orchestrator] Document set: {len(doc_set)} doc(s), "
        f"labels={doc_set.labels()}"
    )
    for label, idx in doc_set:
        page_count = len(idx.page_texts) if idx.page_texts else 0
        logger.info(f"  [orchestrator]   '{label}': {idx.doc_name} ({page_count} pages)")

    if not config.api_key:
        raise ValueError(
            "No API key configured. Set OPENROUTER_API_KEY (OpenRouter) or "
            "DEEPSEEK_API_KEY (DeepSeek API, base_url https://api.deepseek.com)."
        )

    if not doc_set.primary.tree:
        raise ValueError("PaperIndex has no tree structure. Run indexing first.")
    if not doc_set.primary.page_texts:
        logger.warning("PaperIndex has no page_texts — keyword_search will not work")

    # Auto-increase iteration limits for multi-document sets
    if doc_set.has_multiple_documents():
        # Entity discovery only scans primary doc — no extra iterations needed
        config.guided_max_iterations = max(config.guided_max_iterations, 16)
        logger.info(
            f"  [orchestrator] Multi-doc: guided iteration limit → "
            f"{config.guided_max_iterations}"
        )

    llm = create_llm_client(
        api_key=config.api_key,
        model=config.model,
        base_url=config.base_url,
        temperature=config.temperature,
        thinking_enabled=config.thinking_enabled,
        include_extra_body=config.include_extra_body,
        max_concurrent=config.max_concurrent_entities,
        timeout=config.request_timeout,
        reasoning_effort=config.reasoning_effort,
        thinking_disabled=config.thinking_disabled,
    )

    extraction_trace = []

    # Extraction mode: tool-based per-sample navigation, or the whole paper
    # in context with batched extraction (see full_context.py).
    paper_context = None
    if (config.extraction_mode or "agentic").lower() != "agentic":
        paper_context = pack_document_set(doc_set)
    mode, packed_tokens = choose_extraction_mode(config, paper_context)
    fc_stats = BatchStats()
    logger.info(
        f"  [orchestrator] extraction mode: {mode}"
        + (f" (packed paper: {packed_tokens} tokens)" if packed_tokens is not None else "")
    )

    # Step 0: Entity discovery
    logger.info("Step 0: Running entity discovery")
    discovery_mode = mode
    if mode == "full_context":
        try:
            entity_data = await discover_entities_full_context(doc_set, llm, paper_context)
        except Exception as e:
            if is_transport_error(e):
                raise
            logger.warning(
                f"  Full-context discovery failed ({type(e).__name__}: {e}); "
                "using tool-based discovery"
            )
            discovery_mode = "agentic"
            entity_data = await discover_entities(doc_set, llm, config)
    else:
        entity_data = await discover_entities(doc_set, llm, config)
    entity_map = entity_data.get("entities", {})
    base_entity_ids = entity_data.get("base_entities", list(entity_map.keys()))
    derived_entity_ids = entity_data.get("derived_entities", [])

    extraction_trace.append({
        "agent": "entity_discovery",
        "entities_found": len(entity_map),
        "base_entities": len(base_entity_ids),
        "derived_entities": len(derived_entity_ids),
        "mode": discovery_mode,
    })
    logger.info(
        f"  Found {len(entity_map)} entities "
        f"({len(base_entity_ids)} base, {len(derived_entity_ids)} derived)"
    )

    # Build paper metadata reference from primary document
    _pm = doc_set.primary.metadata
    paper_meta = PaperMetadataRef(
        doi=getattr(_pm, "doi", "") or "",
        title=getattr(_pm, "title", "") or "",
        authors=getattr(_pm, "authors", []) or [],
        journal=getattr(_pm, "journal", "") or "",
        year=getattr(_pm, "year", None),
        volume=getattr(_pm, "volume", "") or "",
        pages=getattr(_pm, "pages", "") or "",
    )

    if not entity_map:
        logger.warning("No entities found — returning empty result")
        return ExtractionResult(
            metadata=paper_meta,
            samples=[],
            entity_map=entity_map,
            extraction_trace=extraction_trace,
            sample_counts={"total": 0},
        )

    # Step 1: Extract base entities (parallel with semaphore)
    logger.info("Step 1: Extracting base entities in parallel")
    semaphore = asyncio.Semaphore(config.max_concurrent_entities)

    # Expand base entities into per-sample tasks
    base_tasks_expanded = _expand_entity_tasks(base_entity_ids, entity_map)
    if config.max_samples > 0 and len(base_tasks_expanded) > config.max_samples:
        logger.info(
            f"  max_samples={config.max_samples}: truncating "
            f"{len(base_tasks_expanded)} base tasks to {config.max_samples}"
        )
        base_tasks_expanded = base_tasks_expanded[:config.max_samples]
    logger.info(
        f"  Expanded {len(base_entity_ids)} base entities into "
        f"{len(base_tasks_expanded)} extraction tasks"
    )

    async def extract_one(
        task: ExtractionTask,
        base_ctx: dict | None = None,
    ) -> tuple[ExtractionTask, EntityExtractionOutcome, PolymerSample | None, list[str]]:
        async with semaphore:
            try:
                outcome, sample, flags = await extract_single_entity(
                    entity_id=task.entity_id,
                    entity_info=task.entity_info,
                    entity_map=entity_map,
                    doc_set=doc_set,
                    llm=llm,
                    config=config,
                    base_samples_context=base_ctx,
                    target_sample_id=task.target_sample_id,
                    is_multi_sample=task.is_multi_sample,
                )
            except Exception as e:
                # Isolate the failure: other samples keep going.
                logger.error(
                    f"  Extraction of '{task.target_sample_id or task.entity_id}' "
                    f"raised {type(e).__name__}: {e}"
                )
                return task, EntityExtractionOutcome.FATAL, None, [f"{type(e).__name__}: {e}"]
            return task, outcome, sample, flags

    async def run_wave(items: list[tuple[ExtractionTask, dict | None]]) -> list:
        """Run (task, base_ctx) pairs; results in the per-task tuple format."""
        if mode != "full_context":
            return await asyncio.gather(
                *(extract_one(task, ctx) for task, ctx in items), return_exceptions=True,
            )
        ctx_by_entity: dict[str, dict | None] = {}
        for task, ctx in items:
            ctx_by_entity.setdefault(task.entity_id, ctx)
        batches = group_tasks([task for task, _ in items], config.batch_size)
        nested = await asyncio.gather(*(
            extract_batch(batch, entity_map, doc_set, llm, config, paper_context,
                          ctx_by_entity[batch[0].entity_id], semaphore, extract_one, fc_stats)
            for batch in batches
        ), return_exceptions=True)
        out: list = []
        for batch, res in zip(batches, nested):
            if isinstance(res, BaseException):
                # Unexpected failure of the batch machinery: per-sample fallback.
                logger.error(f"  Full-context batch failed unexpectedly: {res!r}")
                fc_stats.agentic_fallbacks += len(batch)
                ctx = ctx_by_entity[batch[0].entity_id]
                out.extend(await asyncio.gather(
                    *(extract_one(task, ctx) for task in batch), return_exceptions=True,
                ))
            else:
                out.extend(res)
        return out

    # Wave 1: Base entities (expanded to per-sample tasks)
    base_results = await run_wave([(task, None) for task in base_tasks_expanded])

    # Collect successful base samples for blend context
    base_samples: dict[str, PolymerSample] = {}
    base_outcomes: dict[str, EntityExtractionOutcome] = {}

    for result in base_results:
        if isinstance(result, BaseException):
            logger.error(f"  Base entity extraction failed with exception: {result!r}")
            continue

        task, outcome, sample, flags = result
        task_label = task.target_sample_id or task.entity_id
        base_outcomes[task_label] = outcome

        if outcome == EntityExtractionOutcome.SUCCESS and sample:
            base_samples[task_label] = sample
            logger.info(f"  ✓ {task_label}: SUCCESS (quality: {sample.quality_tier})")
        elif outcome == EntityExtractionOutcome.SKIPPED:
            logger.info(f"  ⊘ {task_label}: SKIPPED (not synthesized in paper)")
        else:
            logger.warning(f"  ✗ {task_label}: {outcome.value}")

    extraction_trace.append({
        "wave": "base_entities",
        "total": len(base_entity_ids),
        "expanded_tasks": len(base_tasks_expanded),
        "success": len(base_samples),
        "outcomes": {eid: oc.value for eid, oc in base_outcomes.items()},
    })

    # Aggregate entity-level outcomes for multi-sample base entities.
    # depends_on lists use entity abbreviations, but multi-sample tasks store
    # outcomes under sample IDs. Bridge the gap so dependency checks work.
    for eid in base_entity_ids:
        if eid in base_outcomes:
            continue  # Single-sample entity already stored under entity_id
        # Check if any sample-level task for this entity succeeded
        entity_sample_ids = entity_map.get(eid, {}).get("sample_ids", [])
        for sid in entity_sample_ids:
            if base_outcomes.get(sid) == EntityExtractionOutcome.SUCCESS:
                base_outcomes[eid] = EntityExtractionOutcome.SUCCESS
                if sid in base_samples and eid not in base_samples:
                    base_samples[eid] = base_samples[sid]
                break
        else:
            # No sample succeeded — pick the worst outcome for the entity
            for sid in entity_sample_ids:
                if sid in base_outcomes:
                    base_outcomes[eid] = base_outcomes[sid]
                    break

    # Step 2: Extract derived entities with base context (parallel)
    logger.info("Step 2: Extracting derived entities with base context")

    # Skip derived extraction when max_samples already met
    if config.max_samples > 0 and len(base_samples) >= config.max_samples:
        logger.info(
            f"  max_samples={config.max_samples} reached with base samples — "
            f"skipping derived entity extraction"
        )
        derived_entity_ids = []

    # Expand derived entities into per-sample tasks
    derived_tasks_expanded = _expand_entity_tasks(derived_entity_ids, entity_map)
    logger.info(
        f"  Expanded {len(derived_entity_ids)} derived entities into "
        f"{len(derived_tasks_expanded)} extraction tasks"
    )

    derived_items: list[tuple[ExtractionTask, dict | None]] = []
    for task in derived_tasks_expanded:
        entity_info = task.entity_info
        deps = entity_info.get("depends_on", [])

        # Check if ALL dependencies failed (check both entity-level and sample-level keys)
        all_deps_failed = all(
            dep not in base_outcomes
            or base_outcomes[dep] != EntityExtractionOutcome.SUCCESS
            for dep in deps
        )

        if all_deps_failed and deps:
            task_label = task.target_sample_id or task.entity_id
            logger.warning(
                f"  ⊘ {task_label}: SKIPPED (all base dependencies failed: {deps})"
            )
            continue

        # Build base context from successful dependencies
        base_ctx = {
            ref: base_samples[ref].clean_dump()
            for ref in deps
            if ref in base_samples
        }

        derived_items.append((task, base_ctx))

    derived_results = await run_wave(derived_items)

    # Collect all successful samples
    all_samples = list(base_samples.values())
    derived_outcomes: dict[str, EntityExtractionOutcome] = {}

    for result in derived_results:
        if isinstance(result, BaseException):
            logger.error(f"  Derived entity extraction failed with exception: {result!r}")
            continue

        task, outcome, sample, flags = result
        task_label = task.target_sample_id or task.entity_id
        derived_outcomes[task_label] = outcome

        if outcome == EntityExtractionOutcome.SUCCESS and sample:
            all_samples.append(sample)
            logger.info(f"  ✓ {task_label}: SUCCESS (quality: {sample.quality_tier})")
        elif outcome == EntityExtractionOutcome.SKIPPED:
            logger.info(f"  ⊘ {task_label}: SKIPPED (not synthesized in paper)")
        else:
            logger.warning(f"  ✗ {task_label}: {outcome.value}")

    extraction_trace.append({
        "wave": "derived_entities",
        "total": len(derived_entity_ids),
        "expanded_tasks": len(derived_tasks_expanded),
        "attempted": len(derived_items),
        "success": len(all_samples) - len(base_samples),
        "outcomes": {eid: oc.value for eid, oc in derived_outcomes.items()},
    })

    # Step 3: Cross-entity reconciliation (deterministic, no LLM)
    logger.info("Step 3: Cross-entity reconciliation")
    reconciled_samples = _cross_entity_reconcile(all_samples, entity_map)
    logger.info(f"  Reconciled {len(all_samples)} → {len(reconciled_samples)} samples")

    # Stamp paper metadata onto all samples
    for sample in reconciled_samples:
        sample.metadata = paper_meta

    # Step 4: Save individual JSONs
    if config.output_dir:
        logger.info("Step 4: Saving sample JSONs")
        save_samples(reconciled_samples, config.output_dir, doc_set.primary.doc_name)

    # Build sample counts
    tier_counts = Counter(s.quality_tier for s in reconciled_samples)
    type_counts = Counter(s.sample_type for s in reconciled_samples)
    sample_counts = {
        "total": len(reconciled_samples),
        TYPE_BASE: type_counts.get(TYPE_BASE, 0),
        TYPE_DERIVED_BLEND: type_counts.get(TYPE_DERIVED_BLEND, 0),
        TYPE_COMPOSITE: type_counts.get(TYPE_COMPOSITE, 0),
        TIER_GOLD: tier_counts.get(TIER_GOLD, 0),
        TIER_SILVER: tier_counts.get(TIER_SILVER, 0),
        TIER_BRONZE: tier_counts.get(TIER_BRONZE, 0),
        TIER_REJECT: tier_counts.get(TIER_REJECT, 0),
    }

    if mode == "full_context":
        extraction_trace.append({
            "agent": "full_context", "packed_tokens": packed_tokens,
            "batch_size": config.batch_size, **fc_stats.as_trace(),
        })
    extraction_trace.append({"agent": "llm_usage", "model": llm.model, **llm.usage})

    result = ExtractionResult(
        metadata=paper_meta,
        samples=reconciled_samples,
        entity_map=entity_map,
        extraction_trace=extraction_trace,
        sample_counts=sample_counts,
    )

    logger.info(f"Done! Extracted {len(reconciled_samples)} samples (counts: {sample_counts})")
    return result

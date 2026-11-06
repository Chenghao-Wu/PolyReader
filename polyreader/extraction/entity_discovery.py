"""Standalone entity discovery module — guided_navigation style.

Identifies all polymer entities in a paper and builds an entity map with
triage gates, base/derived classification, and validation+retry. No class
inheritance — self-contained functions following guided_navigation.py pattern.
"""

from __future__ import annotations

import asyncio
import logging
import random
from typing import Any

from polyreader.llm.client import LLMClient
from polyreader.utils.json_utils import extract_json

from .config import ExtractionConfig
from .document_set import DocumentSet
from .prompts import TOPOLOGY_VOCABULARY
from .tools import (
    build_navigation_tools, dispatch_tool, format_doc_structure_json,
    parse_tool_arguments, summarize_tool_calls, supports_vision,
    tool_error_result, wrap_up_conversation,
)
from ._transport import is_transport_error

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Task prompt (4-step workflow)
# ---------------------------------------------------------------------------

ENTITY_DISCOVERY_TASK = """\
Scan this paper to identify ALL polymer entities that pass the entity triage gate.

Follow this 4-step workflow:

## Step 1: SCAN ABSTRACT & INTRODUCTION
Read the Abstract and Introduction sections.
- Identify polymer names, abbreviations, and the paper's main focus
- Note which polymers appear to be synthesized vs. cited from literature

## Step 2: SCAN EXPERIMENTAL / METHODS
Read the Experimental section.
- Confirm which polymers have synthesis procedures described
- Note catalysts, initiators, monomers (these are NOT entities — they go in synthesis fields)
- Identify sample IDs and naming conventions

## Step 3: SCAN TABLES & RESULTS
Check characterization tables in the Results section.
- Confirm which polymers have numerical data (Mn, Mw, PDI, Tg, Tm, etc.)
- Collect all sample IDs from table rows and columns
- Note any blend or composite samples that reference other polymers
- For each entity, enumerate ALL distinct samples from characterization tables.
  A "distinct sample" is a unique polymer batch — identified by having its own
  sample label/ID AND its own characterization data (molecular weight, composition,
  or synthesis conditions that differ from other batches).
- IMPORTANT: Different property measurements on the SAME polymer batch are NOT
  different samples. A single sample can have Tg, Tm, Td, modulus, elongation,
  and other properties all belonging to it. These are all extracted together as
  one record. Only count a new sample when the material itself is different
  (different batch, different molecular weight, different composition, or different
  processing conditions).
- Check ALL characterization data including text and tables in this paper.

## Step 4: BUILD ENTITY MAP
For each candidate polymer:
1. Apply the triage gate (section 2): Does it have BOTH synthesis AND characterization?
2. Classify type using the topology vocabulary (section 3)
3. Set extraction_class: "base" or "derived"
4. For derived entities, fill depends_on with base entity abbreviations
5. Collect all sample_ids referencing this entity
6. Provide a triage_reason explaining why it was included

Output your final JSON following the schema in section 4. Use null for missing values. \
Do NOT include entities that fail the triage gate."""

# ---------------------------------------------------------------------------
# System prompt builder
# ---------------------------------------------------------------------------


# Sections 1-5 of the discovery prompt (role, triage gate, topology, JSON
# schema, validation) — shared by tool-based and full-context discovery.
DISCOVERY_PROMPT_CORE = f"""## 1. Role

You are a **polymer entity identification specialist**. Your task is to scan a \
scientific paper and build a complete entity map: every polymer synthesized and \
characterized in this paper, with abbreviations, topology, sample IDs, and \
extraction ordering.

## 2. Entity Triage Gate

Before including ANY polymer in your output, it MUST pass this gate:

**EXTRACT** (include in entity_map):
- Polymer was synthesized in this paper (synthesis method described in Experimental)
- Polymer has tabulated characterization data (Mn/Mw/PDI in a table, or explicit \
numerical values in text)
- Blend/composite samples made from base polymers synthesized in this paper

**DO NOT EXTRACT** (exclude entirely):
- Commercial polymers used only as reference/control \
(e.g., "commercial PLLA from NatureWorks")
- Polymers cited from other papers in Introduction/literature review — no new data
- Reagents, catalysts, solvents (Sn(Oct)2, toluene, methanol) — these go in \
synthesis fields of the polymers they help create, not as separate entities
- Monomers/precursors (L-lactide, glycolide, e-caprolactone) — these go in \
constituent fields, not as entities
- Polymers with data ONLY in figures (GPC traces, DSC curves) and no numerical \
values in text or tables
- Polymers listed without synthesis context ("Polymer X: Tg = 100C" with no \
method described)

**When in doubt**: if you cannot point to BOTH a synthesis procedure AND a \
numerical characterization value in THIS paper, do NOT include the entity.

## 3. Domain Knowledge — Topology Vocabulary

Use exactly these values for the "type" field:

{TOPOLOGY_VOCABULARY}

### CRITICAL — Blend vs. Copolymer Detection

This is the most common AI misclassification. Apply these rules strictly:

- **COPOLYMER**: A single polymer chain with >=2 monomer types synthesized together. \
Notation: poly(A-co-B), poly(A-b-B), PLGA = poly(lactic-co-glycolic acid).
- **BLEND**: A physical mixture of >=2 separately synthesized polymers. \
Keywords: "blend", "mixture", "composite", "stereocomplex", "mixed".
- **Stereocomplexes** (e.g., PLLA + PDLA) are BLENDS, not copolymers.
- If the paper mixes two homopolymers, each homopolymer is a separate base entity \
and the mixture is a derived "blend" entity.

## 4. JSON Output Schema

Return a JSON object with this structure:
```json
{{
    "entities": {{
        "<abbreviation_or_name>": {{
            "full_name": "<full IUPAC or common polymer name>",
            "type": "<topology from section 3 vocabulary>",
            "extraction_class": "<base | derived>",
            "depends_on": ["<abbreviations of base entities, only for derived>"],
            "aliases": ["<alternative names or abbreviations>"],
            "sample_ids": ["<COMPLETE list of all distinct sample IDs for this entity. Each sample_id represents a unique polymer batch with its own characterization. Samples that differ in molecular weight, composition, or synthesis conditions are distinct. Different property measurements on the same batch are NOT separate samples — they all belong to one record.>"],
            "triage_reason": "<why this entity passed the triage gate>"
        }}
    }},
    "sample_list": ["<all sample IDs found across all entities>"],
    "paper_polymer_focus": "<main polymer type or topic of the paper>",
    "base_entities": ["<abbreviations of base entities, extracted first>"],
    "derived_entities": ["<abbreviations of derived entities (blends/composites), extracted second>"]
}}
```

**Field definitions:**
- **extraction_class**: "base" for independently synthesized polymers; "derived" for \
blends, composites, or samples that depend on other entities in the paper.
- **depends_on**: Only for derived entities — list the abbreviations of the base \
polymers they are made from. Empty list for base entities.
- **triage_reason**: Brief explanation of why this entity was included \
(e.g., "synthesized via ROP, Mn/Mw in Table 1").
- **base_entities** / **derived_entities**: Ordered lists that determine extraction \
wave order. Base entities are extracted first; derived entities second with base \
results as context.

## 5. Validation Rules

Before outputting, verify:
1. Every entity in "entities" passes the section 2 triage gate.
2. Every "type" value matches the section 3 vocabulary exactly.
3. Every "derived" entity has non-empty "depends_on" referencing existing base entities.
4. "base_entities" + "derived_entities" together contain all keys from "entities".
5. "sample_list" contains all sample_ids from all entities (no duplicates, no omissions).
6. No reagents, catalysts, solvents, or monomers are listed as entities.
7. Sample completeness: the total sample_ids across all entities should
   approximately match the number of distinct polymer batches characterized
   in the paper. Under-counting means missing samples; over-counting means
   splitting one batch's different property measurements into separate samples.

"""


def build_entity_discovery_prompt(doc_set: DocumentSet) -> str:
    """Build self-contained 7-section system prompt for entity discovery.

    Follows guided_navigation.py:build_system_prompt() pattern — all sections
    in one f-string with embedded paper metadata and document structure.
    """
    index = doc_set.primary
    structure_json = format_doc_structure_json(doc_set)

    return DISCOVERY_PROMPT_CORE + f"""## 6. Paper Metadata & Document Structure

Paper: {index.doc_name}
Authors: {', '.join(index.metadata.authors) if index.metadata.authors else 'Unknown'}
Total pages: {len(index.page_texts) if index.page_texts else 'unknown'}
{"" if not doc_set.has_multiple_documents() else chr(10) + "Available documents (page numbers are per-document):" + chr(10) + chr(10).join(f'  "{label}": {idx.doc_name} ({len(idx.page_texts) if idx.page_texts else 0} pages)' for label, idx in doc_set) + chr(10)}
Document structure:
{structure_json}

## 7. Execution Strategy

You have 3 tools: get_page_content (PRIMARY), get_page_image (retrieves extracted \
images — when vision is enabled the image will be delivered for visual inspection), \
get_document_structure (already provided above).
{"" if not doc_set.has_multiple_documents() else chr(10) + "You have " + str(len(doc_set)) + " documents available. Scan ALL of them for relevant data." + chr(10) + "Characterization tables, additional spectra, and detailed procedures" + chr(10) + "often appear in supplementary documents." + chr(10) + 'Use the "doc" parameter to specify which document to read from, e.g. get_page_content(doc="<label>", pages="1-3").' + chr(10)}
Follow this {"5" if doc_set.has_multiple_documents() else "4"}-step reading order:
1. Abstract — identify polymer names, abbreviations, paper focus
2. Experimental / Methods — confirm synthesis procedures, note sample IDs
3. Results & Tables — confirm characterization data, collect all sample IDs
{"4. Scan remaining documents for additional tables, characterization data" + chr(10) + "5." if doc_set.has_multiple_documents() else "4."} Build the entity map JSON following the schema in section 4

Use get_page_content systematically, reading 2-3 pages at a time to stay within \
token limits. Use get_page_image when you see ![](img_path) references in page text \
and need to visually inspect the figure or diagram.

When you have all the data you need, provide your final answer as a message \
with NO tool calls."""




# ---------------------------------------------------------------------------
# Answer parsing
# ---------------------------------------------------------------------------


def parse_entity_answer(answer: str) -> dict:
    """Parse entity map from the LLM's final answer.

    Returns dict with keys: entities, sample_list, paper_polymer_focus,
    base_entities, derived_entities. Backfills missing fields with defaults.
    """
    parsed = extract_json(answer)
    if not isinstance(parsed, dict):
        return {"entities": {}, "sample_list": [], "raw": answer}

    parsed["entities"] = _normalize_entities(parsed.get("entities"))
    entities = parsed["entities"]
    parsed["sample_list"] = _str_list(parsed.get("sample_list"))
    parsed.setdefault("paper_polymer_focus", "")
    for key in ("base_entities", "derived_entities"):
        if key in parsed:
            parsed[key] = _str_list(parsed[key])

    # Derive base_entities / derived_entities if LLM didn't provide them
    if "base_entities" not in parsed or "derived_entities" not in parsed:
        base = []
        derived = []
        for abbrev, info in entities.items():
            if not isinstance(info, dict):
                continue
            if info.get("extraction_class") == "derived":
                derived.append(abbrev)
            else:
                base.append(abbrev)
        parsed.setdefault("base_entities", base)
        parsed.setdefault("derived_entities", derived)

    # Backfill per-entity defaults
    for abbrev, info in entities.items():
        if not isinstance(info, dict):
            continue
        if "extraction_class" not in info:
            if abbrev in parsed.get("derived_entities", []):
                info["extraction_class"] = "derived"
            else:
                info["extraction_class"] = "base"
        info["depends_on"] = _str_list(info.get("depends_on"))
        info["aliases"] = _str_list(info.get("aliases"))
        info["sample_ids"] = _str_list(info.get("sample_ids"))
        info.setdefault("triage_reason", "")
        info.setdefault("polymer_class", "")

    # Entities missing from both wave lists would never be extracted.
    listed = set(parsed.get("base_entities", [])) | set(parsed.get("derived_entities", []))
    for abbrev, info in entities.items():
        if abbrev in listed or not isinstance(info, dict):
            continue
        wave = "derived_entities" if info.get("extraction_class") == "derived" else "base_entities"
        parsed.setdefault(wave, []).append(abbrev)
        logger.info(f"  [entity_discovery] '{abbrev}' was missing from the wave lists; added to {wave}")

    return parsed


def _str_list(value) -> list[str]:
    """Coerce an LLM-provided list field to a de-duplicated list of strings.

    A bare string becomes a one-item list (never iterated per character);
    numbers become strings; null/blank items are dropped.
    """
    if value is None:
        return []
    if isinstance(value, (str, int, float)) and not isinstance(value, bool):
        value = [value]
    if not isinstance(value, (list, tuple)):
        return []
    out: list[str] = []
    for item in value:
        if item is None or isinstance(item, (dict, list, bool)):
            continue
        text = str(item).strip()
        if text and text not in out:
            out.append(text)
    return out


def _normalize_entities(entities) -> dict[str, Any]:
    """Coerce the ``entities`` field to ``{abbreviation: info_dict}``.

    Accepts the expected mapping, or a list of entity dicts carrying their
    name under abbreviation/name/id/full_name.
    """
    if isinstance(entities, dict):
        return {str(k).strip(): v for k, v in entities.items() if str(k).strip()}
    if isinstance(entities, list):
        out: dict[str, Any] = {}
        for item in entities:
            if not isinstance(item, dict):
                continue
            key = next(
                (str(item[k]).strip() for k in ("abbreviation", "name", "id", "full_name")
                 if item.get(k) and str(item[k]).strip()),
                "",
            )
            if key and key not in out:
                out[key] = item
        return out
    return {}


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_entity_map(entity_map: dict[str, Any]) -> tuple[bool, list[str]]:
    """Validate entity map before per-entity extraction.

    Returns (is_valid, issues). An empty issues list means the map is valid.
    """
    issues: list[str] = []

    entities = entity_map.get("entities", {})
    if not entities:
        issues.append("No entities found — paper may not contain polymer data")
        return False, issues

    for abbrev, info in entities.items():
        if not isinstance(info, dict):
            issues.append(f"Entity '{abbrev}' has invalid format (expected dict)")
            continue

        if not info.get("full_name"):
            issues.append(f"Entity '{abbrev}' missing full_name")
        if not info.get("type"):
            issues.append(f"Entity '{abbrev}' missing type classification")
        if not info.get("sample_ids"):
            issues.append(
                f"Entity '{abbrev}' has no sample_ids — may be false positive"
            )

        # Derived entities must reference base entities
        if info.get("extraction_class") == "derived":
            deps = info.get("depends_on", [])
            if not deps:
                issues.append(
                    f"Entity '{abbrev}' classified as derived but missing "
                    "depends_on references"
                )
            else:
                for dep in deps:
                    if dep not in entities:
                        issues.append(
                            f"Entity '{abbrev}' depends_on '{dep}' which is "
                            "not in the entity map"
                        )

    # Check for duplicate sample_ids across entities
    all_sample_ids: list[str] = []
    for info in entities.values():
        if isinstance(info, dict):
            all_sample_ids.extend(info.get("sample_ids", []))
    seen = set()
    duplicates = set()
    for sid in all_sample_ids:
        if sid in seen:
            duplicates.add(sid)
        seen.add(sid)
    if duplicates:
        issues.append(f"Duplicate sample_ids across entities: {duplicates}")

    # Check base/derived lists are consistent
    base = set(entity_map.get("base_entities", []))
    derived = set(entity_map.get("derived_entities", []))
    entity_keys = set(entities.keys())

    missing_from_lists = entity_keys - base - derived
    if missing_from_lists:
        issues.append(
            f"Entities not in base_entities or derived_entities: {missing_from_lists}"
        )

    extra_in_lists = (base | derived) - entity_keys
    if extra_in_lists:
        issues.append(
            f"base/derived lists reference non-existent entities: {extra_in_lists}"
        )

    # Log multi-sample entities for visibility
    for abbrev, info in entities.items():
        if isinstance(info, dict):
            sids = info.get("sample_ids", [])
            if len(sids) > 1:
                logger.info(
                    f"Entity '{abbrev}' has {len(sids)} sample_ids "
                    f"— will expand to {len(sids)} extraction tasks"
                )

    return len(issues) == 0, issues


# ---------------------------------------------------------------------------
# Retry message builder
# ---------------------------------------------------------------------------


def _build_retry_message(error_history: list[dict]) -> str:
    """Build a user message summarizing previous validation failures."""
    error_lines = []
    for entry in error_history:
        attempt = entry.get("attempt", "?")
        errors = entry.get("errors", [])
        error_lines.append(f"Attempt {attempt} errors:")
        for e in errors:
            error_lines.append(f"  - {e}")
    return (
        "IMPORTANT: Previous extraction attempts had these issues. "
        "Please correct them in this attempt:\n\n"
        + "\n".join(error_lines)
        + "\n\nFix these issues and try again."
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


async def discover_entities(
    doc_set: DocumentSet,
    llm: LLMClient,
    config: ExtractionConfig,
    max_retries: int = 2,
) -> dict[str, Any]:
    """Run entity discovery with validation and retry.

    Follows guided_navigation.py:main() loop structure. Builds a conversation
    with the LLM using tool calls until a final answer is produced, then
    validates the entity map and retries if needed.

    Returns validated entity_map dict with keys:
    entities, sample_list, paper_polymer_focus, base_entities, derived_entities.
    """
    doc_set.ensure_pages_loaded()
    error_history: list[dict] = []
    entity_map: dict[str, Any] = {"entities": {}, "sample_list": []}

    # Entity discovery focuses on the main paper only.
    # Per-entity extraction still receives the full doc_set.
    if doc_set.has_multiple_documents():
        discovery_doc_set = DocumentSet.from_index(doc_set.primary)
        discovery_doc_set.ensure_pages_loaded()
    else:
        discovery_doc_set = doc_set

    logger.info(
        f"  [entity_discovery] Starting "
        f"(docs={discovery_doc_set.labels()}, multi={discovery_doc_set.has_multiple_documents()}, "
        f"retries={max_retries})"
    )
    docs_consulted: set[str] = set()

    last_transport_error: Exception | None = None

    for attempt in range(max_retries):
        last_transport_error = None
        system_prompt = build_entity_discovery_prompt(discovery_doc_set)
        messages: list[dict] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": ENTITY_DISCOVERY_TASK},
        ]

        # Inject retry context from previous failed attempts
        if error_history:
            messages.append({"role": "user", "content": _build_retry_message(error_history)})

        nav_tools = build_navigation_tools(discovery_doc_set)
        tools = nav_tools if config.vision_enabled else [
            t for t in nav_tools if t["function"]["name"] != "get_page_image"
        ]

        use_vision = supports_vision(config)

        final_answer = None
        for iteration in range(config.entity_max_iterations):
            logger.info(f"  [entity_discovery] iteration {iteration + 1}/{config.entity_max_iterations}")

            try:
                result = await llm.chat_with_tools(
                    messages=messages,
                    tools=tools,
                    tool_choice="auto",
                )
            except Exception as e:
                if is_transport_error(e):
                    # The client already retried; back off before the next
                    # discovery attempt instead of giving up on the paper.
                    last_transport_error = e
                    backoff = (2 ** attempt) + random.random()
                    logger.warning(
                        f"Entity discovery transport error (attempt {attempt + 1}), "
                        f"backing off {backoff:.1f}s: {e}"
                    )
                    await asyncio.sleep(backoff)
                else:
                    logger.error(f"Entity discovery LLM call failed: {type(e).__name__}: {e}")
                break

            messages.append(result.raw_message_dict)

            # No tool calls = final answer
            if not result.tool_calls:
                final_answer = result.content
                break

            # Dispatch tool calls
            pending_images: list[tuple[str, str]] = []
            for tc in result.tool_calls:
                args, arg_error = parse_tool_arguments(tc.arguments)
                if arg_error:
                    logger.warning(f"  [entity_discovery] {tc.name}: {arg_error}")
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": tool_error_result(
                            f"{arg_error}. Call the tool again with valid JSON arguments."
                        ),
                    })
                    continue
                doc_label = args.get("doc") or args.get("tree")
                if doc_label:
                    docs_consulted.add(str(doc_label))
                else:
                    docs_consulted.add(discovery_doc_set.labels()[0])
                result_str, image_data_url = dispatch_tool(
                    discovery_doc_set, tc.name, args,
                    vision_enabled=use_vision,
                    log_context=f"stage=discovery iter={iteration + 1}",
                )
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result_str,
                })
                if image_data_url is not None:
                    img_path = args.get("img_path", "unknown")
                    pending_images.append((img_path, image_data_url))

            logger.info(
                f"  [entity_discovery] tools: "
                f"{summarize_tool_calls(result.tool_calls)}"
            )

            # Inject image user message after all tool responses
            if pending_images:
                parts: list[dict] = [{"type": "text", "text": "Here are the extracted images you requested:"}]
                for img_path, data_url in pending_images:
                    parts.append({"type": "text", "text": f"Image {img_path}:"})
                    parts.append({"type": "image_url", "image_url": {"url": data_url}})
                messages.append({"role": "user", "content": parts})
        else:
            # Max iterations — force final answer
            try:
                final_answer = await wrap_up_conversation(llm, messages, tools)
            except Exception as e:
                if not is_transport_error(e):
                    raise
                # Same treatment as a transport error in the loop: back off
                # and use the next discovery attempt.
                last_transport_error = e
                backoff = (2 ** attempt) + random.random()
                logger.warning(
                    f"Entity discovery wrap-up transport error (attempt {attempt + 1}), "
                    f"backing off {backoff:.1f}s: {e}"
                )
                await asyncio.sleep(backoff)
                final_answer = None

        if not final_answer:
            if last_transport_error is None:
                error_history.append({"attempt": attempt + 1, "errors": ["No answer produced"]})
            continue

        entity_map = parse_entity_answer(final_answer)
        is_valid, issues = validate_entity_map(entity_map)

        if is_valid:
            logger.info(
                f"  [entity_discovery] succeeded (attempt {attempt + 1}): "
                f"{len(entity_map.get('entities', {}))} entities, "
                f"consulted={sorted(docs_consulted)}"
            )
            return entity_map

        if attempt < max_retries - 1:
            logger.warning(
                f"Entity map validation failed (attempt {attempt + 1}), "
                f"retrying: {issues}"
            )
            error_history.append({"attempt": attempt + 1, "errors": issues})
        else:
            logger.warning(
                f"Entity map has issues after {max_retries} attempts "
                f"(proceeding anyway): {issues}"
            )

    if last_transport_error is not None and not entity_map.get("entities"):
        # Every attempt failed to reach the LLM: report it rather than
        # pretending the paper has no polymers.
        raise last_transport_error

    return entity_map

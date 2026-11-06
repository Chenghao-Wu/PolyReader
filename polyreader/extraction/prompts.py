"""System prompt and task templates for per-entity guided extraction.

Follows guided_navigation.py and entity_discovery.py patterns with 6-section
system prompt and 5-step task workflow.
"""

from __future__ import annotations

import json
from typing import Any

from .document_set import DocumentSet

# ---------------------------------------------------------------------------
# Topology vocabulary (shared with entity_discovery.py)
# ---------------------------------------------------------------------------

TOPOLOGY_VOCABULARY = """\
**Chain architecture**: "linear", "star", "branched", "cyclic", "network", "dendrimer"
**Copolymer sequence**: "random", "block", "alternating", "graft", "gradient"
**Special**: "blend", "composite", "homopolymer"
**Compound** (architecture-sequence): "linear-random", "star-block", "branched-graft" """

# ---------------------------------------------------------------------------
# JSON schema template for LLM output (matches PolymerSample structure)
# ---------------------------------------------------------------------------
# NOTE: This template is kept for prompt documentation purposes only.
# The actual schema enforcement is handled by Instructor, which generates
# the schema directly from the PolymerSample Pydantic model.
# ---------------------------------------------------------------------------

JSON_SCHEMA_TEMPLATE = """\
{
  "sample_id": "string — label used in paper (e.g., 'P1', 'PDLLA-5k')",
  "polymer_id": "string — full polymer name",
  "abbreviation": "string or null",
  "constituents": [
    {
      "name": "monomer name",
      "abbreviation": "monomer abbreviation",
      "smiles": "canonical SMILES string for the monomer (use your chemistry knowledge)",
      "mole_fraction": "number 0.0-1.0 or null",
      "weight_fraction": "number 0.0-1.0 or null",
      "role": "monomer | comonomer | crosslinker | initiator_fragment"
    }
  ],
  "topology": "see Section 3 topology vocabulary (e.g. linear, block, star-block, blend)",
  "molecular_weight": {
    "mn": "number (g/mol) or null",
    "mw": "number (g/mol) or null",
    "pdi": "number or null",
    "pdi_derived": "boolean — true if PDI calculated from Mw/Mn",
    "method": "GPC | SEC | MALDI-TOF | viscometry | light_scattering or null",
    "unit": "g/mol (default)",
    "provenance": {
      "source_type": "table | text | figure | supplementary",
      "source_location": "e.g. 'Table 2, row 3'",
      "page_numbers": [1, 2],
      "source_document": "document label or empty for primary"
    }
  },
  "synthesis": {
    "method": "ROP | ATRP | RAFT | free_radical | condensation or null",
    "catalyst": "catalyst name or null",
    "co_catalyst": "co-catalyst name or null",
    "initiator": "initiator name or null",
    "solvent": "solvent name or null",
    "monomer_feed_ratio": "e.g., '50:50 mol%' or null",
    "temp_c": "temperature in Celsius or null",
    "time_h": "time in hours or null",
    "yield_percent": "isolated yield (0-100) or null",
    "conversion_percent": "number 0-100 or null",
    "atmosphere": "N2 | Ar | air | vacuum or null",
    "additional_conditions": {"key": "non-standard synthesis param or null if none"},
    "provenance": {
      "source_type": "table | text | figure | supplementary",
      "source_location": "e.g. 'Section 2.1, paragraph 3'",
      "page_numbers": [3],
      "source_document": "document label or empty for primary"
    }
  },
  "processing_history": [
    {
      "step_id": 1,
      "action": "precipitation | filtration | drying | annealing | solvent_casting | spin_coating | compression_molding | injection_molding | extrusion | electrospinning | uv_curing | etc.",
      "details": "conditions description"
    }
  ],
  "blend_components": [
    {
      "polymer_ref": "reference to base polymer abbreviation",
      "weight_fraction": "number 0.0-1.0 or null",
      "role": "matrix | filler | compatibilizer | reinforcement"
    }
  ],
  // NOTE: Molecular weight (Mn, Mw, PDI) goes in "molecular_weight" above, NOT here.
  "properties": [
    {
      "property_name": "short name, e.g. Tg, T_ODT, morphology, domain_spacing, tensile_strength, storage_modulus",
      "property_display_name": "full name, e.g. 'glass transition temperature' for Tg",
      "value": "numeric value or null",
      "value_str": "non-numeric value",
      "unit": "°C | MPa | % | etc.",
      "method": "DSC | TGA | DMA | tensile_testing | SAXS | WAXS | XRD | rheometry | etc.",
      "method_inferred": "boolean — true if method was inferred from context, not explicitly stated",
      "conditions": {"heating_rate": "10 °C/min"},
      "provenance": {
        "source_type": "table | text | figure | supplementary",
        "source_location": "e.g. 'Table 3, column 4'",
        "page_numbers": [5],
        "source_document": "document label or empty for primary"
      }
    }
  ],
  "anomaly_notes": ["list of validation violations or unusual observations"],
  "sample_type": "base | derived_blend | composite"
}"""

# ---------------------------------------------------------------------------
# Task prompt (5-step workflow)
# ---------------------------------------------------------------------------

GUIDED_EXTRACTION_TASK = """\
Extract ALL data for polymer entity "{entity_id}" from this paper.

Follow this 5-step workflow:

## Step 1: SCAN & TRIAGE
Read the Abstract and Experimental sections.
- Confirm {entity_id} is synthesized in this paper (not commercial/cited)
- Identify its full name, topology, and constituent monomers with their SMILES
- If {entity_id} is a blend, identify its base polymer components

## Step 2: MASTER TABLE
Locate ALL characterization tables for {entity_id} in Results.
Extract ALL reported values: molecular weight (Mn, Mw, PDI → put in
`molecular_weight` field, NOT in `properties`), thermal, mechanical,
structural, and rheological properties (→ put in `properties` array).
Convert kDa to g/mol by multiplying by 1000.
Papers often split data across multiple tables — check every table.

## Step 3: SYNTHESIS
Find polymerization conditions: method, catalyst, initiator, co-catalyst,
solvent, temperature, time, monomer feed ratio, atmosphere.

## Step 4: PROCESSING HISTORY
Trace post-synthesis steps: precipitation, drying, annealing, film casting,
compression molding, electrospinning, etc. Include conditions (solvent, temp, time).

## Step 5: SELF-CHECK
Apply these 4 validation gates before outputting:
(a) Dispersity: Mw ≥ Mn, PDI ≈ Mw/Mn ±5%
(b) Thermal hierarchy: Tg < Tm < Td (when all present)
(c) Missing PDI: calculate if Mn and Mw available, set pdi_derived=true
(d) Entity triage: verify synthesized AND characterized in this paper

Output a single JSON object following the schema in section 4. Use null for
missing values — never use "N/A" or empty strings for numeric fields.

If this entity was NOT synthesized in this paper (only cited, commercial, or
lacking characterization data), output: {{"skipped": true, "reason": "explanation"}}"""

# ---------------------------------------------------------------------------
# Task prompt for targeted (per-sample) extraction
# ---------------------------------------------------------------------------

GUIDED_EXTRACTION_TASK_TARGETED = """\
Extract ALL data for sample **"{target_sample_id}"** of polymer entity "{entity_id}".

This entity contains multiple distinct polymer batches. You must extract data
ONLY for the batch identified as "{target_sample_id}".

Follow this 5-step workflow:

## Step 1: SCAN & TRIAGE
Read the Abstract and Experimental sections.
- Confirm {entity_id} is synthesized in this paper (not commercial/cited)
- Identify its full name, topology, and constituent monomers with their SMILES
- If {entity_id} is a blend, identify its base polymer components

## Step 2: MASTER TABLE
Find ALL characterization data for sample "{target_sample_id}" across all tables.
Molecular weight (Mn, Mw, PDI) → put in `molecular_weight` field, NOT `properties`.
Thermal, mechanical, structural, rheological, and other values → put in `properties` array.
Check every table — data is often split across multiple tables.
Convert kDa to g/mol by multiplying by 1000.
Match data to this sample by its ID label. Do NOT use data from other sample rows
— each sample is a different polymer batch with its own characterization.

## Step 3: SYNTHESIS
Find polymerization conditions: method, catalyst, initiator, co-catalyst,
solvent, temperature, time, monomer feed ratio, atmosphere, etc.
These may be shared across multiple samples of this entity type.

## Step 4: PROCESSING HISTORY
Trace post-synthesis steps: precipitation, drying, annealing, film casting,
compression molding, electrospinning, etc. Include conditions (solvent, temp, time).

## Step 5: SELF-CHECK
Apply these 4 validation gates before outputting:
(a) Dispersity: Mw ≥ Mn, PDI ≈ Mw/Mn ±5%
(b) Thermal hierarchy: Tg < Tm < Td (when all present)
(c) Missing PDI: calculate if Mn and Mw available, set pdi_derived=true
(d) Entity triage: verify synthesized AND characterized in this paper

Output a single JSON object for sample "{target_sample_id}" following the schema
in section 4. Use null for missing values.

If this entity was NOT synthesized in this paper, output: {{"skipped": true, "reason": "explanation"}}"""


# ---------------------------------------------------------------------------
# Base samples context formatter (for derived entities)
# ---------------------------------------------------------------------------

def _format_base_context(base_samples_context: dict[str, Any] | None) -> str:
    """Format base samples context for derived entity extraction."""
    if not base_samples_context:
        return ""

    lines = ["\n## Base Polymer Data (for blend reference)", ""]
    for abbrev, sample_data in base_samples_context.items():
        lines.append(f"### {abbrev}")
        lines.append("```json")
        lines.append(json.dumps(sample_data, indent=2, ensure_ascii=False))
        lines.append("```")
        lines.append("")

    return "\n".join(lines)


def _summarize_entity_map(entity_map: dict[str, Any]) -> str:
    """Create a compact summary of the entity map for context."""
    if not entity_map:
        return "No entities found."

    lines = []
    for abbrev, info in entity_map.items():
        if not isinstance(info, dict):
            continue
        ext_class = info.get("extraction_class", "base")
        deps = info.get("depends_on", [])
        sample_ids = info.get("sample_ids", [])
        full_name = info.get("full_name", "?")

        if ext_class == "derived" and deps:
            lines.append(f"  {abbrev}: {full_name} [{ext_class}, depends on: {', '.join(deps)}]")
        else:
            lines.append(f"  {abbrev}: {full_name} [{ext_class}, samples: {', '.join(sample_ids[:3])}{'...' if len(sample_ids) > 3 else ''}]")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# System prompt builder
# ---------------------------------------------------------------------------

def _format_error_context(error_history: list[dict[str, Any]] | None) -> str:
    """Format previous-attempt errors for a retry prompt."""
    if not error_history:
        return ""
    error_lines = ["\n## Previous Attempt Errors", ""]
    for entry in error_history:
        attempt = entry.get("attempt", "?")
        errors = entry.get("errors", [])
        error_lines.append(f"Attempt {attempt}:")
        for e in errors:
            error_lines.append(f"  - {e}")
    error_lines.append("\n**Fix these issues in your output.**")
    return "\n".join(error_lines)


def _guided_prompt_core(
    entity_id: str,
    entity_info: dict[str, Any],
    entity_map: dict[str, Any],
    base_samples_context: dict[str, Any] | None,
    error_history: list[dict[str, Any]] | None,
    sample_targeting_str: str,
) -> str:
    """Sections 1-5 of the extraction prompt (role, context, ontology,
    schema, validation) — shared by tool-based and full-context extraction."""
    entity_info_str = json.dumps(entity_info, indent=2, ensure_ascii=False)
    base_context_str = _format_base_context(base_samples_context)
    error_context = _format_error_context(error_history)
    return f"""## 1. Role

You are a **polymer data extraction specialist**. Your task is to extract ALL
characterization data for a SINGLE polymer entity: **{entity_id}**

Entity details:
```json
{entity_info_str}
```

## 2. Entity Context

Known entities in this paper:
{_summarize_entity_map(entity_map)}

You are extracting data ONLY for: **{entity_id}**
Other entities are listed for context (e.g., blend component references).
{base_context_str}{sample_targeting_str}
## 3. Domain Ontology (3-Layer Model)

### Layer 1 — Identity
- Full polymer name (IUPAC preferred), abbreviation
- Topology: {TOPOLOGY_VOCABULARY}
- Constituent monomers with feed/incorporation ratios
- SMILES: For each monomer/co-monomer, provide the canonical SMILES string.
  Use stereochemistry-aware SMILES when the paper specifies chirality
  (e.g., L-lactide vs D-lactide vs racemic). Common examples:
  L-lactide: C[C@H]1C(=O)O[C@H](C)C(=O)O1
  glycolide: O=C1COC(=O)CO1
  epsilon-caprolactone: O=C1CCCCCO1
  styrene: C=Cc1ccccc1
  ethylene glycol: OCCO
  If the exact constituent structure is shown in a scheme, reaction diagram,
  or labeled structure image, inspect the image and use it to recover the
  constituent identity and SMILES.
- CRITICAL: Blend = physical mixture of ≥2 distinct polymers.
  Copolymer = single chain with ≥2 monomer types. Most common AI error.

### Layer 2 — Characterization

Extract EVERY characterization value reported for this entity. Scan all tables
and results sections — papers routinely split data across multiple tables.

**Molecular weight** → goes to the `molecular_weight` field (NOT `properties`):
  Mn (number-average), Mw (weight-average), PDI (Đ = Mw/Mn), measurement method (GPC, SEC, etc.)

**Other properties** → go to the `properties` array:
- Thermal: Tg, Tm, Tc, Td, T_ODT, HDT
- Mechanical: tensile_strength, elongation, modulus, toughness, yield_strength
- Structural: morphology, domain_spacing, d_spacing, crystallinity,
  degree_of_branching, crosslink_density, swelling_ratio, gel_fraction
- Rheological: storage_modulus (G'), loss_modulus (G''), complex_viscosity (η*), tan_delta
- Solution: intrinsic_viscosity ([η]), hydrodynamic_radius (Rh), radius_of_gyration (Rg)

Properties vary widely by polymer type — always extract any value the paper
reports, even if not listed above.

- Units: MW in g/mol (if kDa, multiply by 1000). Temperatures in °C (if K, subtract 273.15).
- If PDI not reported but Mn and Mw are, calculate PDI = Mw/Mn, set pdi_derived=true.
- **Property naming**: For each property, provide both a short symbol
  (property_name, e.g. Tg) and the full human-readable name
  (property_display_name, e.g. glass transition temperature).

### Layer 3 — Processing History
- Post-synthesis steps: precipitation, drying, annealing, casting, molding, electrospinning
- Include conditions: solvent, temperature, time

## 4. JSON Output Schema

{JSON_SCHEMA_TEMPLATE}
{error_context}
## 5. Validation Rules (Self-Check)

1. **Dispersity**: If both Mn and Mw present, Mw ≥ Mn. PDI ≈ Mw/Mn (±5%).
2. **Thermal hierarchy**: Tg < Tm < Td (when all present).
3. **Units**: All MW in g/mol, all temperatures in °C, time in hours.
4. **Derived PDI**: If PDI missing but Mn and Mw present, calculate and set pdi_derived=true.

If anomalies exist, record AS-IS and add to anomaly_notes.
"""



def build_guided_extraction_prompt(
    doc_set: DocumentSet,
    entity_id: str,
    entity_info: dict[str, Any],
    entity_map: dict[str, Any],
    base_samples_context: dict[str, Any] | None = None,
    error_history: list[dict[str, Any]] | None = None,
    target_sample_id: str | None = None,
    is_multi_sample: bool = False,
) -> str:
    """Build self-contained 6-section system prompt for single-entity extraction.

    Follows entity_discovery.py:build_entity_discovery_prompt() pattern — all
    sections in one f-string.

    Args:
        doc_set: DocumentSet with one or more indexed documents.
        entity_id: The abbreviation/name of the entity to extract.
        entity_info: Entity metadata from entity_map (full_name, type, sample_ids, etc.).
        entity_map: Full entity map for context on other entities.
        base_samples_context: For derived entities, extracted data from base polymers.
        error_history: Previous validation errors for retry context.
        target_sample_id: When set, extract data for this specific sample only.
        is_multi_sample: True when this entity has multiple samples being extracted separately.

    Returns:
        Complete system prompt string.
    """
    # Build sample targeting section for multi-sample entities
    sample_targeting_str = ""
    if is_multi_sample and target_sample_id:
        sibling_ids = [
            sid for sid in entity_info.get("sample_ids", [])
            if sid != target_sample_id
        ]
        siblings_str = ", ".join(sibling_ids) if sibling_ids else "(none)"
        sample_targeting_str = f"""
## 2a. Sample Targeting

You are extracting data for one specific sample: **{target_sample_id}**

This entity contains multiple distinct polymer batches, each with its own
characterization data. Extract data ONLY for the batch labeled "{target_sample_id}".

Synthesis conditions may be shared across samples of this entity type — extract
them once. Characterization data (molecular weight, thermal properties, mechanical
properties) is unique per sample — match values to this sample by its label.

Sibling samples (for context, do NOT extract): {siblings_str}
"""

    return _guided_prompt_core(
        entity_id, entity_info, entity_map, base_samples_context,
        error_history, sample_targeting_str,
    ) + f"""
## 6. Execution Strategy

You have 3 tools: get_page_content (PRIMARY), get_page_image (retrieves extracted
images by img_path found in page text), get_document_structure (retrieve the paper's section tree).
{"" if not doc_set.has_multiple_documents() else chr(10) + "You have " + str(len(doc_set)) + " documents available. Scan ALL of them for relevant data." + chr(10) + "Characterization tables, additional spectra, and detailed procedures" + chr(10) + "often appear in supplementary documents." + chr(10) + 'Use the "doc" parameter to specify which document to read from, e.g. get_page_content(doc="<label>", pages="1-3").' + chr(10)}
Follow this reading order:
1. Abstract — confirm entity identity
2. Experimental / Methods — synthesis conditions, sample preparation
3. Results & Tables — scan ALL tables; extract every reported value (Mn, Mw, PDI, thermal, structural, mechanical, rheological)
{"4. Scan remaining documents for additional tables, characterization data" + chr(10) + "5." if doc_set.has_multiple_documents() else "4."} Compile all data into the JSON template
{"6." if doc_set.has_multiple_documents() else "5."} Run the 4 validation gates (dispersity, thermal hierarchy, derived PDI, entity triage)
{"7." if doc_set.has_multiple_documents() else "6."} Output the final JSON as your answer — no tool calls in the final message

When you see ![](img_path) in page text, use get_page_image(img_path='...') to
view that figure or diagram. Prioritize images that show reaction schemes,
chemical structures, monomer labels, or tables embedded as figures; these may
contain constituent identities or structures needed to recover SMILES. Use
get_page_content systematically, reading 2-3 pages at a time to stay within
token limits.

When you have all the data you need, provide your final answer as a message
with NO tool calls."""


def build_guided_extraction_task(
    entity_id: str,
    target_sample_id: str | None = None,
) -> str:
    """Build the task prompt for a specific entity.

    Args:
        entity_id: The entity abbreviation to extract.
        target_sample_id: When set, use the targeted prompt for this specific sample.

    Returns:
        Task prompt string.
    """
    if target_sample_id:
        return GUIDED_EXTRACTION_TASK_TARGETED.format(
            entity_id=entity_id,
            target_sample_id=target_sample_id,
        )
    return GUIDED_EXTRACTION_TASK.format(entity_id=entity_id)

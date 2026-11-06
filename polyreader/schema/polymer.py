"""Pydantic v2 models for structured polymer data extraction.

Follows "Strict Scaffold, Flexible Payload": core fields are typed,
domain-specific details use dicts for extensibility.
"""

from __future__ import annotations

import math
import re

from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic_core import PydanticUndefined

# Quality tier constants
TIER_GOLD = "gold"
TIER_SILVER = "silver"
TIER_BRONZE = "bronze"
TIER_REJECT = "reject"

# Sample type constants
TYPE_BASE = "base"
TYPE_DERIVED_BLEND = "derived_blend"
TYPE_COMPOSITE = "composite"


_NUMBER_RE = re.compile(r"^[~≈<>]?\s*([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)\s*(.*)$")
_KILO_MW_UNITS = {"kda", "kg/mol", "kg mol-1", "kg mol^-1", "kg·mol-1", "kg·mol⁻¹", "kg mol⁻¹"}


def _parse_number(value, *, mw_units: bool = False, percent: bool = False):
    """Best-effort conversion of an LLM-provided value to float.

    Returns ``value`` unchanged when it is not a string/number (so Pydantic
    reports the type error), None for blanks, NaN/inf and non-numeric text.
    Strings like "12,300", "12.3 kDa" (with ``mw_units``), "130 °C" or
    "85%" (with ``percent``: "85%" -> 0.85) are understood.
    """
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return None if isinstance(value, float) and not math.isfinite(value) else value
    if not isinstance(value, str):
        return value
    text = value.strip().replace("\u2212", "-").replace("\u2009", "").replace("\u202f", "")
    text = re.sub(r"(?<=\d)[,\s](?=\d{3}(?!\d))", "", text)  # thousands separators
    text = re.sub(r"(?<=\d),(?=\d)", ".", text)  # decimal comma
    match = _NUMBER_RE.match(text)
    if not match:
        return None
    unit = match.group(2).strip().lower()
    if re.match(r"^[-–—~]\s*\d", unit) or re.match(r"^to\s*\d", unit):
        return None  # a range such as "160-165" is not a single number
    number = float(match.group(1))
    if not math.isfinite(number):
        return None
    if mw_units and unit in _KILO_MW_UNITS:
        number *= 1000.0
    elif percent and unit.startswith("%"):
        number /= 100.0
    return number


class CoercingBaseModel(BaseModel):
    """Base model that coerces None to field defaults for non-Optional fields."""

    @model_validator(mode="before")
    @classmethod
    def _coerce_none_to_defaults(cls, data):
        if not isinstance(data, dict):
            return data
        for field_name, field_info in cls.model_fields.items():
            if field_name not in data or data[field_name] is not None:
                continue
            # Field value is None — coerce if the field doesn't accept None
            if field_info.default is not None and field_info.default is not PydanticUndefined:
                # Field has a concrete non-None default (e.g., "", False, 0)
                data[field_name] = field_info.default
            elif field_info.default_factory is not None:
                # Field uses default_factory (e.g., list, dict, Provenance)
                data[field_name] = field_info.default_factory()
            # else: default is None → field accepts None (Optional) → leave it
        return data


class PaperMetadataRef(CoercingBaseModel):
    """Bibliographic metadata for the source paper."""

    doi: str = Field("", description="DOI of the source paper.")
    title: str = Field("", description="Title of the source paper.")
    authors: list[str] = Field(default_factory=list, description="Authors of the source paper.")
    journal: str = Field("", description="Journal name.")
    year: int | None = Field(None, description="Publication year.")
    volume: str = Field("", description="Journal volume.")
    pages: str = Field("", description="Page range.")


class Provenance(CoercingBaseModel):
    """Tracks where a data point was extracted from."""

    source_type: str = Field(
        "",
        description="Source type: table, text, figure, or supplementary.",
    )
    source_location: str = Field(
        "",
        description="Location in paper, e.g. 'Table 2, row 3' or 'Section 3.2, paragraph 1'.",
    )
    page_numbers: list[int] = Field(
        default_factory=list,
        description="Page numbers where the data was found.",
    )
    source_document: str = Field(
        "",
        description="Document label where data was found. Empty = primary document.",
    )


class Monomer(CoercingBaseModel):
    """A monomer or co-monomer constituent."""

    name: str = Field(
        "",
        description="Monomer name, e.g. L-lactide, glycolide, epsilon-caprolactone.",
    )
    abbreviation: str = Field(
        "",
        description="Monomer abbreviation, e.g. LA, GA, CL.",
    )
    smiles: str = Field(
        "",
        description="Canonical SMILES string for the monomer. Use stereochemistry "
        "when specified (e.g., L-lactide: C[C@H]1C(=O)O[C@H](C)C(=O)O1). "
        "Leave empty if unsure.",
    )
    mole_fraction: float | None = Field(
        None,
        description="Mole fraction in copolymer (0.0-1.0).",
    )
    weight_fraction: float | None = Field(
        None,
        description="Weight fraction in copolymer (0.0-1.0).",
    )
    role: str = Field(
        "",
        description="Role: monomer, comonomer, crosslinker, or initiator_fragment.",
    )

    @field_validator("mole_fraction", "weight_fraction", mode="before")
    @classmethod
    def parse_fraction(cls, v):
        return _parse_number(v, percent=True)


class MolecularWeight(CoercingBaseModel):
    """Molecular weight characterization data."""

    mn: float | None = Field(
        None,
        description="Number-average molecular weight in g/mol. Convert kDa to g/mol (×1000).",
    )
    mw: float | None = Field(
        None,
        description="Weight-average molecular weight in g/mol. Must be >= Mn.",
    )
    pdi: float | None = Field(
        None,
        description="Polydispersity index (Mw/Mn). Typically 1.0-5.0 for synthetic polymers.",
    )
    pdi_derived: bool = Field(
        False,
        description="True if PDI was calculated from Mw/Mn rather than directly reported.",
    )
    method: str = Field(
        "",
        description="Measurement method: GPC, SEC, MALDI-TOF, viscometry, or light scattering.",
    )
    unit: str | None = "g/mol"
    provenance: Provenance = Field(default_factory=Provenance)

    @field_validator("mn", "mw", mode="before")
    @classmethod
    def reject_negative_mw(cls, v: float | None) -> float | None:
        v = _parse_number(v, mw_units=True)
        if isinstance(v, (int, float)) and v < 0:
            return None
        return v

    @field_validator("pdi", mode="before")
    @classmethod
    def reject_invalid_pdi(cls, v: float | None) -> float | None:
        v = _parse_number(v)
        if isinstance(v, (int, float)) and (v <= 0 or v > 20):
            return None
        return v

    @model_validator(mode="after")
    def _nullify_unit_when_empty(self) -> MolecularWeight:
        """Unit is meaningless without any numeric molecular weight values."""
        if self.mn is None and self.mw is None and self.pdi is None:
            self.unit = None
        return self


class SynthesisDetails(CoercingBaseModel):
    """Synthesis method and conditions."""

    method: str = Field(
        "",
        description="Polymerization method: ROP, ATRP, RAFT, free radical, condensation, etc.",
    )
    catalyst: str = Field(
        "",
        description="Catalyst name, e.g. Sn(Oct)2, Grubbs 2nd gen, Al(i-Bu)3.",
    )
    co_catalyst: str = Field(
        "",
        description="Co-catalyst if present, e.g. benzyl alcohol, PPN-Cl. Use empty string if not applicable.",
    )
    initiator: str = Field(
        "",
        description="Initiator name, e.g. AIBN, BPO, ethylene glycol.",
    )
    solvent: str = Field(
        "",
        description="Reaction solvent, e.g. toluene, THF, bulk (no solvent). Use empty string if not specified.",
    )
    monomer_feed_ratio: str | None = Field(
        None,
        description="Monomer feed ratio if copolymer, e.g. '50:50 mol%', 'LA:GA = 75:25'.",
    )
    temp_c: float | None = Field(
        None,
        description="Reaction temperature in degrees Celsius.",
    )
    time_h: float | None = Field(
        None,
        description="Reaction time in hours.",
    )
    yield_percent: float | None = Field(
        None,
        description="Isolated yield as percentage (0-100).",
    )
    atmosphere: str = Field(
        "",
        description="Reaction atmosphere: N2, Ar, air, vacuum. Use empty string if not specified.",
    )
    additional_conditions: dict = Field(default_factory=dict)
    provenance: Provenance = Field(default_factory=Provenance)

    @field_validator("co_catalyst", "solvent", "atmosphere", mode="before")
    @classmethod
    def empty_str_if_none(cls, v):
        """Convert None to empty string for optional synthesis fields."""
        return v if v is not None else ""

    @field_validator("temp_c", mode="before")
    @classmethod
    def reject_impossible_temp(cls, v: float | None) -> float | None:
        v = _parse_number(v)
        if isinstance(v, (int, float)) and (v < -273 or v > 5000):
            return None
        return v

    @field_validator("time_h", "yield_percent", mode="before")
    @classmethod
    def parse_numeric(cls, v):
        return _parse_number(v)


class PropertyObject(CoercingBaseModel):
    """A single measured or reported property."""

    property_name: str = Field(
        "",
        description=(
            "Property symbol/short name. "
            "Thermal: Tg, Tm, Tc, Td, T_ODT, HDT. "
            "Mechanical: tensile_strength, elongation, modulus, toughness, yield_strength. "
            "Structural: morphology, domain_spacing, d_spacing, crystallinity, "
            "degree_of_branching, crosslink_density, swelling_ratio, gel_fraction. "
            "Rheological: storage_modulus, loss_modulus, complex_viscosity, tan_delta, melt_viscosity. "
            "Solution: intrinsic_viscosity, hydrodynamic_radius, radius_of_gyration. "
            "Use any concise name for other reported properties."
        ),
    )
    property_display_name: str = Field(
        "",
        description=(
            "Full human-readable name of the property. Examples: "
            "Tg -> 'glass transition temperature', "
            "Tm -> 'melting temperature', "
            "Td -> 'decomposition temperature', "
            "Tc -> 'crystallization temperature', "
            "tensile_strength -> 'tensile strength', "
            "elongation -> 'elongation at break', "
            "modulus -> 'Young's modulus', "
            "morphology -> 'morphology', "
            "domain_spacing -> 'domain spacing', "
            "T_ODT -> 'order-disorder transition temperature', "
            "crystallinity -> 'degree of crystallinity', "
            "storage_modulus -> 'storage modulus', "
            "loss_modulus -> 'loss modulus', "
            "complex_viscosity -> 'complex viscosity', "
            "degree_of_branching -> 'degree of branching', "
            "crosslink_density -> 'crosslink density', "
            "swelling_ratio -> 'swelling ratio', "
            "gel_fraction -> 'gel fraction', "
            "intrinsic_viscosity -> 'intrinsic viscosity', "
            "hydrodynamic_radius -> 'hydrodynamic radius', "
            "radius_of_gyration -> 'radius of gyration'."
        ),
    )
    value: float | None = Field(
        None,
        description="Numeric value of the property.",
    )
    value_str: str = Field(
        "",
        description="Non-numeric or complex value representation. Use empty string if numeric value is provided.",
    )
    unit: str = Field(
        "",
        description="Unit of measurement, e.g. °C, MPa, %.",
    )
    method: str = Field(
        "",
        description="Measurement method: DSC, TGA, DMA, tensile_testing, SAXS, WAXS, XRD, rheometry, etc.",
    )
    method_inferred: bool = Field(
        False,
        description="True if the measurement method was inferred from context rather than explicitly stated.",
    )
    conditions: dict = Field(
        default_factory=dict,
        description="Measurement conditions, e.g. {'heating_rate': '10 C/min'}.",
    )
    provenance: Provenance = Field(default_factory=Provenance)

    @field_validator("value_str", mode="before")
    @classmethod
    def empty_str_if_none(cls, v):
        """Convert None to empty string for value_str."""
        return v if v is not None else ""

    @model_validator(mode="before")
    @classmethod
    def _non_numeric_value_to_str(cls, data):
        """Keep non-numeric values (e.g. "amorphous", "n.d.") in value_str."""
        if not isinstance(data, dict):
            return data
        raw = data.get("value")
        parsed = _parse_number(raw)
        if isinstance(raw, str) and parsed is None:
            data = dict(data)
            data["value"] = None
            if raw.strip() and not data.get("value_str"):
                data["value_str"] = raw.strip()
        elif raw is not None and parsed is None:
            data = dict(data)
            data["value"] = None  # NaN / inf / bool
        elif isinstance(raw, str):
            data = dict(data)
            data["value"] = parsed
        return data


class BlendComponent(CoercingBaseModel):
    """A component in a polymer blend or composite."""

    polymer_ref: str = Field(
        "",
        description="Reference to another sample_id or polymer abbreviation in the blend.",
    )
    weight_fraction: float | None = Field(
        None,
        description="Weight fraction of this component in the blend (0.0-1.0).",
    )
    role: str = Field(
        "",
        description="Role in blend: matrix, filler, compatibilizer, or reinforcement.",
    )


class ProcessingStep(CoercingBaseModel):
    """A single processing step in the sample's history."""

    step_id: int = Field(
        0,
        description="Sequential step number (1, 2, 3, ...).",
    )
    action: str = Field(
        "",
        description="Processing action: synthesis, purification, annealing, melt-pressing, casting, etc.",
    )
    details: str = Field(
        "",
        description="Details of the processing step, e.g. 'methanol precipitation, RT, 2h'.",
    )

    @field_validator("step_id", mode="before")
    @classmethod
    def parse_step_id(cls, v):
        parsed = _parse_number(v)
        if isinstance(v, str) and parsed is None:
            return 0
        return int(parsed) if isinstance(parsed, float) else parsed


def _strip_empty(obj):
    """Recursively remove None, empty strings, empty lists, empty dicts, and default-only structures."""
    if isinstance(obj, dict):
        cleaned = {}
        for k, v in obj.items():
            v = _strip_empty(v)
            if v is None or v == "" or v == [] or v == {}:
                continue
            cleaned[k] = v
        # molecular_weight with only a default unit and no actual data is meaningless
        if cleaned == {"unit": "g/mol"}:
            return {}
        # molecular_weight with only pdi_derived=False and no actual data is meaningless
        if cleaned == {"pdi_derived": False}:
            return {}
        # molecular_weight with only default unit and pdi_derived=False is meaningless
        if cleaned == {"unit": "g/mol", "pdi_derived": False}:
            return {}
        # property with property_name but no value or value_str is meaningless
        if "property_name" in cleaned and "value" not in cleaned and "value_str" not in cleaned:
            return {}
        return cleaned
    elif isinstance(obj, list):
        cleaned_items = [_strip_empty(item) for item in obj]
        return [item for item in cleaned_items if item != {} and item is not None]
    return obj


class PolymerSample(CoercingBaseModel):
    """A single polymer sample with all extracted data."""

    metadata: PaperMetadataRef = Field(
        default_factory=PaperMetadataRef,
        description="Bibliographic metadata of the source paper.",
    )
    sample_id: str = Field(
        "",
        description="Sample label as used in the paper, e.g. 'P1', 'PDLLA-5k', 'Sample A'.",
    )
    polymer_id: str = Field(
        "",
        description="Standardized polymer name, e.g. 'poly(D,L-lactide)', 'poly(ethylene glycol)'.",
    )
    abbreviation: str = Field(
        "",
        description="Polymer abbreviation, e.g. 'PDLLA', 'PEG'.",
    )
    constituents: list[Monomer] = Field(
        default_factory=list,
        description="List of monomer constituents with feed/incorporation ratios.",
    )
    topology: str = Field(
        "",
        description="Polymer topology: linear, star, branched, cyclic, network, blend, block, graft, etc.",
    )
    molecular_weight: MolecularWeight = Field(
        default_factory=MolecularWeight,
        description="Molecular weight characterization data (Mn, Mw, PDI).",
    )
    synthesis: SynthesisDetails = Field(
        default_factory=SynthesisDetails,
        description="Synthesis method and conditions.",
    )
    processing_history: list[ProcessingStep] = Field(
        default_factory=list,
        description="Post-synthesis processing steps (precipitation, drying, annealing, etc.).",
    )
    blend_components: list[BlendComponent] = Field(
        default_factory=list,
        description="For blends/composites: list of component polymers with their fractions.",
    )
    properties: list[PropertyObject] = Field(
        default_factory=list,
        description=(
            "All measured properties: thermal (Tg, Tm, Td, T_ODT), mechanical (tensile, elongation, modulus), "
            "structural (morphology, domain_spacing, crystallinity, degree_of_branching, crosslink_density, swelling_ratio), "
            "rheological (G', G'', viscosity), solution (intrinsic_viscosity, hydrodynamic_radius, radius_of_gyration), "
            "and any other reported values."
        ),
    )
    validation_flags: list[str] = Field(
        default_factory=list,
        description="Non-blocking validation warnings, e.g. 'mw_less_than_mn', 'missing_pdi'.",
    )
    quality_tier: str = Field(
        "",
        description="Quality classification: gold, silver, bronze, or reject.",
    )
    quality_tier_reason: str = Field(
        "",
        description="Human-readable explanation of quality tier, listing missing criteria.",
    )
    sample_type: str = Field(
        "",
        description="Sample type: base (synthesized), derived_blend (physical mixture), or composite.",
    )
    anomaly_notes: list[str] = Field(
        default_factory=list,
        description="Paper-reported anomalies or validation violations, e.g. 'PDI does not match Mw/Mn'.",
    )

    def clean_dump(self) -> dict:
        """Serialize to dict with empty/default values stripped for clean output."""
        return _strip_empty(self.model_dump())


class ExtractionResult(CoercingBaseModel):
    """Complete extraction output for one paper."""

    metadata: PaperMetadataRef = Field(
        default_factory=PaperMetadataRef,
        description="Bibliographic metadata of the source paper.",
    )
    samples: list[PolymerSample] = Field(
        default_factory=list,
        description="List of extracted polymer samples.",
    )
    entity_map: dict = Field(
        default_factory=dict,
        description="Entity map from discovery: abbreviation -> entity info.",
    )
    extraction_trace: list[dict] = Field(
        default_factory=list,
        description="Per-agent trace logs for debugging.",
    )
    sample_counts: dict = Field(
        default_factory=dict,
        description="Sample counts by quality tier and type.",
    )

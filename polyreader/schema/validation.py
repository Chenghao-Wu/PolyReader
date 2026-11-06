"""Physical validation rules for polymer data.

All rules are non-blocking: they return flag strings rather than raising errors.
This allows extraction to proceed even with questionable data, while flagging
potential issues for review.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

try:
    from rdkit import RDLogger
    RDLogger.logger().setLevel(RDLogger.ERROR)
except ImportError:
    pass

from .polymer import (
    PolymerSample,
    TIER_GOLD, TIER_SILVER, TIER_BRONZE, TIER_REJECT,
)


# ---------------------------------------------------------------------------
# Validation issue codes
# ---------------------------------------------------------------------------

# Stable machine-readable codes. Callers (e.g. the retry logic) must match on
# these, never on the human-readable messages, which may change wording.
MW_LESS_THAN_MN = "mw_less_than_mn"
PDI_INCONSISTENT = "pdi_inconsistent_with_mw_mn"
PDI_BELOW_ONE = "pdi_below_one"
PDI_UNUSUALLY_HIGH = "pdi_unusually_high"
MN_SUSPICIOUSLY_LOW = "mn_suspiciously_low"
MN_SUSPICIOUSLY_HIGH = "mn_suspiciously_high"
MW_SUSPICIOUSLY_HIGH = "mw_suspiciously_high"
TG_NOT_BELOW_TM = "tg_not_below_tm"
TM_NOT_BELOW_TD = "tm_not_below_td"
MOLE_FRACTION_SUM = "mole_fraction_sum"
WEIGHT_FRACTION_SUM = "weight_fraction_sum"
HOMOPOLYMER_FRACTION = "homopolymer_fraction"
THERMAL_OUT_OF_RANGE = "thermal_out_of_range"
SYNTHESIS_TEMP_OUT_OF_RANGE = "synthesis_temp_out_of_range"
SMILES_EMPTY = "smiles_empty"
SMILES_INVALID = "smiles_invalid"


@dataclass(frozen=True)
class ValidationIssue:
    """A validation finding: stable ``code`` + human-readable ``message``."""

    code: str
    message: str

    def __str__(self) -> str:
        return self.message


# ---------------------------------------------------------------------------
# Property classification
# ---------------------------------------------------------------------------

# Canonical property names per category (used as documentation + exact-match fallback)
PROPERTY_CATEGORIES = {
    "thermal": {"tg", "tm", "tc", "td", "t_odt", "hdt"},
    "mechanical": {"tensile_strength", "elongation", "modulus", "toughness", "yield_strength"},
    "structural": {"morphology", "domain_spacing", "d_spacing", "crystallinity",
                   "degree_of_branching", "crosslink_density", "swelling_ratio", "gel_fraction"},
    "rheological": {"storage_modulus", "loss_modulus", "complex_viscosity", "tan_delta", "melt_viscosity"},
    "solution": {"intrinsic_viscosity", "hydrodynamic_radius", "radius_of_gyration"},
}

# Prefixes that identify thermal property variants (tg_annealed, tm1, td_5%, etc.)
_THERMAL_PREFIXES = ("tg", "tm", "td", "tc", "t_odt", "hdt")


def classify_property(name: str) -> str:
    """Classify a property name into a category.

    Handles variants like tg_annealed, tm1, td_5%, flexural_modulus, etc.
    Returns category name or "other".
    """
    key = name.lower().strip()

    # 1. Exact match against canonical names
    for category, names in PROPERTY_CATEGORIES.items():
        if key in names:
            return category

    # 2. Thermal: prefix matching (tg_onset, tm1, td_5%, hdt_under_load, etc.)
    if any(key.startswith(p) for p in _THERMAL_PREFIXES):
        return "thermal"

    # 3. Rheological: check BEFORE mechanical (both involve modulus/viscosity)
    if any(kw in key for kw in ("storage_modulus", "loss_modulus",
                                 "complex_viscosity", "tan_delta", "melt_viscosity")):
        return "rheological"

    # 4. Mechanical: keywords (after rheological, so bare "modulus" → mechanical)
    if any(kw in key for kw in ("tensile", "elongation", "modulus", "toughness",
                                 "yield_strength", "flexural", "compressive", "impact")):
        return "mechanical"

    # 5. Structural: keywords
    if any(kw in key for kw in ("morphology", "spacing", "crystallinity", "branching",
                                 "crosslink", "swelling", "gel_fraction")):
        return "structural"

    # 6. Solution: keywords
    if any(kw in key for kw in ("intrinsic_viscosity", "hydrodynamic", "radius_of_gyration")):
        return "solution"

    # 7. Catch-all for "viscosity" not matched above
    if "viscosity" in key:
        return "solution" if "intrinsic" in key else "rheological"

    return "other"


# ---------------------------------------------------------------------------
# MW rescue
# ---------------------------------------------------------------------------

# MW aliases that should live in molecular_weight, not properties
_MW_ALIASES = frozenset({
    "mn", "mw", "pdi",
    "dispersity", "polydispersity", "polydispersity_index",
    "number_average_molecular_weight", "weight_average_molecular_weight",
    "mw/mn",
})

# Map property names to MolecularWeight field names
_MW_FIELD_MAP = {
    "mn": "mn",
    "number_average_molecular_weight": "mn",
    "mw": "mw",
    "weight_average_molecular_weight": "mw",
    "pdi": "pdi",
    "dispersity": "pdi",
    "polydispersity": "pdi",
    "polydispersity_index": "pdi",
    "mw/mn": "pdi",
    "đ": "pdi",
}

# Unit multipliers to g/mol for molecular-weight properties.
_MW_UNIT_FACTORS = {
    "": 1.0,
    "g/mol": 1.0,
    "g mol-1": 1.0,
    "g mol^-1": 1.0,
    "g·mol-1": 1.0,
    "g·mol⁻¹": 1.0,
    "g mol⁻¹": 1.0,
    "da": 1.0,
    "kda": 1000.0,
    "kg/mol": 1000.0,
    "kg mol-1": 1000.0,
    "kg mol^-1": 1000.0,
    "kg·mol-1": 1000.0,
    "kg·mol⁻¹": 1000.0,
    "kg mol⁻¹": 1000.0,
}


def _mw_value_in_g_per_mol(value: float, unit: str) -> float | None:
    """Convert a molecular-weight property value to g/mol (None if unit unknown)."""
    factor = _MW_UNIT_FACTORS.get((unit or "").strip().lower())
    return None if factor is None else value * factor


def _same_value(a: float, b: float, rel_tol: float = 0.01) -> bool:
    return math.isclose(a, b, rel_tol=rel_tol, abs_tol=1e-9)


def rescue_mw_from_properties(sample: PolymerSample) -> PolymerSample:
    """Move misplaced MW data from properties to molecular_weight.

    If the LLM puts Mn/Mw/PDI into the properties array instead of the
    dedicated molecular_weight field, this function rescues them.

    Each of Mn, Mw and PDI is recovered independently: only fields that are
    currently None are filled, existing values are never overwritten.
    Properties are removed when their value was moved, or when they merely
    duplicate the value already in molecular_weight; a property that
    conflicts with an existing value is left in place. Values in kDa or
    kg/mol are converted to g/mol; properties with an unknown unit are not
    moved. Returns the sample for chaining.
    """
    mw = sample.molecular_weight
    remove_indices: list[int] = []
    rescued_numeric = False

    for i, prop in enumerate(sample.properties):
        key = prop.property_name.lower().strip()
        if key not in _MW_ALIASES:
            continue

        field = _MW_FIELD_MAP.get(key)
        if field is None or prop.value is None:
            continue

        if field == "pdi":
            value = prop.value  # dimensionless
        else:
            value = _mw_value_in_g_per_mol(prop.value, prop.unit)
            if value is None:
                continue  # unknown unit — leave it for a human to check

        current = getattr(mw, field)
        if current is None:
            setattr(mw, field, value)
            remove_indices.append(i)
            if field in ("mn", "mw"):
                rescued_numeric = True
            # Also rescue method if available
            if prop.method and not mw.method:
                mw.method = prop.method
        elif _same_value(current, value):
            remove_indices.append(i)  # exact duplicate of existing data
        # else: conflicting value — keep the property, keep molecular_weight

    if remove_indices:
        # Remove rescued entries from properties (iterate in reverse)
        for i in sorted(remove_indices, reverse=True):
            sample.properties.pop(i)

        # Restore unit if we rescued any numeric MW data
        if rescued_numeric or mw.mn is not None or mw.mw is not None:
            mw.unit = "g/mol"

    return sample


def classify_quality_tier(sample: PolymerSample) -> tuple[str, str]:
    """Classify sample into Gold/Silver/Bronze/Reject tier.

    Must-have (4): constituents, topology, molecular weight (Mn or Mw), at least one property.
    Should-have (2): synthesis (method + condition), processing history.

    Returns:
        Tuple of (tier, reason) where reason lists missing criteria.
    """
    must_have = {
        "constituents": bool(sample.constituents and any(m.name for m in sample.constituents)),
        "topology": bool(sample.topology),
        "distribution": (
            sample.molecular_weight.mn is not None
            or sample.molecular_weight.mw is not None
        ),
        "properties": any(
            p.value is not None or p.value_str
            for p in sample.properties
        ),
    }

    should_have = {
        "synthesis": bool(sample.synthesis.method) and (
            bool(sample.synthesis.catalyst)
            or sample.synthesis.temp_c is not None
            or sample.synthesis.time_h is not None
        ),
        "processing": bool(sample.processing_history),
    }

    must_count = sum(must_have.values())
    should_count = sum(should_have.values())

    missing_must = [k for k, v in must_have.items() if not v]
    missing_should = [k for k, v in should_have.items() if not v]
    parts = []
    if missing_must:
        parts.append("missing " + ", ".join(missing_must))
    if missing_should:
        parts.append("no " + ", ".join(missing_should))
    reason = "; ".join(parts) if parts else "all criteria met"

    if must_count == 4:
        return TIER_GOLD, reason
    elif must_count == 3:
        return TIER_SILVER, reason
    elif must_count >= 2:
        return TIER_BRONZE, reason
    else:
        return TIER_REJECT, reason


def reject_invalid_sample(sample: PolymerSample) -> str | None:
    """Hard rejection gate. Returns reason string if sample must be rejected, else None."""
    # Ghost sample: no identity at all
    if not sample.sample_id and not sample.polymer_id:
        return "no sample_id or polymer_id"

    return None


def validate_sample_issues(sample: PolymerSample) -> list[ValidationIssue]:
    """Run all physical validation rules; return coded issues."""
    issues: list[ValidationIssue] = []

    issues.extend(_check_molecular_weight(sample))
    issues.extend(_check_thermal_properties(sample))
    issues.extend(_check_composition(sample))
    issues.extend(_check_temperature_ranges(sample))
    issues.extend(_check_smiles_validity(sample))

    return issues


def validate_sample(sample: PolymerSample) -> list[str]:
    """Run all physical validation rules on a sample.

    Returns:
        List of warning flag strings. Empty list means all checks passed.
        Use :func:`validate_sample_issues` for the stable issue codes.
    """
    return [issue.message for issue in validate_sample_issues(sample)]


def _check_molecular_weight(sample: PolymerSample) -> list[ValidationIssue]:
    """Check Mw > Mn and PDI consistency."""
    flags = []
    mw_data = sample.molecular_weight

    mn = mw_data.mn
    mw = mw_data.mw
    pdi = mw_data.pdi

    # PDI physical bounds
    if pdi is not None and 0 < pdi < 1.0:
        flags.append(ValidationIssue(
            PDI_BELOW_ONE,
            f"PDI = {pdi:.3f} < 1.0: physically impossible (Mw/Mn >= 1 by definition)",
        ))
    if pdi is not None and pdi > 5.0:
        flags.append(ValidationIssue(
            PDI_UNUSUALLY_HIGH,
            f"PDI = {pdi:.3f} is unusually high for synthetic polymers (typical range 1.0-5.0)",
        ))

    # Mw should be >= Mn
    if mn is not None and mw is not None and mn > 0 and mw > 0:
        if mw < mn:
            flags.append(ValidationIssue(
                MW_LESS_THAN_MN,
                f"Mw ({mw}) < Mn ({mn}): weight-average should be >= number-average",
            ))

        # PDI consistency: PDI ≈ Mw/Mn within 15%
        if pdi is not None and pdi > 0:
            expected_pdi = mw / mn
            relative_error = abs(pdi - expected_pdi) / expected_pdi
            if relative_error > 0.15:
                flags.append(ValidationIssue(
                    PDI_INCONSISTENT,
                    f"PDI inconsistency: reported {pdi:.3f} vs calculated Mw/Mn = {expected_pdi:.3f} "
                    f"(error {relative_error:.1%})",
                ))

    # Flag likely unit errors (kDa stored as g/mol)
    if mn is not None and mn > 0 and mn < 500 and mw_data.unit == "g/mol":
        flags.append(ValidationIssue(
            MN_SUSPICIOUSLY_LOW,
            f"Mn = {mn} g/mol is suspiciously low — possible kDa value stored as g/mol",
        ))
    if mn is not None and mn > 1e7:
        flags.append(ValidationIssue(
            MN_SUSPICIOUSLY_HIGH,
            f"Mn = {mn} g/mol is suspiciously high — possible unit error",
        ))
    if mw is not None and mw > 1e7:
        flags.append(ValidationIssue(
            MW_SUSPICIOUSLY_HIGH,
            f"Mw = {mw} g/mol is suspiciously high — possible unit error",
        ))

    return flags


def _is_thermal_symbol(name: str, symbol: str) -> bool:
    """True for ``symbol`` and its variants (Tg, Tg1, Tg_onset, Tg,DSC) but not
    for longer words that merely start with it (TGA_residue, Tmax)."""
    return re.match(rf"{symbol}(?![a-z])", name.lower().strip()) is not None


def _celsius(prop) -> float:
    """Property temperature in °C (values reported in kelvin are converted)."""
    unit = (prop.unit or "").strip()
    if unit in ("K", "kelvin", "Kelvin"):
        return round(prop.value - 273.15, 2)
    return prop.value


def _check_thermal_properties(sample: PolymerSample) -> list[ValidationIssue]:
    """Check Tg < Tm < Td if they exist."""
    flags = []

    tg_values = [_celsius(p) for p in sample.properties
                 if _is_thermal_symbol(p.property_name, "tg") and p.value is not None]
    tm_values = [_celsius(p) for p in sample.properties
                 if _is_thermal_symbol(p.property_name, "tm") and p.value is not None]

    if tg_values and tm_values:
        tg = tg_values[0]
        tm = tm_values[0]
        if tg >= tm:
            flags.append(ValidationIssue(
                TG_NOT_BELOW_TM,
                f"Tg ({tg} C) >= Tm ({tm} C): glass transition should be below melting point",
            ))

    # Tm < Td check (thermal hierarchy)
    td_values = [_celsius(p) for p in sample.properties
                 if _is_thermal_symbol(p.property_name, "td") and p.value is not None]
    if tm_values and td_values:
        tm = tm_values[0]
        td = td_values[0]
        if tm >= td:
            flags.append(ValidationIssue(
                TM_NOT_BELOW_TD,
                f"Tm ({tm} C) >= Td ({td} C): melting should be below decomposition",
            ))

    return flags


def _check_composition(sample: PolymerSample) -> list[ValidationIssue]:
    """Check that mole fractions sum to ~1.0."""
    flags = []

    mole_fractions = [
        m.mole_fraction for m in sample.constituents
        if m.mole_fraction is not None
    ]

    if len(mole_fractions) >= 2:
        total = sum(mole_fractions)
        if abs(total - 1.0) > 0.05:
            flags.append(ValidationIssue(
                MOLE_FRACTION_SUM,
                f"Mole fractions sum to {total:.3f}, expected ~1.0",
            ))

    weight_fractions = [
        m.weight_fraction for m in sample.constituents
        if m.weight_fraction is not None
    ]

    if len(weight_fractions) >= 2:
        total = sum(weight_fractions)
        if abs(total - 1.0) > 0.05:
            flags.append(ValidationIssue(
                WEIGHT_FRACTION_SUM,
                f"Weight fractions sum to {total:.3f}, expected ~1.0",
            ))

    # Single monomer with fraction != 1.0 is suspicious for homopolymers
    if len(mole_fractions) == 1 and mole_fractions[0] < 0.95:
        if sample.topology and sample.topology.lower() == "homopolymer":
            flags.append(ValidationIssue(
                HOMOPOLYMER_FRACTION,
                f"Single monomer with mole_fraction = {mole_fractions[0]:.3f} "
                f"for {sample.topology} topology — expected ~1.0",
            ))

    return flags


def _check_temperature_ranges(sample: PolymerSample) -> list[ValidationIssue]:
    """Check that temperatures are in reasonable ranges."""
    flags = []

    for prop in sample.properties:
        if classify_property(prop.property_name) == "thermal" and prop.value is not None:
            celsius = _celsius(prop)
            if celsius < -200 or celsius > 1000:
                flags.append(ValidationIssue(
                    THERMAL_OUT_OF_RANGE,
                    f"{prop.property_name} = {prop.value} {'K' if celsius != prop.value else 'C'} "
                    "is outside reasonable range (-200 to 1000 C)",
                ))

    # Synthesis temperature
    if sample.synthesis.temp_c is not None:
        if sample.synthesis.temp_c < -200 or sample.synthesis.temp_c > 1000:
            flags.append(ValidationIssue(
                SYNTHESIS_TEMP_OUT_OF_RANGE,
                f"Synthesis temperature {sample.synthesis.temp_c} C is outside reasonable range",
            ))

    return flags


def _check_smiles_validity(sample: PolymerSample) -> list[ValidationIssue]:
    """Check SMILES strings on monomers for validity."""
    try:
        from rdkit import Chem
    except ImportError:
        return []

    flags = []
    for i, monomer in enumerate(sample.constituents):
        label = monomer.name or monomer.abbreviation or f"constituent[{i}]"

        if not monomer.smiles:
            flags.append(ValidationIssue(SMILES_EMPTY, f"SMILES_EMPTY: no SMILES provided for {label}"))
            continue

        mol = Chem.MolFromSmiles(monomer.smiles)
        if mol is None:
            flags.append(ValidationIssue(
                SMILES_INVALID,
                f"SMILES_INVALID: '{monomer.smiles}' for {label} is not a valid SMILES string",
            ))

    return flags


def canonicalize_smiles(sample: PolymerSample) -> PolymerSample:
    """Replace valid SMILES with their canonical form. Invalid/empty SMILES are left untouched.

    Returns the sample for chaining.
    """
    try:
        from rdkit import Chem
    except ImportError:
        return sample

    for monomer in sample.constituents:
        if not monomer.smiles:
            continue
        mol = Chem.MolFromSmiles(monomer.smiles)
        if mol is not None:
            monomer.smiles = Chem.MolToSmiles(mol, canonical=True)

    return sample

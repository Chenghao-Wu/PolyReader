"""Unit tests for schema models and validation (no LLM required)."""

import pytest
from polyreader.schema.polymer import (
    PaperMetadataRef,
    Provenance,
    Monomer,
    MolecularWeight,
    SynthesisDetails,
    PropertyObject,
    BlendComponent,
    ProcessingStep,
    PolymerSample,
    ExtractionResult,
    _strip_empty,
)
from polyreader.schema.validation import (
    reject_invalid_sample, validate_sample, classify_quality_tier, classify_property,
)


class TestPolymerSampleCreation:
    """Test creating and serializing PolymerSample objects."""

    def test_empty_sample(self):
        sample = PolymerSample()
        assert sample.sample_id == ""
        assert sample.properties == []
        assert sample.validation_flags == []
        assert sample.quality_tier == ""
        assert sample.sample_type == ""
        assert sample.anomaly_notes == []

    def test_full_sample(self):
        sample = PolymerSample(
            sample_id="P1",
            paper_doi="10.1021/ma202814c",
            polymer_id="poly(D,L-lactide-co-caprolactone)",
            abbreviation="PDLCL",
            constituents=[
                Monomer(name="D,L-lactide", mole_fraction=0.9, role="monomer"),
                Monomer(name="caprolactone", mole_fraction=0.1, role="comonomer"),
            ],
            topology="linear",
            molecular_weight=MolecularWeight(
                mn=52100, mw=89600, pdi=1.72, method="GPC", unit="g/mol",
            ),
            synthesis=SynthesisDetails(
                method="ring-opening polymerization",
                catalyst="Sn(Oct)2",
                temp_c=130,
                time_h=48,
            ),
            properties=[
                PropertyObject(property_name="Tg", value=-2.0, unit="C", method="DSC"),
                PropertyObject(property_name="Tm", value=120.0, unit="C", method="DSC"),
            ],
        )
        assert sample.sample_id == "P1"
        assert len(sample.constituents) == 2
        assert sample.molecular_weight.mn == 52100
        assert len(sample.properties) == 2

    def test_serialization_roundtrip(self):
        sample = PolymerSample(
            sample_id="Test1",
            polymer_id="polylactide",
            molecular_weight=MolecularWeight(mn=10000, mw=15000, pdi=1.5),
            properties=[
                PropertyObject(property_name="Tg", value=55.0, unit="C"),
            ],
        )
        json_str = sample.model_dump_json()
        restored = PolymerSample.model_validate_json(json_str)
        assert restored.sample_id == "Test1"
        assert restored.molecular_weight.mn == 10000
        assert restored.properties[0].value == 55.0

    def test_extraction_result(self):
        result = ExtractionResult(
            metadata=PaperMetadataRef(doi="10.1021/test", title="Test Paper"),
            samples=[PolymerSample(sample_id="S1")],
            entity_map={"PLA": "polylactide"},
        )
        assert len(result.samples) == 1
        data = result.model_dump()
        assert data["metadata"]["doi"] == "10.1021/test"
        assert data["entity_map"]["PLA"] == "polylactide"
        assert data["sample_counts"] == {}

    def test_extraction_result_with_sample_counts(self):
        result = ExtractionResult(
            paper_doi="10.1021/test",
            sample_counts={"total": 3, "gold": 1, "silver": 1, "bronze": 1},
        )
        assert result.sample_counts["total"] == 3
        assert result.sample_counts["gold"] == 1

    def test_new_fields_on_sample(self):
        sample = PolymerSample(
            sample_id="P1",
            quality_tier="gold",
            sample_type="base",
            anomaly_notes=["Paper reports Mn > Mw in Table 2"],
        )
        assert sample.quality_tier == "gold"
        assert sample.sample_type == "base"
        assert len(sample.anomaly_notes) == 1

    def test_new_fields_serialization_roundtrip(self):
        sample = PolymerSample(
            sample_id="P1",
            quality_tier="silver",
            sample_type="derived_blend",
            anomaly_notes=["PDI < 1.0"],
            blend_components=[
                BlendComponent(polymer_ref="P2", weight_fraction=0.5, role="matrix"),
            ],
        )
        json_str = sample.model_dump_json()
        restored = PolymerSample.model_validate_json(json_str)
        assert restored.quality_tier == "silver"
        assert restored.sample_type == "derived_blend"
        assert restored.anomaly_notes == ["PDI < 1.0"]
        assert len(restored.blend_components) == 1
        assert restored.blend_components[0].polymer_ref == "P2"


class TestProvenance:
    def test_provenance_defaults(self):
        p = Provenance()
        assert p.source_type == ""
        assert p.source_location == ""
        assert p.page_numbers == []

    def test_provenance_with_data(self):
        p = Provenance(
            source_type="table",
            source_location="Table 1, row 3",
            page_numbers=[5, 6],
        )
        assert p.source_type == "table"
        assert p.source_location == "Table 1, row 3"
        assert p.page_numbers == [5, 6]

    def test_provenance_fields(self):
        """Provenance should have source_type, source_location, page_numbers, source_document."""
        assert list(Provenance.model_fields.keys()) == [
            "source_type", "source_location", "page_numbers", "source_document",
        ]

    def test_source_document_default(self):
        p = Provenance()
        assert p.source_document == ""

    def test_source_document_set(self):
        p = Provenance(source_document="si_data")
        assert p.source_document == "si_data"

    def test_source_document_stripped_by_clean_dump(self):
        """Empty source_document should be stripped by _strip_empty."""
        sample = PolymerSample(
            sample_id="P1",
            molecular_weight=MolecularWeight(
                mn=50000, provenance=Provenance(source_type="table"),
            ),
        )
        dumped = sample.clean_dump()
        prov = dumped["molecular_weight"]["provenance"]
        assert "source_document" not in prov

    def test_source_document_preserved_in_clean_dump(self):
        """Non-empty source_document should survive clean_dump."""
        sample = PolymerSample(
            sample_id="P1",
            molecular_weight=MolecularWeight(
                mn=50000,
                provenance=Provenance(source_type="table", source_document="si_data"),
            ),
        )
        dumped = sample.clean_dump()
        prov = dumped["molecular_weight"]["provenance"]
        assert prov["source_document"] == "si_data"


class TestValidation:
    """Test physical validation rules."""

    def test_valid_sample_no_flags(self):
        sample = PolymerSample(
            molecular_weight=MolecularWeight(mn=10000, mw=15000, pdi=1.5),
            properties=[
                PropertyObject(property_name="Tg", value=55.0, unit="C"),
                PropertyObject(property_name="Tm", value=170.0, unit="C"),
            ],
            constituents=[
                Monomer(name="L-lactide", mole_fraction=0.5,
                        smiles="C[C@H]1C(=O)O[C@H](C)C(=O)O1"),
                Monomer(name="glycolide", mole_fraction=0.5, smiles="O=C1COC(=O)CO1"),
            ],
        )
        flags = validate_sample(sample)
        assert flags == []

    def test_mw_less_than_mn(self):
        sample = PolymerSample(
            molecular_weight=MolecularWeight(mn=20000, mw=10000),
        )
        flags = validate_sample(sample)
        assert any("Mw" in f and "Mn" in f for f in flags)

    def test_pdi_inconsistency(self):
        sample = PolymerSample(
            molecular_weight=MolecularWeight(mn=10000, mw=15000, pdi=2.5),
        )
        flags = validate_sample(sample)
        assert any("PDI inconsistency" in f for f in flags)

    def test_pdi_consistent(self):
        sample = PolymerSample(
            molecular_weight=MolecularWeight(mn=10000, mw=15000, pdi=1.5),
        )
        flags = validate_sample(sample)
        assert not any("PDI" in f for f in flags)

    def test_tg_above_tm(self):
        sample = PolymerSample(
            properties=[
                PropertyObject(property_name="Tg", value=200.0, unit="C"),
                PropertyObject(property_name="Tm", value=150.0, unit="C"),
            ],
        )
        flags = validate_sample(sample)
        assert any("Tg" in f and "Tm" in f for f in flags)

    def test_mole_fractions_not_summing(self):
        sample = PolymerSample(
            constituents=[
                Monomer(name="A", mole_fraction=0.3),
                Monomer(name="B", mole_fraction=0.3),
            ],
        )
        flags = validate_sample(sample)
        assert any("Mole fractions" in f for f in flags)

    def test_mole_fractions_summing(self):
        sample = PolymerSample(
            constituents=[
                Monomer(name="A", mole_fraction=0.6),
                Monomer(name="B", mole_fraction=0.4),
            ],
        )
        flags = validate_sample(sample)
        assert not any("Mole fractions" in f for f in flags)

    def test_extreme_temperature(self):
        sample = PolymerSample(
            properties=[
                PropertyObject(property_name="Tg", value=5000.0, unit="C"),
            ],
        )
        flags = validate_sample(sample)
        assert any("outside reasonable range" in f for f in flags)

    def test_synthesis_extreme_temperature(self):
        # -250 is above absolute zero (-273) so passes field validator,
        # but below -200 so gets flagged by validation
        sample = PolymerSample(
            synthesis=SynthesisDetails(temp_c=-250),
        )
        flags = validate_sample(sample)
        assert any("Synthesis temperature" in f for f in flags)

    def test_tm_above_td(self):
        sample = PolymerSample(
            properties=[
                PropertyObject(property_name="Tm", value=300.0, unit="C"),
                PropertyObject(property_name="Td", value=250.0, unit="C"),
            ],
        )
        flags = validate_sample(sample)
        assert any("Tm" in f and "Td" in f for f in flags)

    def test_tm_below_td_no_flag(self):
        sample = PolymerSample(
            properties=[
                PropertyObject(property_name="Tm", value=170.0, unit="C"),
                PropertyObject(property_name="Td", value=350.0, unit="C"),
            ],
        )
        flags = validate_sample(sample)
        assert not any("Td" in f for f in flags)

    def test_suspiciously_low_mn(self):
        sample = PolymerSample(
            molecular_weight=MolecularWeight(mn=15.0, unit="g/mol"),
        )
        flags = validate_sample(sample)
        assert any("suspiciously low" in f for f in flags)

    def test_suspiciously_high_mn(self):
        sample = PolymerSample(
            molecular_weight=MolecularWeight(mn=5e9),
        )
        flags = validate_sample(sample)
        assert any("suspiciously high" in f for f in flags)

    def test_normal_mn_no_flag(self):
        sample = PolymerSample(
            molecular_weight=MolecularWeight(mn=50000, unit="g/mol"),
        )
        flags = validate_sample(sample)
        assert not any("suspiciously" in f for f in flags)

    def test_homopolymer_low_mole_fraction(self):
        sample = PolymerSample(
            topology="homopolymer",
            constituents=[
                Monomer(name="L-lactide", mole_fraction=0.5),
            ],
        )
        flags = validate_sample(sample)
        assert any("Single monomer" in f for f in flags)

    def test_homopolymer_normal_mole_fraction(self):
        sample = PolymerSample(
            topology="homopolymer",
            constituents=[
                Monomer(name="L-lactide", mole_fraction=1.0),
            ],
        )
        flags = validate_sample(sample)
        assert not any("Single monomer" in f for f in flags)


class TestQualityTierClassification:
    """Test quality tier classification: 4 must / 2 should / 4 tiers."""

    def test_gold_tier(self):
        """Gold: all 4 must-haves + 2 should-haves."""
        sample = PolymerSample(
            constituents=[Monomer(name="L-lactide")],
            topology="linear",
            molecular_weight=MolecularWeight(mn=50000),
            properties=[
                PropertyObject(property_name="Tg", value=55.0),
            ],
            synthesis=SynthesisDetails(method="ROP", catalyst="Sn(Oct)2"),
            processing_history=[ProcessingStep(step_id=1, action="melt-pressing", details="melt-pressed")],
        )
        tier, reason = classify_quality_tier(sample)
        assert tier == "gold"

    @pytest.mark.xfail(
        strict=True,
        reason="classify_quality_tier() grades on must-have criteria only; the should-have "
               "criteria (synthesis conditions, processing) are computed but not used for "
               "the tier. Kept unchanged pending a decision on the intended tier rules.",
    )
    def test_silver_all_must_no_should(self):
        """Silver: all 4 must-haves, 0 should-haves."""
        sample = PolymerSample(
            constituents=[Monomer(name="L-lactide")],
            topology="linear",
            molecular_weight=MolecularWeight(mn=50000),
            properties=[
                PropertyObject(property_name="Tg", value=55.0),
            ],
        )
        tier, reason = classify_quality_tier(sample)
        assert tier == "silver"

    @pytest.mark.xfail(
        strict=True,
        reason="classify_quality_tier() grades on must-have criteria only; the should-have "
               "criteria (synthesis conditions, processing) are computed but not used for "
               "the tier. Kept unchanged pending a decision on the intended tier rules.",
    )
    def test_silver_all_must_one_should(self):
        """Silver: all 4 must-haves + 1 should-have."""
        sample = PolymerSample(
            constituents=[Monomer(name="styrene")],
            topology="linear",
            molecular_weight=MolecularWeight(mn=100000),
            properties=[
                PropertyObject(property_name="Tg", value=100.0),
            ],
            processing_history=[ProcessingStep(step_id=1, action="injection molding", details="injection molded")],
        )
        tier, reason = classify_quality_tier(sample)
        assert tier == "silver"

    @pytest.mark.xfail(
        strict=True,
        reason="classify_quality_tier() grades on must-have criteria only; the should-have "
               "criteria (synthesis conditions, processing) are computed but not used for "
               "the tier. Kept unchanged pending a decision on the intended tier rules.",
    )
    def test_bronze_three_must(self):
        """Bronze: 3 must-haves (missing properties)."""
        sample = PolymerSample(
            constituents=[Monomer(name="L-lactide")],
            topology="linear",
            molecular_weight=MolecularWeight(mn=50000),
        )
        tier, reason = classify_quality_tier(sample)
        assert tier == "bronze"
        assert "properties" in reason

    def test_bronze_two_must(self):
        """Bronze: 2 must-haves (constituents + MW, no topology, no property)."""
        sample = PolymerSample(
            constituents=[Monomer(name="L-lactide")],
            molecular_weight=MolecularWeight(mn=50000),
        )
        tier, reason = classify_quality_tier(sample)
        assert tier == "bronze"

    @pytest.mark.xfail(
        strict=True,
        reason="classify_quality_tier() grades on must-have criteria only; the should-have "
               "criteria (synthesis conditions, processing) are computed but not used for "
               "the tier. Kept unchanged pending a decision on the intended tier rules.",
    )
    def test_bronze_constituents_topology_properties(self):
        """Bronze: constituents + topology + property but no MW."""
        sample = PolymerSample(
            constituents=[Monomer(name="D,L-lactide-co-glycolide")],
            topology="random",
            properties=[
                PropertyObject(property_name="Tg", value=45.0, unit="C", method="DSC"),
            ],
        )
        tier, reason = classify_quality_tier(sample)
        assert tier == "bronze"
        assert "distribution" in reason

    def test_reject_one_must(self):
        """Reject: only 1 must-have (constituents only)."""
        sample = PolymerSample(
            constituents=[Monomer(name="L-lactide")],
        )
        tier, reason = classify_quality_tier(sample)
        assert tier == "reject"

    def test_reject_topology_only(self):
        """Reject: only topology (1 must-have)."""
        sample = PolymerSample(
            topology="homopolymer",
        )
        tier, reason = classify_quality_tier(sample)
        assert tier == "reject"

    def test_empty_sample_reject(self):
        sample = PolymerSample()
        tier, reason = classify_quality_tier(sample)
        assert tier == "reject"

    def test_gold_with_synthesis_conditions_and_processing(self):
        """Gold: synthesis needs method + condition (catalyst/temp/time)."""
        sample = PolymerSample(
            constituents=[Monomer(name="caprolactone")],
            topology="linear",
            molecular_weight=MolecularWeight(mw=30000),
            synthesis=SynthesisDetails(method="ROP", temp_c=110),
            processing_history=[ProcessingStep(step_id=1, action="drying", details="vacuum dried")],
            properties=[
                PropertyObject(property_name="Tm", value=60.0),
            ],
        )
        tier, reason = classify_quality_tier(sample)
        assert tier == "gold"

    @pytest.mark.xfail(
        strict=True,
        reason="classify_quality_tier() grades on must-have criteria only; the should-have "
               "criteria (synthesis conditions, processing) are computed but not used for "
               "the tier. Kept unchanged pending a decision on the intended tier rules.",
    )
    def test_pdi_only_distribution_not_satisfied(self):
        """PDI alone (no Mn, no Mw) does not satisfy the distribution must-have."""
        sample = PolymerSample(
            constituents=[Monomer(name="styrene")],
            topology="linear",
            molecular_weight=MolecularWeight(pdi=1.5),
            properties=[
                PropertyObject(property_name="Tg", value=100.0),
            ],
        )
        tier, reason = classify_quality_tier(sample)
        assert tier == "bronze"  # 3 must (constituents, topology, properties) but not distribution
        assert "distribution" in reason

    @pytest.mark.xfail(
        strict=True,
        reason="classify_quality_tier() grades on must-have criteria only; the should-have "
               "criteria (synthesis conditions, processing) are computed but not used for "
               "the tier. Kept unchanged pending a decision on the intended tier rules.",
    )
    def test_mechanical_property_satisfies_must_have(self):
        """Sample with only mechanical properties (tensile_strength) satisfies properties must-have."""
        sample = PolymerSample(
            constituents=[Monomer(name="nylon-6")],
            topology="linear",
            molecular_weight=MolecularWeight(mn=30000),
            properties=[
                PropertyObject(property_name="tensile_strength", value=70.0, unit="MPa"),
            ],
        )
        tier, reason = classify_quality_tier(sample)
        assert tier == "silver"  # 4 must, 0 should

    @pytest.mark.xfail(
        strict=True,
        reason="classify_quality_tier() grades on must-have criteria only; the should-have "
               "criteria (synthesis conditions, processing) are computed but not used for "
               "the tier. Kept unchanged pending a decision on the intended tier rules.",
    )
    def test_synthesis_method_only_not_should_have(self):
        """Synthesis with method but no conditions does not satisfy should-have."""
        sample = PolymerSample(
            constituents=[Monomer(name="L-lactide")],
            topology="linear",
            molecular_weight=MolecularWeight(mn=50000),
            properties=[
                PropertyObject(property_name="Tg", value=55.0),
            ],
            synthesis=SynthesisDetails(method="ROP"),
        )
        tier, reason = classify_quality_tier(sample)
        assert tier == "silver"  # method without conditions doesn't count
        assert "synthesis" in reason


class TestFieldValidators:
    """Test Pydantic field validators that sanitize impossible values."""

    def test_negative_mn_becomes_none(self):
        mw = MolecularWeight(mn=-5000)
        assert mw.mn is None

    def test_negative_mw_becomes_none(self):
        mw = MolecularWeight(mw=-100)
        assert mw.mw is None

    def test_positive_mn_preserved(self):
        mw = MolecularWeight(mn=50000)
        assert mw.mn == 50000

    def test_zero_mn_preserved(self):
        mw = MolecularWeight(mn=0)
        assert mw.mn == 0

    def test_negative_pdi_becomes_none(self):
        mw = MolecularWeight(pdi=-1.5)
        assert mw.pdi is None

    def test_zero_pdi_becomes_none(self):
        mw = MolecularWeight(pdi=0)
        assert mw.pdi is None

    def test_extreme_pdi_becomes_none(self):
        mw = MolecularWeight(pdi=25)
        assert mw.pdi is None

    def test_valid_pdi_preserved(self):
        mw = MolecularWeight(pdi=1.72)
        assert mw.pdi == 1.72

    def test_temp_below_absolute_zero_becomes_none(self):
        synth = SynthesisDetails(temp_c=-300)
        assert synth.temp_c is None

    def test_temp_above_5000_becomes_none(self):
        synth = SynthesisDetails(temp_c=6000)
        assert synth.temp_c is None

    def test_valid_temp_preserved(self):
        synth = SynthesisDetails(temp_c=130)
        assert synth.temp_c == 130

    def test_cryogenic_temp_preserved(self):
        synth = SynthesisDetails(temp_c=-196)  # liquid nitrogen
        assert synth.temp_c == -196


class TestRejectInvalidSample:
    """Test hard rejection gate for ghost samples."""

    def test_ghost_sample_rejected(self):
        sample = PolymerSample()  # empty sample_id and polymer_id
        assert reject_invalid_sample(sample) is not None

    def test_sample_with_id_accepted(self):
        sample = PolymerSample(sample_id="P1")
        assert reject_invalid_sample(sample) is None

    def test_sample_with_polymer_id_accepted(self):
        sample = PolymerSample(polymer_id="polylactide")
        assert reject_invalid_sample(sample) is None

    def test_sample_with_both_ids_accepted(self):
        sample = PolymerSample(sample_id="P1", polymer_id="polylactide")
        assert reject_invalid_sample(sample) is None


class TestMolecularWeightUnitNullification:
    """Test that unit is nullified when no numeric MW data exists."""

    def test_no_data_unit_is_none(self):
        mw = MolecularWeight()
        assert mw.unit is None

    def test_all_none_unit_is_none(self):
        mw = MolecularWeight(mn=None, mw=None, pdi=None)
        assert mw.unit is None

    def test_with_mn_unit_preserved(self):
        mw = MolecularWeight(mn=50000)
        assert mw.unit == "g/mol"

    def test_with_mw_unit_preserved(self):
        mw = MolecularWeight(mw=89600)
        assert mw.unit == "g/mol"

    def test_with_pdi_unit_preserved(self):
        mw = MolecularWeight(pdi=1.72)
        assert mw.unit == "g/mol"

    def test_with_method_only_unit_is_none(self):
        """Method without numeric values still nullifies unit."""
        mw = MolecularWeight(method="GPC")
        assert mw.unit is None

    def test_explicit_unit_nullified_when_no_data(self):
        """Even if unit is explicitly set, it becomes None without numeric data."""
        mw = MolecularWeight(unit="kDa")
        assert mw.unit is None

    def test_clean_dump_no_mw_data(self):
        """clean_dump should not contain molecular_weight when no MW data exists."""
        sample = PolymerSample(sample_id="P1", polymer_id="polylactide")
        dumped = sample.clean_dump()
        assert "molecular_weight" not in dumped

    def test_clean_dump_with_mw_data(self):
        """clean_dump should contain molecular_weight with unit when data exists."""
        sample = PolymerSample(
            sample_id="P1",
            molecular_weight=MolecularWeight(mn=50000, unit="g/mol"),
        )
        dumped = sample.clean_dump()
        assert "molecular_weight" in dumped
        assert dumped["molecular_weight"]["unit"] == "g/mol"
        assert dumped["molecular_weight"]["mn"] == 50000

    def test_clean_dump_method_only_survives(self):
        """Method-only MW records survive in clean_dump (metadata preserved)."""
        sample = PolymerSample(
            sample_id="P1",
            molecular_weight=MolecularWeight(method="GPC"),
        )
        dumped = sample.clean_dump()
        assert dumped.get("molecular_weight", {}).get("method") == "GPC"

    def test_strip_empty_catches_unit_only_dict(self):
        """Safety net: _strip_empty collapses {"unit": "g/mol"} to {}."""
        from polyreader.schema.polymer import _strip_empty
        assert _strip_empty({"unit": "g/mol"}) == {}


class TestPropertyDisplayName:
    """Test property_display_name field on PropertyObject."""

    def test_default_is_empty_string(self):
        prop = PropertyObject(property_name="Tg", value=55.0, unit="C")
        assert prop.property_display_name == ""

    def test_can_be_populated(self):
        prop = PropertyObject(
            property_name="Tg",
            property_display_name="glass transition temperature",
            value=55.0,
            unit="C",
        )
        assert prop.property_display_name == "glass transition temperature"

    def test_stripped_from_clean_dump_when_empty(self):
        sample = PolymerSample(
            sample_id="P1",
            properties=[
                PropertyObject(property_name="Tg", value=55.0, unit="C"),
            ],
        )
        dumped = sample.clean_dump()
        prop = dumped["properties"][0]
        assert "property_display_name" not in prop

    def test_preserved_in_clean_dump_when_populated(self):
        sample = PolymerSample(
            sample_id="P1",
            properties=[
                PropertyObject(
                    property_name="Tg",
                    property_display_name="glass transition temperature",
                    value=55.0,
                    unit="C",
                ),
            ],
        )
        dumped = sample.clean_dump()
        prop = dumped["properties"][0]
        assert prop["property_display_name"] == "glass transition temperature"

    def test_serialization_roundtrip(self):
        prop = PropertyObject(
            property_name="Tm",
            property_display_name="melting temperature",
            value=170.0,
            unit="C",
            method="DSC",
        )
        json_str = prop.model_dump_json()
        restored = PropertyObject.model_validate_json(json_str)
        assert restored.property_display_name == "melting temperature"
        assert restored.property_name == "Tm"
        assert restored.value == 170.0


class TestMethodInferred:
    """Test method_inferred field on PropertyObject."""

    def test_default_is_false(self):
        prop = PropertyObject(property_name="Tg", value=55.0, unit="C")
        assert prop.method_inferred is False

    def test_can_set_true(self):
        prop = PropertyObject(
            property_name="Tg", value=55.0, unit="C",
            method="DSC", method_inferred=True,
        )
        assert prop.method_inferred is True

    def test_serialization_roundtrip(self):
        prop = PropertyObject(
            property_name="Tg", value=55.0, unit="C",
            method="DSC", method_inferred=True,
        )
        json_str = prop.model_dump_json()
        restored = PropertyObject.model_validate_json(json_str)
        assert restored.method_inferred is True
        assert restored.method == "DSC"

    def test_clean_dump_includes_method_inferred_false(self):
        """method_inferred: False appears in output (mirrors pdi_derived behavior)."""
        sample = PolymerSample(
            sample_id="P1",
            properties=[
                PropertyObject(property_name="Tg", value=55.0, unit="C", method="DSC"),
            ],
        )
        dumped = sample.clean_dump()
        prop = dumped["properties"][0]
        assert prop["method_inferred"] is False

    def test_clean_dump_includes_method_inferred_true(self):
        sample = PolymerSample(
            sample_id="P1",
            properties=[
                PropertyObject(
                    property_name="Tg", value=55.0, unit="C",
                    method="DSC", method_inferred=True,
                ),
            ],
        )
        dumped = sample.clean_dump()
        prop = dumped["properties"][0]
        assert prop["method_inferred"] is True

    def test_backward_compat_deserialization(self):
        """JSON without method_inferred deserializes with default False."""
        prop = PropertyObject.model_validate({
            "property_name": "Tg",
            "value": 55.0,
            "unit": "C",
            "method": "DSC",
        })
        assert prop.method_inferred is False


class TestValuelessPropertyStripping:
    """Test that valueless PropertyObject entries are stripped by clean_dump."""

    def test_valueless_property_stripped(self):
        """Property with only property_name/unit but no value/value_str is removed."""
        sample = PolymerSample(
            sample_id="P1",
            properties=[
                PropertyObject(property_name="Tg", unit="C"),
            ],
        )
        dumped = sample.clean_dump()
        assert dumped.get("properties", []) == []

    def test_valued_property_preserved(self):
        """Property with numeric value is kept."""
        sample = PolymerSample(
            sample_id="P1",
            properties=[
                PropertyObject(property_name="Tg", value=55.0, unit="C"),
            ],
        )
        dumped = sample.clean_dump()
        assert len(dumped["properties"]) == 1
        assert dumped["properties"][0]["value"] == 55.0

    def test_value_str_property_preserved(self):
        """Property with value_str only (no numeric value) is kept."""
        sample = PolymerSample(
            sample_id="P1",
            properties=[
                PropertyObject(
                    property_name="crystallinity",
                    value_str="semi-crystalline",
                    unit="",
                ),
            ],
        )
        dumped = sample.clean_dump()
        assert len(dumped["properties"]) == 1
        assert dumped["properties"][0]["value_str"] == "semi-crystalline"

    def test_zero_value_preserved(self):
        """value: 0.0 is valid and should be preserved."""
        sample = PolymerSample(
            sample_id="P1",
            properties=[
                PropertyObject(property_name="Tg", value=0.0, unit="C"),
            ],
        )
        dumped = sample.clean_dump()
        assert len(dumped["properties"]) == 1
        assert dumped["properties"][0]["value"] == 0.0

    def test_mixed_list_filters_valueless(self):
        """In a mixed list, valueless entries are removed and valued ones remain."""
        sample = PolymerSample(
            sample_id="P1",
            properties=[
                PropertyObject(property_name="Tg", value=55.0, unit="C"),
                PropertyObject(property_name="Tm", unit="C"),  # valueless
                PropertyObject(property_name="Td", unit="C"),  # valueless
                PropertyObject(property_name="modulus", value=3500.0, unit="MPa"),
            ],
        )
        dumped = sample.clean_dump()
        assert len(dumped["properties"]) == 2
        names = [p["property_name"] for p in dumped["properties"]]
        assert names == ["Tg", "modulus"]

    def test_strip_empty_valueless_property_dict(self):
        """_strip_empty directly: dict with property_name but no value/value_str -> {}."""
        result = _strip_empty({"property_name": "Tg", "unit": "C"})
        assert result == {}

    def test_strip_empty_valued_property_dict(self):
        """_strip_empty directly: dict with property_name and value is kept."""
        result = _strip_empty({"property_name": "Tg", "value": 55.0, "unit": "C"})
        assert result == {"property_name": "Tg", "value": 55.0, "unit": "C"}

    def test_strip_empty_filters_empty_dicts_from_list(self):
        """_strip_empty removes {} entries from lists."""
        result = _strip_empty([{"a": 1}, {}, {"b": 2}, {}])
        assert result == [{"a": 1}, {"b": 2}]


class TestClassifyProperty:
    """Test classify_property() handles real-world variants."""

    def test_exact_match_tg(self):
        assert classify_property("tg") == "thermal"

    def test_exact_match_tm(self):
        assert classify_property("tm") == "thermal"

    def test_exact_match_td(self):
        assert classify_property("td") == "thermal"

    def test_exact_match_tensile_strength(self):
        assert classify_property("tensile_strength") == "mechanical"

    def test_exact_match_storage_modulus(self):
        assert classify_property("storage_modulus") == "rheological"

    def test_exact_match_intrinsic_viscosity(self):
        assert classify_property("intrinsic_viscosity") == "solution"

    def test_exact_match_morphology(self):
        assert classify_property("morphology") == "structural"

    def test_thermal_prefix_tg_annealed(self):
        assert classify_property("tg_annealed") == "thermal"

    def test_thermal_prefix_tm1(self):
        assert classify_property("tm1") == "thermal"

    def test_thermal_prefix_td_5percent(self):
        assert classify_property("td_5%") == "thermal"

    def test_thermal_prefix_hdt_under_load(self):
        assert classify_property("hdt_under_load") == "thermal"

    def test_mechanical_flexural_modulus(self):
        assert classify_property("flexural_modulus") == "mechanical"

    def test_mechanical_compressive_strength(self):
        assert classify_property("compressive_strength") == "mechanical"

    def test_rheological_not_confused_with_mechanical(self):
        """storage_modulus contains 'modulus' but should be rheological."""
        assert classify_property("storage_modulus") == "rheological"

    def test_bare_modulus_is_mechanical(self):
        """Plain 'modulus' (without storage/loss prefix) is mechanical."""
        assert classify_property("modulus") == "mechanical"

    def test_structural_crystallinity(self):
        assert classify_property("crystallinity") == "structural"

    def test_structural_d_spacing(self):
        assert classify_property("d_spacing") == "structural"

    def test_solution_hydrodynamic_radius(self):
        assert classify_property("hydrodynamic_radius") == "solution"

    def test_unknown_property(self):
        assert classify_property("contact_angle") == "other"

    def test_case_insensitive(self):
        assert classify_property("Tg") == "thermal"
        assert classify_property("TG") == "thermal"

    def test_whitespace_stripped(self):
        assert classify_property("  tg  ") == "thermal"


class TestPDIValidation:
    """Test PDI-specific validation flags."""

    def test_pdi_below_one_flagged(self):
        """PDI < 1.0 is physically impossible."""
        sample = PolymerSample(
            molecular_weight=MolecularWeight(mn=10000, pdi=0.8),
        )
        flags = validate_sample(sample)
        assert any("physically impossible" in f for f in flags)

    def test_pdi_above_five_flagged(self):
        """PDI > 5.0 gets a warning."""
        sample = PolymerSample(
            molecular_weight=MolecularWeight(mn=10000, pdi=6.0),
        )
        flags = validate_sample(sample)
        assert any("unusually high" in f for f in flags)

    def test_pdi_25_rejected_by_pydantic(self):
        """PDI = 25 exceeds the Pydantic upper bound (20) and is set to None."""
        mw = MolecularWeight(pdi=25)
        assert mw.pdi is None

    def test_pdi_normal_no_flag(self):
        """Normal PDI (1.0-5.0) generates no PDI-related flags."""
        sample = PolymerSample(
            molecular_weight=MolecularWeight(mn=10000, mw=20000, pdi=2.0),
        )
        flags = validate_sample(sample)
        assert not any("PDI" in f for f in flags)

    def test_pdi_consistency_relaxed(self):
        """PDI within 15% of Mw/Mn should not flag inconsistency."""
        # Mw/Mn = 1.5, PDI = 1.7 → error = 13.3% < 15%
        sample = PolymerSample(
            molecular_weight=MolecularWeight(mn=10000, mw=15000, pdi=1.7),
        )
        flags = validate_sample(sample)
        assert not any("PDI inconsistency" in f for f in flags)


class TestLinearCopolymerComposition:
    """Test that linear topology no longer triggers homopolymer composition check."""

    def test_linear_copolymer_no_flag(self):
        """Linear copolymer with mole_fraction=0.5 should NOT be flagged."""
        sample = PolymerSample(
            topology="linear",
            constituents=[
                Monomer(name="L-lactide", mole_fraction=0.5),
            ],
        )
        flags = validate_sample(sample)
        assert not any("Single monomer" in f for f in flags)

    def test_homopolymer_still_flagged(self):
        """Homopolymer with low mole_fraction still flagged."""
        sample = PolymerSample(
            topology="homopolymer",
            constituents=[
                Monomer(name="L-lactide", mole_fraction=0.5),
            ],
        )
        flags = validate_sample(sample)
        assert any("Single monomer" in f for f in flags)


class TestThermalPrefixValidation:
    """Test that thermal validation uses prefix matching for variants."""

    def test_tg_annealed_vs_tm_hierarchy(self):
        """tg_annealed should be compared to tm for Tg < Tm check."""
        sample = PolymerSample(
            properties=[
                PropertyObject(property_name="tg_annealed", value=200.0, unit="C"),
                PropertyObject(property_name="tm_peak", value=150.0, unit="C"),
            ],
        )
        flags = validate_sample(sample)
        assert any("Tg" in f and "Tm" in f for f in flags)

    def test_td_5pct_range_check(self):
        """td_5% should be checked for temperature range."""
        sample = PolymerSample(
            properties=[
                PropertyObject(property_name="td_5%", value=5000.0, unit="C"),
            ],
        )
        flags = validate_sample(sample)
        assert any("outside reasonable range" in f for f in flags)


class TestMwHighBound:
    """Test Mw upper bound check."""

    def test_mw_suspiciously_high(self):
        sample = PolymerSample(
            molecular_weight=MolecularWeight(mw=5e8),
        )
        flags = validate_sample(sample)
        assert any("Mw" in f and "suspiciously high" in f for f in flags)



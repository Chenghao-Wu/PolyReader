"""Regression tests: validation codes, critical-flag retry, MW recovery, numeric edge cases."""

from __future__ import annotations

import math
from unittest.mock import AsyncMock, MagicMock, patch

import openai
import pytest

from polyreader.extraction.config import ExtractionConfig
from polyreader.extraction.extraction_loop import EntityExtractionOutcome, extract_single_entity
from polyreader.schema.polymer import MolecularWeight, PolymerSample, PropertyObject, SynthesisDetails
from polyreader.schema.validation import (
    MW_LESS_THAN_MN,
    PDI_INCONSISTENT,
    rescue_mw_from_properties,
    validate_sample,
    validate_sample_issues,
)


def _raw(mn=None, mw=None, pdi=None, **extra) -> dict:
    data = {
        "sample_id": "P1",
        "polymer_id": "poly(L-lactide)",
        "topology": "linear",
        "constituents": [{"name": "L-lactide", "smiles": "C[C@H]1C(=O)O[C@H](C)C(=O)O1"}],
        "molecular_weight": {"mn": mn, "mw": mw, "pdi": pdi},
        "properties": [{"property_name": "Tg", "value": 55.0, "unit": "C"}],
    }
    data.update(extra)
    return data


async def _run(side_effect, retries=3):
    with patch("polyreader.extraction.extraction_loop.extract_entity", new=AsyncMock(side_effect=side_effect)) as m, \
         patch("polyreader.extraction.extraction_loop.asyncio.sleep", new=AsyncMock()):
        outcome, sample, flags = await extract_single_entity(
            entity_id="PLLA", entity_info={"extraction_class": "base"}, entity_map={},
            doc_set=MagicMock(labels=MagicMock(return_value=["paper"])), llm=MagicMock(),
            config=ExtractionConfig(api_key="k"), max_retries=retries,
        )
    return outcome, sample, flags, m


# ---------------------------------------------------------------------------
# Validation codes
# ---------------------------------------------------------------------------


class TestValidationCodes:
    def test_codes_and_unchanged_messages(self):
        sample = PolymerSample.model_validate(_raw(mn=20000, mw=15000, pdi=2.0))
        issues = validate_sample_issues(sample)
        codes = {i.code for i in issues}
        assert MW_LESS_THAN_MN in codes and PDI_INCONSISTENT in codes
        # validate_sample keeps the existing human-readable output
        assert validate_sample(sample) == [i.message for i in issues]
        assert any(m.startswith("Mw (15000.0) < Mn (20000.0)") for m in validate_sample(sample))

    def test_tga_and_tmax_not_mistaken_for_tg_tm(self):
        sample = PolymerSample.model_validate(_raw(properties=[
            {"property_name": "TGA_residue", "value": 300.0, "unit": "%"},
            {"property_name": "Tm", "value": 170.0, "unit": "C"},
        ]))
        assert not any("Tg (" in m for m in validate_sample(sample))

    def test_kelvin_converted_for_thermal_order(self):
        sample = PolymerSample.model_validate(_raw(properties=[
            {"property_name": "Tg", "value": 330.0, "unit": "K"},  # 56.85 C
            {"property_name": "Tm", "value": 170.0, "unit": "C"},
        ]))
        assert not any("Tg (" in m for m in validate_sample(sample))


# ---------------------------------------------------------------------------
# Critical-flag retry (the old string match never fired)
# ---------------------------------------------------------------------------


class TestCriticalRetry:
    @pytest.mark.asyncio
    async def test_mw_less_than_mn_triggers_retry(self):
        outcome, sample, flags, m = await _run([_raw(20000, 15000), _raw(15000, 20000)])
        assert m.await_count == 2
        assert outcome == EntityExtractionOutcome.SUCCESS
        assert sample.molecular_weight.mn == 15000
        error_history = m.await_args_list[1].kwargs["error_history"]
        assert "Mw (15000.0) < Mn (20000.0)" in error_history[0]["errors"][0]

    @pytest.mark.asyncio
    async def test_inconsistent_pdi_triggers_retry(self):
        outcome, _, _, m = await _run([_raw(10000, 15000, 3.0), _raw(10000, 15000, 1.5)])
        assert m.await_count == 2 and outcome == EntityExtractionOutcome.SUCCESS

    @pytest.mark.asyncio
    async def test_valid_record_not_retried(self):
        outcome, sample, flags, m = await _run([_raw(10000, 15000, 1.5)])
        assert m.await_count == 1 and outcome == EntityExtractionOutcome.SUCCESS
        assert sample.quality_tier == "gold"

    @pytest.mark.asyncio
    async def test_missing_and_partial_mw_not_retried(self):
        _, _, _, m = await _run([_raw()])
        assert m.await_count == 1
        _, _, _, m = await _run([_raw(mn=12000)])
        assert m.await_count == 1

    @pytest.mark.asyncio
    async def test_last_attempt_keeps_flagged_result(self):
        outcome, sample, flags, m = await _run([_raw(20000, 15000)] * 3)
        assert m.await_count == 3
        assert outcome == EntityExtractionOutcome.SUCCESS
        assert any("Mw (" in f for f in flags)

    @pytest.mark.asyncio
    async def test_retry_failure_does_not_lose_earlier_result(self):
        """A retry that crashes must not discard the flagged-but-usable sample."""
        outcome, sample, flags, m = await _run([_raw(20000, 15000), ValueError("boom")])
        assert outcome == EntityExtractionOutcome.SUCCESS
        assert sample.molecular_weight.mn == 20000
        assert any("Mw (" in f for f in sample.validation_flags)

    @pytest.mark.asyncio
    async def test_transport_exhaustion_reported(self):
        err = openai.APIConnectionError(request=MagicMock())
        outcome, sample, flags, m = await _run([err, err, err])
        assert outcome == EntityExtractionOutcome.TRANSPORT_ERROR
        assert sample is None


# ---------------------------------------------------------------------------
# Molecular-weight recovery
# ---------------------------------------------------------------------------


def _sample_with(mw: dict, props: list[dict]) -> PolymerSample:
    return PolymerSample.model_validate({"molecular_weight": mw, "properties": props})


class TestMwRescue:
    def test_independent_field_recovery(self):
        """Mn already set: Mw and PDI in properties must still be recovered."""
        s = _sample_with({"mn": 10000}, [
            {"property_name": "Mw", "value": 15000, "unit": "g/mol"},
            {"property_name": "PDI", "value": 1.5},
            {"property_name": "Tg", "value": 50},
        ])
        rescue_mw_from_properties(s)
        assert (s.molecular_weight.mn, s.molecular_weight.mw, s.molecular_weight.pdi) == (10000, 15000, 1.5)
        assert [p.property_name for p in s.properties] == ["Tg"]

    def test_existing_value_not_overwritten_conflict_kept(self):
        s = _sample_with({"mn": 10000}, [{"property_name": "Mn", "value": 12000}])
        rescue_mw_from_properties(s)
        assert s.molecular_weight.mn == 10000
        assert [p.value for p in s.properties] == [12000]  # conflicting value kept for review

    def test_duplicate_removed(self):
        s = _sample_with({"mn": 10000}, [{"property_name": "Mn", "value": 10000}])
        rescue_mw_from_properties(s)
        assert s.properties == []

    def test_kda_converted(self):
        s = _sample_with({}, [{"property_name": "Mn", "value": 12.5, "unit": "kDa"},
                              {"property_name": "Mw", "value": 20, "unit": "kg/mol"}])
        rescue_mw_from_properties(s)
        assert s.molecular_weight.mn == 12500 and s.molecular_weight.mw == 20000
        assert s.molecular_weight.unit == "g/mol"

    def test_unknown_unit_not_moved(self):
        s = _sample_with({}, [{"property_name": "Mn", "value": 12.5, "unit": "mg"}])
        rescue_mw_from_properties(s)
        assert s.molecular_weight.mn is None and len(s.properties) == 1

    def test_original_full_rescue_still_works(self):
        s = _sample_with({}, [{"property_name": "Mn", "value": 10000, "method": "GPC"},
                              {"property_name": "dispersity", "value": 1.2}])
        rescue_mw_from_properties(s)
        assert s.molecular_weight.mn == 10000 and s.molecular_weight.pdi == 1.2
        assert s.molecular_weight.method == "GPC"


# ---------------------------------------------------------------------------
# Numeric edge cases (previously TypeError / whole-sample ValidationError)
# ---------------------------------------------------------------------------


class TestNumericInputs:
    @pytest.mark.parametrize("raw,expected", [
        ("12,300", 12300.0), ("12.3 kDa", 12300.0), ("1.2e4", 12000.0), (12000, 12000),
        ("n.d.", None), ("", None), (float("nan"), None),
    ])
    def test_mn_strings(self, raw, expected):
        assert MolecularWeight(mn=raw).mn == expected

    def test_pdi_string(self):
        assert MolecularWeight(pdi="1.25").pdi == 1.25
        assert MolecularWeight(pdi="1,25").pdi == 1.25  # decimal comma

    def test_temperature_and_time_strings(self):
        s = SynthesisDetails(temp_c="130 °C", time_h="overnight", yield_percent="85%")
        assert s.temp_c == 130 and s.time_h is None and s.yield_percent == 85

    def test_property_non_numeric_kept_as_text(self):
        p = PropertyObject(property_name="morphology", value="lamellar")
        assert p.value is None and p.value_str == "lamellar"
        r = PropertyObject(property_name="Tm", value="160-165", unit="C")
        assert r.value is None and r.value_str == "160-165"
        n = PropertyObject(property_name="Tg", value="55.2", unit="C")
        assert n.value == 55.2

    def test_whole_sample_with_messy_values_validates(self):
        sample = PolymerSample.model_validate(_raw(
            mn="15,000", mw="22 kDa", pdi="1.47",
            properties=[{"property_name": "Tm", "value": "n.d."}],
            processing_history=[{"step_id": "1a", "action": "drying"},
                                {"step_id": "first", "action": "annealing"}],
        ))
        assert sample.molecular_weight.mw == 22000
        assert [p.step_id for p in sample.processing_history] == [1, 0]
        assert math.isclose(sample.molecular_weight.pdi, 1.47)

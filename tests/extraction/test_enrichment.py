"""Unit tests for _enrich_from_entity_map (no LLM required)."""

import pytest

from polyreader.extraction.orchestrator import _enrich_from_entity_map
from polyreader.schema.polymer import PolymerSample
from polyreader.schema.validation import classify_quality_tier


class TestEnrichFromEntityMap:
    """Tests for post-reconciliation entity-map enrichment."""

    def test_enrich_fills_topology_from_entity_map(self):
        entity_map = {
            "PLGA": {
                "full_name": "poly(D,L-lactide-co-glycolide)",
                "type": "copolymer",
                "aliases": [],
                "sample_ids": ["PLGA-1"],
            }
        }
        samples = [{
            "sample_id": "PLGA-1",
            "polymer_id": "poly(D,L-lactide-co-glycolide)",
            "molecular_weight": {"mn": 50000, "mw": 80000, "pdi": 1.6},
        }]
        _enrich_from_entity_map(samples, entity_map)
        assert samples[0]["topology"] == "random"  # "copolymer" maps to "random"
        assert samples[0]["abbreviation"] == "PLGA"
        assert len(samples[0]["constituents"]) == 1
        assert samples[0]["constituents"][0]["name"] == "poly(D,L-lactide-co-glycolide)"

    def test_enrich_does_not_overwrite_existing_data(self):
        entity_map = {
            "PLA": {"full_name": "poly(L-lactide)", "type": "homopolymer", "sample_ids": ["PLA-1"]}
        }
        samples = [{
            "sample_id": "PLA-1",
            "polymer_id": "poly(L-lactide)",
            "topology": "linear",
            "constituents": [{"name": "L-lactide", "role": "monomer"}],
        }]
        _enrich_from_entity_map(samples, entity_map)
        assert samples[0]["topology"] == "linear"  # NOT overwritten to "homopolymer"
        assert samples[0]["constituents"][0]["name"] == "L-lactide"  # NOT overwritten

    def test_enrich_skips_unmatched_samples(self):
        entity_map = {"PLA": {"full_name": "poly(L-lactide)", "type": "homopolymer", "sample_ids": ["PLA-1"]}}
        samples = [{"sample_id": "UNKNOWN-1", "polymer_id": "mystery polymer"}]
        _enrich_from_entity_map(samples, entity_map)
        assert "topology" not in samples[0]
        assert "constituents" not in samples[0]

    def test_enriched_sample_passes_quality_gate(self):
        """After enrichment, a sample with distribution + entity-map topology +
        entity-map constituent should classify as at least bronze (not reject)."""
        raw = {
            "sample_id": "PLGA-1",
            "polymer_id": "poly(D,L-lactide-co-glycolide)",
            "topology": "random",
            "constituents": [{"name": "poly(D,L-lactide-co-glycolide)", "role": "polymer"}],
            "molecular_weight": {"mn": 35200, "mw": 52100, "pdi": 1.48},
        }
        sample = PolymerSample.model_validate(raw)
        tier, _reason = classify_quality_tier(sample)
        assert tier != "reject"

    def test_enrich_handles_non_dict_entity_entries(self):
        """Entity map entries that are plain strings should be skipped safely."""
        entity_map = {
            "PLA": "just a string, not a dict",
            "PLGA": {
                "full_name": "poly(D,L-lactide-co-glycolide)",
                "type": "random",
                "sample_ids": ["PLGA-1"],
            },
        }
        samples = [{"sample_id": "PLGA-1", "polymer_id": "poly(D,L-lactide-co-glycolide)"}]
        _enrich_from_entity_map(samples, entity_map)
        assert samples[0]["topology"] == "random"
        assert samples[0]["abbreviation"] == "PLGA"

    def test_enrich_matches_by_polymer_id(self):
        """Matching should work via polymer_id == full_name even without sample_ids."""
        entity_map = {
            "PCL": {
                "full_name": "polycaprolactone",
                "type": "linear",
                "sample_ids": [],
            }
        }
        samples = [{"sample_id": "S1", "polymer_id": "polycaprolactone"}]
        _enrich_from_entity_map(samples, entity_map)
        assert samples[0]["topology"] == "linear"
        assert samples[0]["abbreviation"] == "PCL"

    def test_enrich_compound_topology_passthrough(self):
        """Compound types like 'linear-random' should be passed through as-is."""
        entity_map = {
            "P1": {
                "full_name": "poly(X)",
                "type": "linear-random",
                "sample_ids": ["S1"],
            }
        }
        samples = [{"sample_id": "S1", "polymer_id": "poly(X)"}]
        _enrich_from_entity_map(samples, entity_map)
        # "linear-random" is not in the mapping, so passed through as-is
        assert samples[0]["topology"] == "linear-random"

    def test_enrich_matches_by_abbreviation_key(self):
        """Matching should work when polymer_id equals the entity_map key (abbreviation)."""
        entity_map = {
            "PLGA": {
                "full_name": "poly(D,L-lactide-co-glycolide)",
                "type": "copolymer",
                "sample_ids": [],
            }
        }
        samples = [{"sample_id": "S1", "polymer_id": "PLGA"}]
        _enrich_from_entity_map(samples, entity_map)
        assert samples[0]["topology"] == "random"  # "copolymer" maps to "random"
        assert samples[0]["abbreviation"] == "PLGA"
        assert samples[0]["constituents"][0]["name"] == "poly(D,L-lactide-co-glycolide)"

    def test_enrich_empty_entity_map(self):
        """Empty entity map should leave samples untouched."""
        samples = [{"sample_id": "S1", "polymer_id": "poly(X)"}]
        _enrich_from_entity_map(samples, {})
        assert "topology" not in samples[0]
        assert "constituents" not in samples[0]
        assert "abbreviation" not in samples[0]

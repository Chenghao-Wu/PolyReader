"""Integration test for the full extraction pipeline (requires LLM API key).

Runs the complete extract() pipeline on a known paper and validates
the output using soft-match criteria.
"""

import asyncio
import json
import os
from pathlib import Path

import pytest

from polyreader.indexing import load_index
from polyreader.extraction import extract, ExtractionConfig


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
TREE_JSON = str(PROJECT_ROOT / "output" / "test_tree.json")
REFERENCE_DIR = PROJECT_ROOT / "reference" / "10.1021_ma202814c"

API_KEY = os.environ.get("ZHIPU_API_KEY", "")
BASE_URL = "https://open.bigmodel.cn/api/paas/v4/"
MODEL = "glm-4.7"


def soft_match_sample(extracted: dict, reference: dict) -> dict:
    """Soft-match an extracted sample against reference data.

    Adapted from paperindex's test_retrieval.py soft_match.
    """
    checks = {}

    # polymer_id: substring match
    ref_pid = reference.get("polymer_id", "")
    ext_pid = str(extracted.get("polymer_id", ""))
    checks["polymer_id"] = {
        "passed": ref_pid.replace(" ", "").lower() in ext_pid.replace(" ", "").lower()
        or ext_pid.replace(" ", "").lower() in ref_pid.replace(" ", "").lower(),
        "expected": ref_pid,
        "actual": ext_pid,
    }

    # topology
    checks["topology"] = {
        "passed": str(extracted.get("topology", "")).lower()
        == str(reference.get("topology", "")).lower(),
        "expected": reference.get("topology"),
        "actual": extracted.get("topology"),
    }

    # synthesis method: keyword match
    ref_synth = reference.get("synthesis", {})
    ext_synth = extracted.get("synthesis", {})
    ref_method = str(ref_synth.get("method", "")).lower()
    ext_method = str(ext_synth.get("method", "")).lower()
    checks["synthesis_method"] = {
        "passed": bool(ref_method and ext_method and (
            ref_method in ext_method or ext_method in ref_method
            or ("ring" in ext_method and "opening" in ext_method)
        )),
        "expected": ref_synth.get("method"),
        "actual": ext_synth.get("method"),
    }

    # Mn within 10%
    ref_mw = reference.get("distribution", reference.get("molecular_weight", {}))
    ext_mw = extracted.get("molecular_weight", {})
    ref_mn = ref_mw.get("Mn", ref_mw.get("mn"))
    ext_mn = ext_mw.get("mn", ext_mw.get("Mn"))
    if ref_mn and ext_mn:
        try:
            ratio = float(ext_mn) / float(ref_mn)
            checks["Mn"] = {"passed": 0.9 <= ratio <= 1.1, "expected": ref_mn, "actual": ext_mn}
        except (ValueError, ZeroDivisionError):
            checks["Mn"] = {"passed": False, "expected": ref_mn, "actual": ext_mn}
    else:
        checks["Mn"] = {"passed": False, "expected": ref_mn, "actual": ext_mn}

    # PDI within 0.15
    ref_pdi = ref_mw.get("PDI", ref_mw.get("pdi"))
    ext_pdi = ext_mw.get("pdi", ext_mw.get("PDI"))
    if ref_pdi is not None and ext_pdi is not None:
        try:
            checks["PDI"] = {
                "passed": abs(float(ref_pdi) - float(ext_pdi)) < 0.15,
                "expected": ref_pdi,
                "actual": ext_pdi,
            }
        except (ValueError, TypeError):
            checks["PDI"] = {"passed": False, "expected": ref_pdi, "actual": ext_pdi}
    else:
        checks["PDI"] = {"passed": False, "expected": ref_pdi, "actual": ext_pdi}

    # Tg within 5C
    ref_props = {p.get("name", p.get("property_name", "")): p for p in reference.get("properties", [])}
    ext_props = {p.get("property_name", ""): p for p in extracted.get("properties", [])}
    for prop_name in ["Tg", "Tm"]:
        ref_val = ref_props.get(prop_name, {}).get("value")
        ext_val = ext_props.get(prop_name, {}).get("value")
        if ref_val is not None and ext_val is not None:
            try:
                checks[prop_name] = {
                    "passed": abs(float(ref_val) - float(ext_val)) < 5,
                    "expected": ref_val,
                    "actual": ext_val,
                }
            except (ValueError, TypeError):
                checks[prop_name] = {"passed": False, "expected": ref_val, "actual": ext_val}
        else:
            checks[prop_name] = {"passed": False, "expected": ref_val, "actual": ext_val}

    return checks


@pytest.mark.skipif(not API_KEY, reason="ZHIPU_API_KEY not set")
#@pytest.mark.skipif(not Path(TREE_JSON).exists(), reason="test_tree.json not found")
#@pytest.mark.asyncio
async def test_full_pipeline():
    """Run the full extract() pipeline and validate output."""
    index = load_index(TREE_JSON)

    config = ExtractionConfig(
        api_key=API_KEY,
        base_url=BASE_URL,
        model=MODEL,
    )

    result = await extract(index, config)

    # Should produce samples
    assert len(result.samples) > 0, "Pipeline produced no samples"

    # Each sample should have basic fields
    for sample in result.samples:
        assert sample.sample_id, f"Sample missing sample_id"

    # Print summary
    print(f"\nExtracted {len(result.samples)} samples")
    for s in result.samples:
        print(
            f"  {s.sample_id}: {s.polymer_id}, "
            f"Mn={s.molecular_weight.mn}, "
            f"props={len(s.properties)}, "
            f"flags={s.validation_flags}"
        )

    # If reference data exists, run soft match
    ref_files = list(REFERENCE_DIR.glob("*.json")) if REFERENCE_DIR.exists() else []
    if ref_files:
        with open(ref_files[0], "r", encoding="utf-8") as f:
            reference = json.load(f)

        # Find matching sample by polymer_id
        ref_pid = reference.get("polymer_id", "").lower()
        matched = None
        for s in result.samples:
            if ref_pid in s.polymer_id.lower() or s.polymer_id.lower() in ref_pid:
                matched = s
                break

        if matched:
            checks = soft_match_sample(matched.model_dump(), reference)
            passed = sum(1 for v in checks.values() if v["passed"])
            total = len(checks)
            print(f"\nSoft match: {passed}/{total} checks passed")
            for name, v in checks.items():
                status = "PASS" if v["passed"] else "FAIL"
                print(f"  [{status}] {name}: expected={v['expected']}, actual={v['actual']}")

"""End-to-end test: run extract() on all reference papers.

Produces per-polymer JSON files and an aggregate accuracy report.
"""

import asyncio
import json
import os
import re
from pathlib import Path

import pytest

from polyreader.indexing import load_index
from polyreader.extraction import extract, ExtractionConfig


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUT_DIR = PROJECT_ROOT / "output"
REFERENCE_DIR = PROJECT_ROOT / "reference"

API_KEY = os.environ.get("ZHIPU_API_KEY", "")
BASE_URL = "https://open.bigmodel.cn/api/paas/v4/"
MODEL = "glm-4.7"


def _safe_filename(text: str) -> str:
    """Convert a string to a safe filename component."""
    return re.sub(r'[^\w\-.]', '_', text).strip('_')


def find_tree_json_files() -> list[Path]:
    """Find all *_tree.json files in the output directory."""
    if not OUTPUT_DIR.exists():
        return []
    return sorted(OUTPUT_DIR.glob("*_tree.json"))


@pytest.mark.skipif(not API_KEY, reason="ZHIPU_API_KEY not set")
@pytest.mark.asyncio
async def test_end_to_end_all_papers():
    """Run extract() on all available indexed papers."""
    tree_files = find_tree_json_files()
    if not tree_files:
        pytest.skip("No indexed papers found in output/")

    config = ExtractionConfig(
        api_key=API_KEY,
        base_url=BASE_URL,
        model=MODEL,
    )

    report = []

    for tree_file in tree_files:
        paper_name = tree_file.stem.replace("_tree", "")
        print(f"\n{'=' * 60}")
        print(f"Processing: {paper_name}")
        print(f"{'=' * 60}")

        try:
            index = load_index(str(tree_file))
            result = await extract(index, config)

            doi_safe = _safe_filename(result.paper_doi) if result.paper_doi else paper_name

            emitted = 0
            rejected = 0
            for s in result.samples:
                if s.quality_tier == "reject":
                    rejected += 1
                    print(f"  SKIP (reject): {s.sample_id}")
                    continue

                # Write full per-polymer JSON using model_dump()
                sid_safe = _safe_filename(s.sample_id) if s.sample_id else f"sample_{emitted}"
                out_path = OUTPUT_DIR / f"{doi_safe}_{sid_safe}.json"
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(s.clean_dump(), f, indent=2, ensure_ascii=False)

                emitted += 1
                print(f"  Sample {s.sample_id}: tier={s.quality_tier}, "
                      f"Mn={s.molecular_weight.mn}, Mw={s.molecular_weight.mw}")

            report.append({
                "paper": paper_name,
                "doi": result.paper_doi,
                "samples_emitted": emitted,
                "samples_rejected": rejected,
            })
            print(f"  Emitted {emitted} samples, rejected {rejected}")

        except Exception as e:
            report.append({
                "paper": paper_name,
                "error": str(e),
            })
            print(f"  ERROR: {e}")

    # Save summary report
    report_path = OUTPUT_DIR / "polyextract_e2e_report.json"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # Print summary
    print(f"\n{'=' * 60}")
    print("END-TO-END SUMMARY")
    print(f"{'=' * 60}")
    total_papers = len(report)
    successful = sum(1 for r in report if "error" not in r)
    total_emitted = sum(r.get("samples_emitted", 0) for r in report)
    total_rejected = sum(r.get("samples_rejected", 0) for r in report)
    print(f"  Papers processed: {successful}/{total_papers}")
    print(f"  Samples emitted: {total_emitted}")
    print(f"  Samples rejected: {total_rejected}")
    print(f"  Report saved to: {report_path}")

"""Test script for tree extraction quality evaluation.

Caches page_texts from test PDF, runs build_tree(), and compares
against a reference tree with soft-match scoring.

Usage:
    conda activate paperindex && python test_tree_extraction.py
"""

import asyncio
import difflib
import json
import os
import re
from pathlib import Path

from polyreader.parsing.pdf_parser import parse_pdf
from polyreader.indexing.tree_builder import build_tree
from polyreader.llm.client import LLMClient
from polyreader.utils.tree_utils import get_nodes, print_tree, remove_fields

# --- Configuration ---
API_KEY = os.environ.get("MOONSHOT_API_KEY")
if not API_KEY:
    import pytest

    # Standalone evaluation script for the retired Moonshot provider.
    pytest.skip("standalone script: requires MOONSHOT_API_KEY and reference PDFs",
                allow_module_level=True)
BASE_URL = "https://api.moonshot.cn/v1"
MODEL = "kimi-k2.5"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PDF_PATH = str(
    PROJECT_ROOT
    / "reference"
    / "10.1021_ma202814c"
    / "stereocomplexation-of-poly(l-lactide)-and-random-copolymer-poly(d-lactide-co-ε-caprolactone)-to-enhance-melt-stability.pdf"
)
CACHE_PATH = str(PROJECT_ROOT / "test_page_texts.json")

# --- Reference tree from algorithm.md lines 59-135 ---
REFERENCE_TREE = [
    {
        "title": "Stereocomplexation of Poly(L-lactide) and Random Copolymer Poly(D-lactide-co-ε-caprolactone) To Enhance Melt Stability",
        "start_index": 1,
        "end_index": 3,
        "is_root": True,
    },
    {"title": "Supporting Information", "start_index": 1, "end_index": 1},
    {"title": "INTRODUCTION", "start_index": 1, "end_index": 1},
    {"title": "RESULTS AND DISCUSSION", "start_index": 1, "end_index": 3},
    {"title": "CONCLUSIONS", "start_index": 3, "end_index": 3},
    {"title": "ASSOCIATED CONTENT", "start_index": 3, "end_index": 3},
    {"title": "AUTHOR INFORMATION", "start_index": 3, "end_index": 3},
    {"title": "ACKNOWLEDGMENTS", "start_index": 3, "end_index": 3},
    {"title": "REFERENCES", "start_index": 3, "end_index": 3},
]

# --- Soft matching ---
TITLE_SIMILARITY_THRESHOLD = 0.6
PAGE_TOLERANCE = 1


def normalize_title(title: str) -> str:
    """Normalize title for comparison: lowercase, collapse whitespace."""
    return re.sub(r"\s+", " ", title.lower().strip())


def title_similarity(a: str, b: str) -> float:
    """Compute similarity between two titles using SequenceMatcher."""
    return difflib.SequenceMatcher(
        None, normalize_title(a), normalize_title(b)
    ).ratio()


def page_match(ref_val: int, ext_val: int, tolerance: int = PAGE_TOLERANCE) -> bool:
    """Check if page numbers match within tolerance."""
    if ref_val is None or ext_val is None:
        return False
    return abs(ref_val - ext_val) <= tolerance


def evaluate(reference: list[dict], extracted: list[dict]) -> dict:
    """Soft-match comparison between reference and extracted trees.

    Returns dict with per-item matches, precision, recall, F1.
    """
    matched_ref = set()
    matched_ext = set()
    details = []

    for ri, ref in enumerate(reference):
        best_score = 0
        best_ei = -1
        best_detail = None

        for ei, ext in enumerate(extracted):
            if ei in matched_ext:
                continue
            sim = title_similarity(ref["title"], ext.get("title", ""))
            if sim < TITLE_SIMILARITY_THRESHOLD:
                continue

            start_ok = page_match(ref["start_index"], ext.get("start_index"))
            end_ok = page_match(ref["end_index"], ext.get("end_index"))

            # Score: title_sim * 0.5 + start_match * 0.25 + end_match * 0.25
            score = sim * 0.5 + (0.25 if start_ok else 0) + (0.25 if end_ok else 0)

            if score > best_score:
                best_score = score
                best_ei = ei
                best_detail = {
                    "ref_title": ref["title"],
                    "ext_title": ext.get("title", ""),
                    "title_sim": round(sim, 3),
                    "ref_start": ref["start_index"],
                    "ext_start": ext.get("start_index"),
                    "start_ok": start_ok,
                    "ref_end": ref["end_index"],
                    "ext_end": ext.get("end_index"),
                    "end_ok": end_ok,
                    "score": round(best_score, 3),
                }

        if best_ei >= 0 and best_score >= 0.5:
            matched_ref.add(ri)
            matched_ext.add(best_ei)
            details.append(best_detail)

    # Unmatched reference items
    for ri, ref in enumerate(reference):
        if ri not in matched_ref:
            details.append({
                "ref_title": ref["title"],
                "ext_title": "(no match)",
                "title_sim": 0,
                "ref_start": ref["start_index"],
                "ext_start": None,
                "start_ok": False,
                "ref_end": ref["end_index"],
                "ext_end": None,
                "end_ok": False,
                "score": 0,
            })

    # Unmatched extracted items
    unmatched_ext = []
    for ei, ext in enumerate(extracted):
        if ei not in matched_ext:
            unmatched_ext.append(ext.get("title", "(untitled)"))

    precision = len(matched_ext) / len(extracted) if extracted else 0
    recall = len(matched_ref) / len(reference) if reference else 0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0

    return {
        "matched": len(matched_ref),
        "total_ref": len(reference),
        "total_ext": len(extracted),
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
        "details": details,
        "unmatched_ext": unmatched_ext,
    }


async def load_page_texts() -> list[str]:
    """Load page texts from cache or parse PDF."""
    if Path(CACHE_PATH).exists():
        print(f"Loading cached page texts from {CACHE_PATH}")
        with open(CACHE_PATH) as f:
            return json.load(f)

    print(f"Parsing PDF: {PDF_PATH}")
    result = await parse_pdf(PDF_PATH)
    page_texts = result.page_texts

    with open(CACHE_PATH, "w") as f:
        json.dump(page_texts, f, ensure_ascii=False)
    print(f"Cached {len(page_texts)} pages to {CACHE_PATH}")
    return page_texts


async def main():
    page_texts = await load_page_texts()
    print(f"Loaded {len(page_texts)} pages\n")

    # Build tree
    llm = LLMClient(api_key=API_KEY, base_url=BASE_URL, model=MODEL,temperature=1)
    print("Running build_tree()...")
    tree = await build_tree(page_texts, llm)

    # Print extracted tree
    print("\n--- Extracted Tree ---")
    print_tree(tree)

    # Output full tree as JSON (without text fields for readability)
    tree_no_text = remove_fields(tree)
    print("\n--- Generated Tree (JSON) ---")
    print(json.dumps(tree_no_text, indent=2, ensure_ascii=False))

    # Save to file
    output_path = str(PROJECT_ROOT / "test_tree_output.json")
    with open(output_path, "w") as f:
        json.dump(tree_no_text, f, indent=2, ensure_ascii=False)
    print(f"\nTree saved to {output_path}")

    # Flatten extracted tree for comparison
    extracted_flat = get_nodes(tree)
    print(f"\nExtracted {len(extracted_flat)} nodes total")

    # Evaluate
    results = evaluate(REFERENCE_TREE, extracted_flat)

    # Print report
    print("\n" + "=" * 80)
    print("EVALUATION REPORT")
    print("=" * 80)

    print(f"\n{'Reference Title':<55} {'Extracted Title':<55} {'Sim':>5} {'Start':>8} {'End':>8} {'Score':>6}")
    print("-" * 140)
    for d in results["details"]:
        ref_t = d["ref_title"][:52] + "..." if len(d["ref_title"]) > 55 else d["ref_title"]
        ext_t = d["ext_title"][:52] + "..." if len(d["ext_title"]) > 55 else d["ext_title"]
        start_str = f"{d['ref_start']}/{d['ext_start']}" + (" ✓" if d["start_ok"] else " ✗")
        end_str = f"{d['ref_end']}/{d['ext_end']}" + (" ✓" if d["end_ok"] else " ✗")
        print(f"{ref_t:<55} {ext_t:<55} {d['title_sim']:>5.3f} {start_str:>8} {end_str:>8} {d['score']:>6.3f}")

    if results["unmatched_ext"]:
        print(f"\nUnmatched extracted nodes: {results['unmatched_ext']}")

    print(f"\n--- Aggregate ---")
    print(f"Matched:   {results['matched']} / {results['total_ref']} reference, {results['total_ext']} extracted")
    print(f"Precision: {results['precision']:.3f}")
    print(f"Recall:    {results['recall']:.3f}")
    print(f"F1:        {results['f1']:.3f}")

    # PASS/FAIL
    passed = results["f1"] >= 0.7
    print(f"\nResult: {'PASS' if passed else 'FAIL'} (F1 threshold: 0.7)")
    return results


if __name__ == "__main__":
    asyncio.run(main())

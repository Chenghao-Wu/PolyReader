"""Test tree_builder using pre-extracted page contents JSON.

Bypasses slow PDF parsing — loads page texts from a page_contents JSON file,
runs build_tree(), and compares against a reference tree with soft-match scoring.

Usage:
    python tests/test_tree_builder.py my_output/ma3c00513_si_001_page_contents.json
    python tests/test_tree_builder.py page_contents.json --reference tree.json
    python tests/test_tree_builder.py page_contents.json -o test_output/ -v
    python tests/test_tree_builder.py page_contents.json --model deepseek-reasoner
"""

import argparse
import asyncio
import difflib
import json
import logging
import os
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from polyreader.indexing.tree_builder import build_tree
from polyreader.indexing.node_enrichment import enrich_nodes, generate_doc_description
from polyreader.llm.client import LLMClient
from polyreader.utils.tree_utils import get_nodes, print_tree, remove_fields, write_node_id

logger = logging.getLogger(__name__)

# --- Soft matching constants ---
TITLE_SIMILARITY_THRESHOLD = 0.6
PAGE_TOLERANCE = 1


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_page_texts(path: str) -> list[str]:
    """Load page_texts from page_contents JSON (no Marker needed).

    Expects JSON with ``page_contents`` list where each item has a ``text`` key.
    """
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    pages = data.get("page_contents", [])
    if not pages:
        raise ValueError(f"No page_contents found in {path}")

    page_texts = [p["text"] for p in pages]
    logger.info(f"Loaded {len(page_texts)} pages from {path}")
    return page_texts


def load_reference_tree(path: str) -> list[dict]:
    """Load reference tree JSON and flatten to node list."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    tree = data.get("tree", data)  # support both {tree: [...]} and bare list
    if isinstance(tree, dict):
        tree = [tree]

    flat = get_nodes(tree)
    logger.info(f"Loaded reference tree from {path}: {len(flat)} nodes")
    return flat


# ---------------------------------------------------------------------------
# Tree building
# ---------------------------------------------------------------------------

def create_llm(args: argparse.Namespace) -> LLMClient:
    """Create an LLM client from CLI args."""
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        print("Error: DEEPSEEK_API_KEY environment variable must be set.", file=sys.stderr)
        sys.exit(1)

    return LLMClient(
        api_key=api_key,
        base_url=args.base_url,
        model=args.model,
        temperature=1,
    )


async def run_tree_building(page_texts: list[str], llm: LLMClient, args: argparse.Namespace) -> list[dict]:
    """Build tree from page texts using LLM."""
    print(f"\n{'=' * 60}")
    print(f"BUILDING TREE  ({len(page_texts)} pages)")
    print(f"  model:    {args.model}")
    print(f"  base_url: {args.base_url}")
    print(f"{'=' * 60}")

    tree = await build_tree(page_texts, llm, max_token_num=args.max_token_num, max_page_num=args.max_page_num)
    return tree


# ---------------------------------------------------------------------------
# Evaluation (soft-match scoring)
# ---------------------------------------------------------------------------

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

            start_ok = page_match(ref.get("start_index"), ext.get("start_index"))
            end_ok = page_match(ref.get("end_index"), ext.get("end_index"))

            # Score: title_sim * 0.5 + start_match * 0.25 + end_match * 0.25
            score = sim * 0.5 + (0.25 if start_ok else 0) + (0.25 if end_ok else 0)

            if score > best_score:
                best_score = score
                best_ei = ei
                best_detail = {
                    "ref_title": ref["title"],
                    "ext_title": ext.get("title", ""),
                    "title_sim": round(sim, 3),
                    "ref_start": ref.get("start_index"),
                    "ext_start": ext.get("start_index"),
                    "start_ok": start_ok,
                    "ref_end": ref.get("end_index"),
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
                "ref_start": ref.get("start_index"),
                "ext_start": None,
                "start_ok": False,
                "ref_end": ref.get("end_index"),
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


def print_evaluation_report(results: dict, f1_threshold: float) -> bool:
    """Print evaluation table and PASS/FAIL. Returns True if passed."""
    print(f"\n{'=' * 60}")
    print("EVALUATION REPORT")
    print(f"{'=' * 60}")

    print(f"\n{'Reference Title':<50} {'Extracted Title':<50} {'Sim':>5} {'Start':>10} {'End':>10} {'Score':>6}")
    print("-" * 135)
    for d in results["details"]:
        ref_t = d["ref_title"][:47] + "..." if len(d["ref_title"]) > 50 else d["ref_title"]
        ext_t = d["ext_title"][:47] + "..." if len(d["ext_title"]) > 50 else d["ext_title"]
        start_str = f"{d['ref_start']}/{d['ext_start']}" + (" ok" if d["start_ok"] else " X")
        end_str = f"{d['ref_end']}/{d['ext_end']}" + (" ok" if d["end_ok"] else " X")
        print(f"{ref_t:<50} {ext_t:<50} {d['title_sim']:>5.3f} {start_str:>10} {end_str:>10} {d['score']:>6.3f}")

    if results["unmatched_ext"]:
        print(f"\nUnmatched extracted nodes ({len(results['unmatched_ext'])}):")
        for title in results["unmatched_ext"]:
            print(f"  - {title}")

    print(f"\n--- Aggregate ---")
    print(f"Matched:   {results['matched']} / {results['total_ref']} reference, {results['total_ext']} extracted")
    print(f"Precision: {results['precision']:.3f}")
    print(f"Recall:    {results['recall']:.3f}")
    print(f"F1:        {results['f1']:.3f}")

    passed = results["f1"] >= f1_threshold
    print(f"\nResult: {'PASS' if passed else 'FAIL'} (F1={results['f1']:.3f}, threshold={f1_threshold})")
    return passed


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

async def pipeline(args: argparse.Namespace) -> None:
    """Load page texts -> build tree -> save -> evaluate -> report."""
    input_path = Path(args.page_contents_json).resolve()
    if not input_path.exists():
        print(f"Error: file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    # Load page texts
    print(f"\n{'=' * 60}")
    print(f"INPUT: {input_path}")
    print(f"{'=' * 60}")
    page_texts = load_page_texts(str(input_path))
    print(f"Loaded {len(page_texts)} pages")

    # Create LLM client
    llm = create_llm(args)

    # Build tree
    tree = await run_tree_building(page_texts, llm, args)

    # Add node IDs
    write_node_id(tree)

    # Enrich nodes
    if not args.skip_enrichment:
        print("Enriching nodes with summaries...")
        await enrich_nodes(
            tree, llm, page_texts=page_texts,
            summary_token_threshold=args.summary_token_threshold,
            max_concurrent=args.max_concurrent,
        )

        # Generate doc description
        doc_description = await generate_doc_description(tree, llm)
        print(f"Doc description: {doc_description}")

    # Print extracted tree
    print(f"\n{'=' * 60}")
    print("EXTRACTED TREE")
    print(f"{'=' * 60}")
    print_tree(tree)

    # Save generated tree JSON
    output_dir = Path(args.output_dir).resolve() if args.output_dir else input_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    stem = input_path.stem.replace("_page_contents", "")
    output_path = output_dir / f"{stem}_tree_test.json"

    tree_no_text = remove_fields(tree)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(tree_no_text, f, indent=2, ensure_ascii=False)
    print(f"\nTree saved to {output_path}")

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test tree_builder using pre-extracted page contents JSON.",
    )
    parser.add_argument(
        "page_contents_json",
        help="Path to page contents JSON file (e.g. my_output/ma3c00513_si_001_page_contents.json)",
    )
    parser.add_argument(
        "--reference",
        default=None,
        help="Path to reference tree JSON for evaluation "
             "(default: auto-derived from input filename: *_page_contents.json -> *_tree.json)",
    )
    parser.add_argument(
        "--model",
        default="deepseek-chat",
        help="LLM model (default: deepseek-chat)",
    )
    parser.add_argument(
        "--base-url",
        default="https://api.deepseek.com",
        help="LLM base URL (default: https://api.deepseek.com)",
    )
    parser.add_argument(
        "-o", "--output-dir",
        default=None,
        help="Output directory for generated tree JSON (default: same dir as input)",
    )
    parser.add_argument(
        "--f1-threshold",
        type=float,
        default=0.7,
        help="Minimum F1 score for PASS (default: 0.7)",
    )
    parser.add_argument(
        "--max-token-num",
        type=int,
        default=20000,
        help="Max tokens per node before splitting (default: 20000)",
    )
    parser.add_argument(
        "--max-page-num",
        type=int,
        default=5,
        help="Max pages per node before splitting (default: 5)",
    )
    parser.add_argument(
        "--skip-enrichment",
        action="store_true",
        help="Skip node enrichment step (default: run enrichment)",
    )
    parser.add_argument(
        "--summary-token-threshold",
        type=int,
        default=200,
        help="Token threshold for raw text vs LLM summary (default: 200)",
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=3,
        help="Max concurrent LLM calls for enrichment (default: 3)",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable DEBUG logging",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )

    asyncio.run(pipeline(args))


if __name__ == "__main__":
    main()

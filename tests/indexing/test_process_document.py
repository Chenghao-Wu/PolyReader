"""End-to-end test for the refactored process_document pipeline using test.pdf."""

import asyncio
import json
import os
from pathlib import Path

import pytest

import polyreader

# Legacy standalone script: it needs the removed query() API, the
# retired Zhipu provider and a local test.pdf. Skip cleanly under pytest.
if not hasattr(polyreader, "query") or not os.environ.get("ZHIPU_API_KEY"):
    pytest.skip(
        "legacy script: requires the removed query() API, ZHIPU_API_KEY and test.pdf",
        allow_module_level=True,
    )

from polyreader.indexing import (  # noqa: E402
    process_document,
    query,
    print_tree,
    save_index,
    load_index,
)
from polyreader.utils.tree_utils import structure_to_list  # noqa: E402


# --- Configuration ---
API_KEY = os.environ.get("ZHIPU_API_KEY")
BASE_URL = "https://open.bigmodel.cn/api/paas/v4/"
MODEL = "glm-4.7"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PDF_PATH = str(PROJECT_ROOT / "test.pdf")
OUTPUT_DIR = PROJECT_ROOT / "output"
SAVE_PATH = str(OUTPUT_DIR / "test.json")  # save_index derives *_tree.json and *_page_contents.json
TREE_JSON = str(OUTPUT_DIR / "test_tree.json")
PAGES_JSON = str(OUTPUT_DIR / "test_page_contents.json")


def validate_tree_structure(tree):
    """Walk all nodes and validate field invariants.

    Returns list of (check_name, passed, detail) tuples.
    """
    results = []
    all_nodes = structure_to_list(tree)

    if not all_nodes:
        results.append(("tree_non_empty", False, "Tree has no nodes"))
        return results
    results.append(("tree_non_empty", True, f"{len(all_nodes)} nodes found"))

    # Classify nodes
    summary_errors = []
    text_errors = []
    id_errors = []

    for node in all_nodes:
        nid = node.get("node_id", "???")
        is_parent = bool(node.get("nodes"))

        # Parent nodes must have prefix_summary, must NOT have summary
        # Leaf nodes must have summary, must NOT have prefix_summary
        if is_parent:
            if "prefix_summary" not in node:
                summary_errors.append(f"{nid}: parent missing 'prefix_summary'")
            if "summary" in node:
                summary_errors.append(f"{nid}: parent has unexpected 'summary'")
        else:
            if "summary" not in node:
                summary_errors.append(f"{nid}: leaf missing 'summary'")
            if "prefix_summary" in node:
                summary_errors.append(f"{nid}: leaf has unexpected 'prefix_summary'")

        # No text field on any node (default config: if_add_node_text: "no")
        if "text" in node:
            text_errors.append(f"{nid}: unexpected 'text' field")

        # Every node must have node_id, start_index, end_index
        for field in ("node_id", "start_index", "end_index"):
            if field not in node:
                id_errors.append(f"{nid}: missing '{field}'")

    results.append((
        "summary_fields",
        len(summary_errors) == 0,
        f"{len(summary_errors)} errors" if summary_errors else "All nodes have correct summary fields",
    ))
    for e in summary_errors:
        results.append(("  summary_detail", False, e))

    results.append((
        "no_text_field",
        len(text_errors) == 0,
        f"{len(text_errors)} errors" if text_errors else "No 'text' fields found",
    ))

    results.append((
        "required_id_fields",
        len(id_errors) == 0,
        f"{len(id_errors)} errors" if id_errors else "All nodes have node_id, start_index, end_index",
    ))
    for e in id_errors:
        results.append(("  id_detail", False, e))

    return results


def validate_index_fields(index):
    """Check PaperIndex-level fields are populated."""
    results = []

    results.append((
        "page_texts_populated",
        bool(index.page_texts) and len(index.page_texts) > 0,
        f"{len(index.page_texts)} pages" if index.page_texts else "empty",
    ))
    results.append((
        "doc_name_populated",
        bool(index.doc_name),
        f"'{index.doc_name}'" if index.doc_name else "empty",
    ))
    results.append((
        "pdf_path_populated",
        bool(index.pdf_path),
        index.pdf_path if index.pdf_path else "empty",
    ))
    results.append((
        "no_references_attr",
        not hasattr(index, "references"),
        "confirmed no 'references' attribute",
    ))

    return results


def validate_round_trip(original, loaded):
    """Compare key fields between original and loaded index.

    The loaded index uses lazy loading, so page_texts starts empty.
    We verify that lazy loading populates it correctly.
    """
    results = []

    results.append((
        "tree_match",
        original.tree == loaded.tree,
        "trees match" if original.tree == loaded.tree else "trees differ",
    ))
    results.append((
        "doc_name_match",
        original.doc_name == loaded.doc_name,
        f"'{loaded.doc_name}'" if original.doc_name == loaded.doc_name else f"'{original.doc_name}' vs '{loaded.doc_name}'",
    ))

    # page_texts should be empty before lazy load
    results.append((
        "page_texts_empty_before_lazy_load",
        len(loaded.page_texts) == 0,
        "empty as expected" if len(loaded.page_texts) == 0 else f"unexpectedly has {len(loaded.page_texts)} pages",
    ))

    # Trigger lazy load via get_page_content
    loaded._lazy_load_pages()

    results.append((
        "page_texts_len_after_lazy_load",
        len(original.page_texts) == len(loaded.page_texts),
        f"{len(loaded.page_texts)} pages" if len(original.page_texts) == len(loaded.page_texts) else f"{len(original.page_texts)} vs {len(loaded.page_texts)}",
    ))
    results.append((
        "page_texts_content_match",
        original.page_texts == loaded.page_texts,
        "content matches" if original.page_texts == loaded.page_texts else "content differs",
    ))
    results.append((
        "pdf_path_match",
        original.pdf_path == loaded.pdf_path,
        "paths match" if original.pdf_path == loaded.pdf_path else f"'{original.pdf_path}' vs '{loaded.pdf_path}'",
    ))
    return results


async def main():
    all_results = []

    # --- Step 1: Index test.pdf ---
    print("=" * 60)
    print("Step 1: Indexing test.pdf")
    print("=" * 60)
    index = await process_document(
        PDF_PATH,
        api_key=API_KEY,
        base_url=BASE_URL,
        model=MODEL,
    )
    print()

    # --- Step 2: Print tree ---
    print("=" * 60)
    print("Step 2: Tree Structure")
    print("=" * 60)
    print_tree(index)
    print()

    # --- Step 3: Save index (produces tree.json + page_contents.json) ---
    print("=" * 60)
    print("Step 3: Saving outputs")
    print("=" * 60)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    save_index(index, SAVE_PATH)
    print()

    # --- Step 4: Validate tree structure and output files ---
    print("=" * 60)
    print("Step 4: Validating tree structure")
    print("=" * 60)
    tree_results = validate_tree_structure(index.tree)
    all_results.extend(tree_results)
    for name, passed, detail in tree_results:
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}: {detail}")

    # Validate output files exist
    tree_exists = Path(TREE_JSON).exists()
    pages_exists = Path(PAGES_JSON).exists()
    all_results.append(("tree_json_exists", tree_exists, TREE_JSON if tree_exists else "missing"))
    all_results.append(("page_contents_json_exists", pages_exists, PAGES_JSON if pages_exists else "missing"))
    print(f"  [{'PASS' if tree_exists else 'FAIL'}] tree_json_exists: {TREE_JSON}")
    print(f"  [{'PASS' if pages_exists else 'FAIL'}] page_contents_json_exists: {PAGES_JSON}")

    # Validate tree.json structure
    if tree_exists:
        with open(TREE_JSON, "r", encoding="utf-8") as f:
            tree_data = json.load(f)
        has_version = tree_data.get("version") == 2
        has_pages_file = "pages_file" in tree_data
        no_page_texts = "page_texts" not in tree_data
        all_results.append(("tree_version_2", has_version, f"version={tree_data.get('version')}"))
        all_results.append(("tree_has_pages_file", has_pages_file, tree_data.get("pages_file", "missing")))
        all_results.append(("tree_no_page_texts", no_page_texts, "no page_texts key" if no_page_texts else "unexpected page_texts key present"))
        no_metadata = "metadata" not in tree_data
        no_mode_used = "mode_used" not in tree_data
        all_results.append(("tree_no_metadata", no_metadata, "no metadata key" if no_metadata else "unexpected metadata key present"))
        all_results.append(("tree_no_mode_used", no_mode_used, "no mode_used key" if no_mode_used else "unexpected mode_used key present"))
        print(f"  [{'PASS' if has_version else 'FAIL'}] tree_version_2: version={tree_data.get('version')}")
        print(f"  [{'PASS' if has_pages_file else 'FAIL'}] tree_has_pages_file: {tree_data.get('pages_file', 'missing')}")
        print(f"  [{'PASS' if no_page_texts else 'FAIL'}] tree_no_page_texts: {'clean' if no_page_texts else 'has page_texts'}")
        print(f"  [{'PASS' if no_metadata else 'FAIL'}] tree_no_metadata: {'clean' if no_metadata else 'has metadata'}")
        print(f"  [{'PASS' if no_mode_used else 'FAIL'}] tree_no_mode_used: {'clean' if no_mode_used else 'has mode_used'}")

    # Validate page_contents.json structure
    if pages_exists:
        with open(PAGES_JSON, "r", encoding="utf-8") as f:
            pages_data = json.load(f)
        has_page_contents = "page_contents" in pages_data
        pages_count_match = pages_data.get("total_pages") == len(index.page_texts)
        all_results.append(("pages_has_page_contents", has_page_contents, "has page_contents array"))
        all_results.append(("pages_count_match", pages_count_match, f"{pages_data.get('total_pages')} pages"))
        print(f"  [{'PASS' if has_page_contents else 'FAIL'}] pages_has_page_contents")
        print(f"  [{'PASS' if pages_count_match else 'FAIL'}] pages_count_match: {pages_data.get('total_pages')} pages")
    print()

    # --- Step 5: Check PaperIndex fields ---
    print("=" * 60)
    print("Step 5: Checking PaperIndex fields")
    print("=" * 60)
    field_results = validate_index_fields(index)
    all_results.extend(field_results)
    for name, passed, detail in field_results:
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}: {detail}")
    print()

    # --- Step 6: Save/load round-trip (with lazy loading) ---
    print("=" * 60)
    print("Step 6: Save/load round-trip")
    print("=" * 60)
    loaded = load_index(TREE_JSON)
    rt_results = validate_round_trip(index, loaded)
    all_results.extend(rt_results)
    for name, passed, detail in rt_results:
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}: {detail}")
    print()

    # --- Step 7: Simple query ---
    print("=" * 60)
    print("Step 7: Simple query")
    print("=" * 60)
    question = "What is this paper about?"
    print(f"Q: {question}")
    result = await query(
        index,
        question,
        api_key=API_KEY,
        base_url=BASE_URL,
        model=MODEL,
    )
    print(f"A: {result}")
    query_ok = result is not None and len(str(result)) > 0
    all_results.append(("query_returns_result", query_ok, "got response" if query_ok else "empty response"))
    print()

    # --- Summary ---
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    passed = sum(1 for _, p, _ in all_results if p)
    failed = sum(1 for _, p, _ in all_results if not p)
    total = passed + failed
    print(f"  {passed}/{total} checks passed")
    if failed:
        print(f"  {failed} FAILED:")
        for name, p, detail in all_results:
            if not p:
                print(f"    - {name}: {detail}")
    else:
        print("  All checks passed!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())

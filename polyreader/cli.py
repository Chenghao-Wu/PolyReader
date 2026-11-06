"""CLI entry point for polyreader.

Usage:
    polyreader paper.pdf                        # index + extract single paper
    polyreader paper.pdf si.pdf                 # main paper + supplementary info
    polyreader paper.pdf --extract-only         # extract only (index must exist)
    polyreader paper.pdf --index-only           # index only
    polyreader paper.pdf --skip-index           # skip indexing if tree exists
    polyreader paper.pdf --provider deepseek    # DeepSeek API (DEEPSEEK_API_KEY)
    polyreader paper.pdf --pdf-parser mineru    # MinerU instead of Marker
"""

import argparse
import asyncio
import json
import logging
import os
import re
from datetime import datetime
import sys
import traceback
from pathlib import Path

from .indexing import PaperIndex, process_document, save_index, load_index, verify_index
from .llm import create_llm_client
from .llm.providers import (
    DEEPSEEK,
    DEFAULT_MODELS,
    REASONING_EFFORTS,
    api_key_env_var,
    api_key_from_env,
    resolve_provider,
)
from .utils.json_utils import write_json_atomic
from .extraction import extract, ExtractionConfig, DocumentSet
from .extraction.orchestrator import unique_name

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_filename(text: str) -> str:
    """Convert a string to a safe filename component."""
    return re.sub(r'[^\w\-.]', '_', text).strip('_')


def _derive_name(pdf_path: str) -> str:
    """Derive a clean base name from the PDF filename."""
    return Path(pdf_path).stem


def _model_output_subdir(model: str, base_url: str) -> str | None:
    """Return an output subdirectory label for non-default model providers."""
    model_lower = (model or "").lower()
    base_lower = (base_url or "").lower()

    if "openrouter.ai/api" in base_lower:
        return "openrouter"
    if "api.deepseek.com" in base_lower:
        return "deepseek"
    if model_lower.startswith("gemini"):
        return "gemini"

    return None


def _check_api_keys(*, model: str, base_url: str, index: bool, extraction: bool) -> None:
    """Validate required API keys are present before running anything."""
    env_var = api_key_env_var(base_url)
    if (index or extraction) and not os.environ.get(env_var):
        print(f"Error: {env_var} environment variable is required "
              f"for {base_url}.", file=sys.stderr)
        sys.exit(1)


def _index_reuse_problem(tree_path: Path, parser: str) -> str | None:
    """Why an existing index cannot be reused for this run (None if it can)."""
    problems = verify_index(str(tree_path))
    if problems:
        return "; ".join(problems)
    try:
        with open(tree_path, "r", encoding="utf-8") as f:
            built_with = (json.load(f).get("parser") or {}).get("backend", "marker")
    except (OSError, ValueError):
        return "unreadable index"
    if built_with != parser:
        return f"built with the '{built_with}' parser, '{parser}' requested"
    return None


# ---------------------------------------------------------------------------
# Pipeline phases
# ---------------------------------------------------------------------------

async def run_indexing(
    pdf_path: str,
    output_dir: Path,
    name: str,
    *,
    api_key: str | None = None,
    base_url: str | None = None,
    model: str | None = None,
    **kwargs,
) -> PaperIndex:
    """Index a PDF and save the result to disk."""
    print(f"\n{'=' * 60}")
    print(f"INDEXING: {pdf_path}")
    print(f"{'=' * 60}")

    index = await process_document(pdf_path, api_key=api_key, base_url=base_url, model=model, **kwargs)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_index(index, str(output_dir / f"{name}.json"))
    return index


async def run_extraction(
    index: PaperIndex | DocumentSet,
    output_dir: Path,
    name: str,
    config: ExtractionConfig,
) -> dict:
    """Run extraction on an index or DocumentSet, write per-sample JSONs, return a report entry."""
    print(f"\n{'=' * 60}")
    print(f"EXTRACTING: {name}")
    print(f"{'=' * 60}")

    result = await extract(index, config)

    doi_safe = _safe_filename(result.metadata.doi) if result.metadata.doi else name

    samples_dir = output_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    emitted = 0
    rejected = 0
    used_names: set[str] = set()
    sample_files: list[str] = []
    for s in result.samples:
        if s.quality_tier == "reject":
            rejected += 1
            print(f"  SKIP (reject): {s.sample_id} — {s.quality_tier_reason}")
            continue

        sid_safe = _safe_filename(s.sample_id) if s.sample_id else f"sample_{emitted}"
        # Distinct samples whose IDs sanitize alike get _2, _3, ... suffixes
        # instead of silently overwriting each other.
        out_path = samples_dir / f"{unique_name(f'{doi_safe}_{sid_safe}', used_names)}.json"
        write_json_atomic(out_path, s.clean_dump())
        sample_files.append(out_path.name)

        emitted += 1
        print(f"  Sample {s.sample_id}: tier={s.quality_tier}, "
              f"Mn={s.molecular_weight.mn}, Mw={s.molecular_weight.mw}, "
              f"PDI={s.molecular_weight.pdi} -> {out_path.name}")

    # Extract extraction summary from trace
    entity_discovery = next(
        (t for t in result.extraction_trace if t.get("agent") == "entity_discovery"),
        {},
    )
    base_wave = next(
        (t for t in result.extraction_trace if t.get("wave") == "base_entities"),
        {},
    )
    derived_wave = next(
        (t for t in result.extraction_trace if t.get("wave") == "derived_entities"),
        {},
    )
    llm_usage = next(
        (t for t in result.extraction_trace if t.get("agent") == "llm_usage"),
        None,
    )

    report = {
        "paper": name,
        "doi": result.metadata.doi,
        "samples_emitted": emitted,
        "samples_rejected": rejected,
        "sample_counts": result.sample_counts,
        "entities_found": entity_discovery.get("entities_found", 0),
        "base_entities_success": base_wave.get("success", 0),
        "derived_entities_success": derived_wave.get("success", 0),
        "sample_files": sample_files,
    }
    if llm_usage:
        report["llm_usage"] = {k: v for k, v in llm_usage.items() if k != "agent"}
    print(f"  Emitted {emitted} samples, rejected {rejected}")
    return report


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

async def pipeline(args: argparse.Namespace) -> None:
    """Orchestrate indexing and extraction based on CLI flags."""
    pdf_paths = [str(Path(p).resolve()) for p in args.pdf_paths]
    missing = [p for p in pdf_paths if not Path(p).is_file()]
    if missing:
        print(f"Error: PDF not found: {', '.join(missing)}", file=sys.stderr)
        sys.exit(1)
    base_output_dir = Path(args.output_dir).resolve()
    name = args.name or _derive_name(pdf_paths[0])
    model_subdir = _model_output_subdir(args.model, args.base_url)
    output_dir = base_output_dir / model_subdir / name if model_subdir else base_output_dir / name
    tree_path = output_dir / f"{name}_tree.json"

    do_index = not args.extract_only
    do_extract = not args.index_only

    def doc_tree_path(i: int) -> Path:
        return tree_path if i == 0 else output_dir / f"{name}_doc{i + 1}_tree.json"

    # With --skip-index, reuse every complete index built with the requested
    # parser; (re-)index only documents whose index is missing or broken.
    reusable: dict[int, Path] = {}
    if do_index and args.skip_index:
        for i in range(len(pdf_paths)):
            path_i = doc_tree_path(i)
            if not path_i.exists():
                continue
            problem = _index_reuse_problem(path_i, args.pdf_parser)
            if problem:
                print(f"Re-indexing {path_i.name}: {problem}")
            else:
                print(f"Tree file exists, skipping indexing: {path_i}")
                reusable[i] = path_i
        if len(reusable) == len(pdf_paths):
            do_index = False

    _check_api_keys(
        model=args.index_model if do_index else args.model,
        base_url=args.base_url,
        index=do_index,
        extraction=do_extract,
    )

    indices: list[PaperIndex] = []
    classification_error: str | None = None

    # --- Indexing phase ---
    if do_index:
        index_api_key = api_key_from_env(args.base_url)
        extra_index_kw = {
            "pdf_parser_backend": args.pdf_parser,
            "request_timeout": args.request_timeout,
            "reasoning_effort": args.reasoning_effort,
        }
        if args.pdf_parser == "mineru":
            extra_index_kw["mineru"] = {
                "tier": args.mineru_tier,
                "executable": args.mineru_executable,
                "timeout": args.mineru_timeout,
            }
        if args.max_concurrent_index is not None:
            extra_index_kw["max_concurrent_api_calls"] = args.max_concurrent_index
        for i, pdf_path in enumerate(pdf_paths):
            idx_name = name if i == 0 else f"{name}_doc{i + 1}"
            if i in reusable:
                indices.append(load_index(str(reusable[i]), pdf_path=pdf_path))
                continue
            idx = await run_indexing(
                pdf_path, output_dir, idx_name,
                api_key=index_api_key,
                base_url=args.base_url,
                model=args.index_model,
                parse_output_dir=str(output_dir / f".parse_{idx_name}"),
                **extra_index_kw,
            )
            indices.append(idx)

    # --- Extraction phase ---
    if do_extract:
        if not indices:
            # Load from disk (--extract-only or --skip-index)
            if not tree_path.exists():
                print(f"Error: tree file not found: {tree_path}", file=sys.stderr)
                print("Run indexing first or check --output-dir / --name.", file=sys.stderr)
                sys.exit(1)
            problems = verify_index(str(tree_path))
            if problems:
                print(f"Error: index {tree_path} is not usable: {'; '.join(problems)}", file=sys.stderr)
                print("Re-run indexing (without --extract-only).", file=sys.stderr)
                sys.exit(1)
            print(f"Loading existing index: {tree_path}")
            indices.append(load_index(str(tree_path), pdf_path=pdf_paths[0]))
            # Load additional doc trees if they exist
            for i, pdf_path in enumerate(pdf_paths[1:], start=2):
                extra_tree = output_dir / f"{name}_doc{i}_tree.json"
                if extra_tree.exists() and not verify_index(str(extra_tree)):
                    print(f"Loading additional index: {extra_tree}")
                    indices.append(load_index(str(extra_tree), pdf_path=pdf_path))
                else:
                    print(f"WARNING: no usable index for {Path(pdf_path).name} "
                          f"({extra_tree.name}); it is left out of this extraction.",
                          file=sys.stderr)

        # Build DocumentSet or pass single index
        if len(indices) == 1:
            extract_input: PaperIndex | DocumentSet = indices[0]
        elif not args.no_auto_detect:
            # Auto-detect main vs. supplementary via LLM classification
            from .extraction.doc_classifier import sort_documents, DocumentRole
            from .extraction.document_set import _sanitize_label

            classify_api_key = api_key_from_env(args.base_url)
            classify_llm = create_llm_client(
                api_key=classify_api_key,
                model=args.index_model,
                base_url=args.base_url,
                include_extra_body=not args.no_extra_body,
                timeout=args.request_timeout,
                reasoning_effort=args.reasoning_effort,
                thinking_disabled=args.no_thinking,
            )
            try:
                main_index, si_list, classifications = await sort_documents(indices, classify_llm)
            except Exception as e:
                # Don't lose the (saved) indexes over a classification failure:
                # fall back to the documented positional order and say so.
                classification_error = f"{type(e).__name__}: {e}"
                print(f"WARNING: document role classification failed ({classification_error}); "
                      f"using positional order (first PDF = main).", file=sys.stderr)
                main_index, si_list, classifications = indices[0], indices[1:], None

            if classifications is not None:
                # Print classification results for transparency
                print(f"\nDocument role classification:")
                for idx, cls in zip(indices, classifications):
                    marker = "-> MAIN" if cls.role == DocumentRole.MAIN else "   SI "
                    print(f"  {marker}  {idx.doc_name}  ({cls.confidence} confidence: {cls.reasoning})")
                print()

            # Build DocumentSet with correct ordering
            documents: dict[str, PaperIndex] = {}
            main_label = _sanitize_label(main_index.doc_name) if main_index.doc_name else "main"
            documents[main_label] = main_index
            for i, si_index in enumerate(si_list):
                label = "si" if len(si_list) == 1 else f"si_{i + 1}"
                documents[label] = si_index
            extract_input = DocumentSet(documents)
        else:
            extract_input = DocumentSet.from_indices(*indices)

        vision_enabled = not args.no_vision
        config = ExtractionConfig(
            base_url=args.base_url,
            model=args.model,
            vision_enabled=vision_enabled,
            thinking_enabled=args.thinking_enabled,
            include_extra_body=not args.no_extra_body,
            reasoning_effort=args.reasoning_effort,
            thinking_disabled=args.no_thinking,
            request_timeout=args.request_timeout,
            guided_max_iterations=10,
            max_concurrent_entities=args.max_concurrent,
            extraction_mode=args.extraction_mode,
            full_context_budget=args.full_context_budget,
            batch_size=args.batch_size,
            max_entity_retries=3,
            max_samples=args.max_samples,
        )

        try:
            report = await run_extraction(extract_input, output_dir, name, config)
        except Exception as e:
            tb = traceback.format_exc()
            report = {"paper": name, "error": str(e), "traceback": tb}
            print(f"  ERROR: {e}")
            print(f"  Traceback:\n{tb}")

        # Write per-paper pipeline report
        report["parser"] = indices[0].parser_info if indices and indices[0].parser_info else {}
        if classification_error:
            report["document_classification_error"] = classification_error
        report_path = output_dir / f"{name}_pipeline_report.json"
        write_json_atomic(report_path, report)
        print(f"\nPipeline report saved to: {report_path}")

    print("\nDone.")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        prog="polyreader",
        description="Extract structured polymer data from scientific papers.",
    )
    parser.add_argument(
        "pdf_paths",
        nargs="+",
        help="Path(s) to PDF file(s). First is main, rest are SI "
             "(unless --no-auto-detect is off, in which case roles are auto-classified).",
    )
    parser.add_argument(
        "-o", "--output-dir",
        default="output",
        help=(
            "Output root directory (default: output/). "
            "Non-default model providers are stored under model-specific subfolders, "
            "for example output/openrouter/<paper>/."
        ),
    )
    parser.add_argument(
        "--name",
        default=None,
        help="Override base name for output files (default: PDF stem)",
    )
    parser.add_argument(
        "--skip-index",
        action="store_true",
        help="Skip indexing if tree file already exists",
    )
    parser.add_argument(
        "--index-only",
        action="store_true",
        help="Only index, skip extraction",
    )
    parser.add_argument(
        "--extract-only",
        action="store_true",
        help="Only extract (tree file must exist)",
    )
    parser.add_argument(
        "--provider",
        choices=["auto", "openrouter", "deepseek"],
        default="auto",
        help="LLM provider. auto (default): OpenRouter if OPENROUTER_API_KEY is set, "
             "else the DeepSeek API if DEEPSEEK_API_KEY is set. Ignored when --base-url is given.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model name (default: deepseek/deepseek-reasoner on OpenRouter, "
             "deepseek-flash on the DeepSeek API)",
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help="API base URL (default: from --provider; https://openrouter.ai/api/v1 "
             "or https://api.deepseek.com)",
    )
    parser.add_argument(
        "--index-model",
        default=None,
        help="Model for indexing phase (default: same as --model)",
    )
    parser.add_argument(
        "--no-vision",
        action="store_true",
        help="Disable vision in extraction",
    )
    parser.add_argument(
        "--thinking-enabled",
        action="store_true",
        help="Enable thinking/reasoning mode",
    )
    parser.add_argument(
        "--no-thinking",
        action="store_true",
        help="DeepSeek API: run without thinking mode (faster, cheaper)",
    )
    parser.add_argument(
        "--reasoning-effort",
        choices=list(REASONING_EFFORTS),
        default=None,
        help="DeepSeek API: thinking effort (default: API default, high)",
    )
    parser.add_argument(
        "--request-timeout",
        type=float,
        default=None,
        help="Per-request LLM timeout in seconds (default: 600 for DeepSeek, 120 otherwise)",
    )
    parser.add_argument(
        "--pdf-parser",
        choices=["marker", "mineru"],
        default="marker",
        help="PDF parser backend (default: marker)",
    )
    parser.add_argument(
        "--mineru-tier",
        choices=["flash", "basic", "standard", "advanced"],
        default="standard",
        help="MinerU quality tier (default: standard)",
    )
    parser.add_argument(
        "--mineru-executable",
        default=None,
        help="Path to the mineru-kit executable (default: $POLYREADER_MINERU_EXECUTABLE or PATH)",
    )
    parser.add_argument(
        "--mineru-timeout",
        type=int,
        default=1800,
        help="MinerU timeout per document in seconds (default: 1800)",
    )
    parser.add_argument(
        "--no-extra-body",
        action="store_true",
        help="Disable sending extra_body in OpenAI SDK requests "
             "(needed for providers that reject it)",
    )
    parser.add_argument(
        "--no-auto-detect",
        action="store_true",
        help="Disable LLM-based auto-detection of main vs. supplementary documents. "
             "When disabled, first PDF is main, rest are SI (positional order).",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Cap number of samples to extract (0 = unlimited)",
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=None,
        help="Max concurrent extraction tasks / batches (default: 8 on the DeepSeek API, 3 otherwise)",
    )
    parser.add_argument(
        "--extraction-mode",
        choices=["agentic", "full_context", "auto"],
        default="agentic",
        help="agentic (default): per-sample tool navigation; full_context: whole paper in "
             "the prompt, batched extraction; auto: full_context when the paper fits "
             "--full-context-budget",
    )
    parser.add_argument(
        "--full-context-budget",
        type=int,
        default=80_000,
        help="Max packed paper size in tokens for full-context extraction in auto mode (default: 80000)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Samples per full-context extraction call (default: 4)",
    )
    parser.add_argument(
        "--max-concurrent-index",
        type=int,
        default=None,
        help="Max concurrent API calls during indexing (default: from config.yaml)",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable DEBUG logging",
    )
    parser.add_argument(
        "--save-log",
        action="store_true",
        help="Write a polyreader log file under the output root",
    )

    args = parser.parse_args()

    # Resolve provider -> base URL and default model
    try:
        provider, args.base_url = resolve_provider(args.provider, args.base_url)
    except ValueError as e:
        parser.error(str(e))
    if args.model is None:
        args.model = DEFAULT_MODELS.get(provider, DEFAULT_MODELS["openrouter"])
    if args.max_concurrent is None:
        # The DeepSeek API allows thousands of concurrent requests per account.
        args.max_concurrent = 8 if provider == DEEPSEEK else 3

    # Default index model to extraction model
    if args.index_model is None:
        args.index_model = args.model

    if args.index_only and args.extract_only:
        parser.error("--index-only and --extract-only are mutually exclusive")

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)

    # Set up file logging in output directory
    if args.save_log:
        base_output_dir = Path(args.output_dir).resolve()
        model_subdir = _model_output_subdir(args.model, args.base_url)
        log_root = base_output_dir / model_subdir if model_subdir else base_output_dir
        log_root.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_root / f"polyreader_{timestamp}.log"
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s"))
        logging.getLogger().addHandler(file_handler)
        logger.info("Log file: %s", log_file)

    asyncio.run(pipeline(args))


if __name__ == "__main__":
    main()

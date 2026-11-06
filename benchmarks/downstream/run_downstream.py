"""Run the unchanged polyreader pipeline on each parser backend's output.

For every (parser configuration, case) it

1. indexes each PDF of the case from the *saved* benchmark parse
   (``benchmarks/parser_bench/saved_parse.py``), so the Markdown is exactly the
   one whose speed/quality was measured, and
2. runs the polyreader CLI with ``--skip-index`` on those indexes — same LLM,
   prompts, schemas, configuration and validation for every backend.

Usage (API key in DEEPSEEK_API_KEY):
    python run_downstream.py --parse-root ~/polyreader/bench/outputs \\
        --papers ~/polyreader/bench/papers --configs marker mineru_standard \\
        -o ~/polyreader/bench/downstream

Then score with evaluate.py --runs <out>/<config>/deepseek.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "parser_bench"))

from saved_parse import use_saved_parse  # noqa: E402


def case_pdfs(gt: dict, papers: Path) -> list[Path]:
    docs = gt["documents"]
    order = ["main"] + [k for k in docs if k != "main"]
    return [papers / gt["case"] / docs[k] for k in order]


def saved_dir(parse_root: Path, config: str, case: str, pdf: Path) -> Path:
    return parse_root / config / case / pdf.stem


async def index_case(pdfs, out_dir: Path, case: str, parse_root: Path, config: str,
                     model: str, base_url: str, index_kwargs: dict) -> list[str]:
    import polyreader
    from polyreader.indexing import save_index, verify_index

    out_dir.mkdir(parents=True, exist_ok=True)
    done = []
    for i, pdf in enumerate(pdfs):
        idx_name = case if i == 0 else f"{case}_doc{i + 1}"
        tree = out_dir / f"{idx_name}_tree.json"
        if tree.exists() and not verify_index(str(tree)):
            done.append(f"{idx_name}: reused")
            continue
        src = saved_dir(parse_root, config, case, pdf)
        if not (src / "page_contents.json").exists():
            raise FileNotFoundError(f"no saved parse for {config}/{case}/{pdf.stem} in {src}")
        with use_saved_parse(src):
            index = await polyreader.process_document(
                str(pdf), base_url=base_url, model=model, **index_kwargs)
            save_index(index, str(out_dir / f"{idx_name}.json"))
        done.append(f"{idx_name}: indexed")
    return done


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parse-root", required=True, type=Path)
    ap.add_argument("--papers", required=True, type=Path)
    ap.add_argument("--configs", nargs="+", required=True)
    ap.add_argument("--cases", nargs="*", default=None)
    ap.add_argument("--gt", type=Path, default=HERE / "ground_truth")
    ap.add_argument("--model", default="deepseek-flash")
    ap.add_argument("--index-model", default=None, help="default: --model")
    ap.add_argument("--base-url", default="https://api.deepseek.com")
    ap.add_argument("--extra-cli", default="", help="extra flags for the polyreader CLI")
    ap.add_argument("-o", "--output", required=True, type=Path)
    args = ap.parse_args()

    parse_root = args.parse_root.expanduser()
    papers = args.papers.expanduser()
    out_root = args.output.expanduser()
    gts = {p.stem: json.loads(p.read_text()) for p in sorted(args.gt.glob("*.json"))}
    cases = args.cases or list(gts)
    log = []
    for config in args.configs:
        backend = "mineru" if config.startswith("mineru") else "marker"
        for case in cases:
            gt = gts[case]
            pdfs = case_pdfs(gt, papers)
            out_dir = out_root / config / "deepseek" / case
            t0 = time.time()
            entry = {"config": config, "case": case}
            try:
                entry["index"] = asyncio.run(index_case(
                    pdfs, out_dir, case, parse_root, config,
                    model=args.index_model or args.model, base_url=args.base_url, index_kwargs={}))
                entry["index_s"] = round(time.time() - t0, 1)
                cmd = ["polyreader", *map(str, pdfs), "--base-url", args.base_url, "--model", args.model,
                       "--index-model", args.index_model or args.model,
                       "-o", str(out_root / config), "--name", case, "--skip-index",
                       "--pdf-parser", backend, "--save-log", *args.extra_cli.split()]
                t1 = time.time()
                proc = subprocess.run(cmd, capture_output=True, text=True)
                (out_dir / "cli_stdout.log").write_text(proc.stdout + "\n--- stderr ---\n" + proc.stderr)
                entry["extract_s"] = round(time.time() - t1, 1)
                entry["exit_code"] = proc.returncode
            except Exception as e:  # keep going with the other cases
                entry["error"] = f"{type(e).__name__}: {e}"
            print(json.dumps(entry), flush=True)
            log.append(entry)
            (out_root / f"run_log_{'_'.join(args.configs)}.json").write_text(json.dumps(log, indent=2))


if __name__ == "__main__":
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    main()

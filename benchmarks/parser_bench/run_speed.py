"""Speed/memory benchmark for ONE parser backend configuration.

Run in a fresh process per configuration so the first document includes model
loading (cold start). For each document it records wall time, pages, s/page,
pages/s, and peak host RSS / VRAM of this process tree, and saves the parse
output under ``<bench_root>/outputs/<config>/<case>/<pdf stem>/``:

    page_contents.json   polyreader index format {"total_pages", "page_contents": [{"page", "text"}]}
    parse_meta.json      parser_info, timings, image list
    images/              extracted images (referenced from the page texts)

Order of work: (1) model init (Marker only, in-process); (2) "cold": the first
document parsed once into a scratch dir — first-call costs (CUDA kernel
loading, caches) land here; (3) the measured pass over all documents, whose
outputs are saved; (4) N documents in parallel ("concurrency"). Scratch outputs
are deleted afterwards. MinerU starts a subprocess per document, so every
MinerU document time includes its model loading.

Examples (on the workstation, `polyextract` env, PYTHONPATH=<repo>):

    python run_speed.py --bench-root ~/polyreader/bench --backend marker \
        --config-name marker --cuda-device 2 --watch-gpus 2

    python run_speed.py --bench-root ~/polyreader/bench --backend mineru \
        --config-name mineru_standard --option tier=standard \
        --option executable=~/miniconda3/envs/mineru/bin/mineru-kit \
        --env MINERU_MODEL_ONNX_DEVICE=gpu --env MINERU_LLAMA_CPP_BACKEND=vulkan \
        --env GGML_VK_VISIBLE_DEVICES=3 --watch-gpus 0,2
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path


def _parse_kv(items: list[str], what: str) -> dict:
    out = {}
    for item in items or []:
        if "=" not in item:
            raise SystemExit(f"--{what} expects KEY=VALUE, got {item!r}")
        k, v = item.split("=", 1)
        try:
            out[k] = json.loads(v)  # numbers / booleans / JSON objects
        except json.JSONDecodeError:
            out[k] = os.path.expanduser(v) if v.startswith("~") else v
    return out


def _worker(args, env_opts: dict) -> None:
    """Concurrency worker: init + parse one PDF, print one JSON line, exit."""
    import asyncio

    options = _parse_kv(args.option, "option")
    if args.backend == "mineru" and env_opts:
        options["extra_env"] = {k: str(v) for k, v in env_opts.items()}
    out: dict = {}
    try:
        from polyreader.parsing.pdf_parser import parse_pdf
        t = time.perf_counter()
        if args.backend == "marker":
            from polyreader.parsing.pdf_parser import _get_artifact_dict
            _get_artifact_dict()
        out["init_s"] = round(time.perf_counter() - t, 3)
        t = time.perf_counter()
        pr = asyncio.run(parse_pdf(str(args.worker_pdf), output_dir=str(args.worker_out),
                                   backend=args.backend, **options))
        out.update({"parse_s": round(time.perf_counter() - t, 3), "pages": len(pr.page_texts)})
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {e}"
    print(json.dumps(out), flush=True)
    os._exit(0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bench-root", required=True, type=Path)
    ap.add_argument("--backend", required=True, choices=["marker", "mineru"])
    ap.add_argument("--config-name", required=True, help="Label for outputs, e.g. mineru_standard")
    ap.add_argument("--cases", nargs="*", help="Subset of case names (default: all)")
    ap.add_argument("--option", action="append", default=[], help="Backend option KEY=VALUE (repeatable)")
    ap.add_argument("--env", action="append", default=[],
                    help="Environment for the backend KEY=VALUE (MinerU: passed as extra_env)")
    ap.add_argument("--cuda-device", default=None, help="Sets CUDA_VISIBLE_DEVICES before importing torch")
    ap.add_argument("--watch-gpus", default="", help="Physical GPU indices for device-level memory delta")
    ap.add_argument("--concurrency", type=int, default=2, help="Docs parsed in parallel in the concurrency run (0 = skip)")
    ap.add_argument("--results", type=Path, default=None, help="Results JSON (default: <bench_root>/results/speed_<config>.json)")
    ap.add_argument("--worker-pdf", type=Path, help=argparse.SUPPRESS)
    ap.add_argument("--worker-out", type=Path, help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args.cuda_device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda_device)
    os.environ.setdefault("TQDM_DISABLE", "1")  # keep logs readable
    env_opts = _parse_kv(args.env, "env")
    if args.backend == "marker":
        # Marker runs in-process: the environment applies to this process.
        os.environ.update({k: str(v) for k, v in env_opts.items()})

    import asyncio
    import shutil

    if args.worker_pdf:
        _worker(args, env_opts)
        return

    import psutil

    from common import (ResourceMonitor, doc_key, environment_info, load_manifest,
                        pdf_page_count, save_parse_output, select_cases, write_json)

    t_process_start = psutil.Process().create_time()
    bench_root = args.bench_root.expanduser().resolve()
    out_root = bench_root / "outputs" / args.config_name
    scratch = bench_root / "scratch" / args.config_name
    results_path = args.results or (bench_root / "results" / f"speed_{args.config_name}.json")
    cases = select_cases(load_manifest(bench_root), args.cases)

    options = _parse_kv(args.option, "option")
    if args.backend == "mineru" and env_opts:
        options["extra_env"] = {k: str(v) for k, v in env_opts.items()}

    watch = [int(x) for x in args.watch_gpus.split(",") if x.strip()]
    mon = ResourceMonitor(watch_gpus=watch).start()

    result: dict = {
        "config": args.config_name,
        "backend": args.backend,
        "options": {k: v for k, v in options.items()},
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "watch_gpus": watch,
        "environment": environment_info(),
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    t_wall_before_import = time.time()
    t0 = time.perf_counter()
    with mon.phase("import"):
        from polyreader.parsing.pdf_parser import parse_pdf
        from polyreader.parsing import ParserError, get_backend
    result["startup"] = {
        "process_start_to_import_s": round(t_wall_before_import - t_process_start, 3),
        "import_s": round(time.perf_counter() - t0, 3),
        **mon.result("import"),
    }

    backend_obj = get_backend(args.backend, **options)
    result["backend_version"] = backend_obj.version()

    # Model initialization (Marker loads its models once per process; MinerU
    # starts a subprocess per document, so its init is inside every doc time).
    init = {"model_init_s": None, "note": ""}
    if args.backend == "marker":
        from polyreader.parsing.pdf_parser import _get_artifact_dict
        t = time.perf_counter()
        with mon.phase("init"):
            _get_artifact_dict()
        init["model_init_s"] = round(time.perf_counter() - t, 3)
        init.update(mon.result("init"))
        try:
            import torch
            init["torch_cuda"] = torch.cuda.is_available()
            init["torch_device"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
            init["torch_version"] = torch.__version__
        except Exception as e:  # pragma: no cover
            init["torch_error"] = str(e)
    else:
        init["note"] = "MinerU runs as a subprocess per document; model loading is included in every document time."
    result["init"] = init

    async def parse_one(pdf: Path, out_dir: Path) -> dict:
        if out_dir.exists():
            shutil.rmtree(out_dir)
        t = time.perf_counter()
        try:
            pr = await parse_pdf(str(pdf), output_dir=str(out_dir), backend=args.backend, **options)
        except ParserError as e:
            return {"error": f"{type(e).__name__}: {e}", "diagnostics": getattr(e, "diagnostics", {}),
                    "seconds": round(time.perf_counter() - t, 3)}
        except Exception as e:
            return {"error": f"{type(e).__name__}: {e}", "seconds": round(time.perf_counter() - t, 3)}
        return {"seconds": round(time.perf_counter() - t, 3), "parse_result": pr}

    async def run_all() -> None:
        # Cold: first document once, right after init (not saved).
        first_case, first_doc = cases[0], cases[0]["docs"][0]
        first_pdf = bench_root / first_doc["pdf"]
        print(f"[{args.config_name}] cold run: {first_case['case']}/{doc_key(first_doc)}", flush=True)
        with mon.phase("cold"):
            r = await parse_one(first_pdf, scratch / "cold")
        first_pages = pdf_page_count(first_pdf)
        result["cold_start"] = {
            "doc": f"{first_case['case']}/{doc_key(first_doc)}", "pdf_pages": first_pages,
            "first_doc_s": r["seconds"], "model_init_s": result["init"]["model_init_s"],
            "import_s": result["startup"]["import_s"],
            "cold_total_s": round(r["seconds"] + (result["init"]["model_init_s"] or 0.0)
                                  + result["startup"]["import_s"], 3),
            "error": r.get("error"), **mon.result("cold"),
        }
        print(f"   cold {r['seconds']:.1f}s", flush=True)

        docs_out = []
        for case in cases:
            for doc in case["docs"]:
                pdf = bench_root / doc["pdf"]
                key = doc_key(doc)
                out_dir = out_root / case["case"] / key
                pdf_pages = pdf_page_count(pdf)
                phase = f"doc:{case['case']}/{key}"
                print(f"[{args.config_name}] {case['case']}/{key} ({pdf_pages} pages) ...", flush=True)
                with mon.phase(phase):
                    r = await parse_one(pdf, out_dir)
                entry = {
                    "case": case["case"], "doc": key, "role": doc["role"], "scanned": doc.get("scanned", False),
                    "pdf_pages": pdf_pages, "seconds": r["seconds"], **mon.result(phase),
                }
                if "error" in r:
                    entry["error"] = r["error"]
                    entry["diagnostics"] = r.get("diagnostics", {})
                    print(f"   ERROR {r['error']}", flush=True)
                else:
                    pr = r["parse_result"]
                    n = len(pr.page_texts)
                    entry.update({
                        "parsed_pages": n,
                        "images": len(pr.images),
                        "s_per_page": round(r["seconds"] / max(pdf_pages, 1), 3),
                        "pages_per_s": round(pdf_pages / r["seconds"], 3) if r["seconds"] else None,
                        "parser_info": pr.parser_info,
                    })
                    save_parse_output(out_dir, pr.page_texts, {
                        "config": args.config_name, "backend": args.backend, "case": case["case"],
                        "doc": key, "role": doc["role"], "pdf": doc["pdf"],
                        "reference_pdf": doc["reference_pdf"], "scanned": doc.get("scanned", False),
                        "pdf_pages": pdf_pages, "seconds": r["seconds"], "parser_info": pr.parser_info,
                        "images": [im.filename for im in pr.images], "mode_used": pr.mode_used,
                    })
                    print(f"   {r['seconds']:.1f}s, {n} pages, {len(pr.images)} images", flush=True)
                docs_out.append(entry)
        result["docs"] = docs_out

        ok = [d for d in docs_out if "error" not in d]
        pages = sum(d["pdf_pages"] for d in ok)
        secs = sum(d["seconds"] for d in ok)
        result["totals"] = {
            "docs_ok": len(ok), "docs_failed": len(docs_out) - len(ok), "pages": pages,
            "seconds": round(secs, 2),
            "s_per_page": round(secs / pages, 3) if pages else None,
            "pages_per_s": round(pages / secs, 3) if secs else None,
        }
        if ok:
            # Warm = the same first document in the measured pass (after the cold run).
            w = next((d for d in ok if f"{d['case']}/{d['doc']}" == result["cold_start"]["doc"]), None)
            if w:
                result["warm"] = {"doc": result["cold_start"]["doc"], "seconds": w["seconds"],
                                  "s_per_page": w["s_per_page"]}

        # Checkpoint before the concurrency run (a crash there must not lose the results).
        write_json(results_path, {**result, "partial": True})

        # Concurrency: N documents parsed at the same time by N worker processes
        # (each loads its own models, like N independent Polyextract runs).
        # In-process threads are not used: Marker aborts ("corrupted
        # double-linked list") when two documents are converted concurrently in
        # one process.
        if args.concurrency and args.concurrency > 1:
            pool = [(c, d) for c in cases for d in c["docs"] if any(o["doc"] == doc_key(d) for o in ok)]
            pool.sort(key=lambda cd: cd[1]["role"] != "main")  # prefer main papers
            picked = pool[: args.concurrency]
            if len(picked) >= 2:
                pages_c = sum(pdf_page_count(bench_root / d["pdf"]) for _, d in picked)
                passthrough = ["--bench-root", str(bench_root), "--backend", args.backend,
                               "--config-name", args.config_name]
                for o in args.option:
                    passthrough += ["--option", o]
                for e in args.env:
                    passthrough += ["--env", e]
                if args.cuda_device is not None:
                    passthrough += ["--cuda-device", str(args.cuda_device)]
                t = time.perf_counter()
                with mon.phase("concurrency"):
                    procs = [await asyncio.create_subprocess_exec(
                        sys.executable, os.path.abspath(__file__), *passthrough,
                        "--worker-pdf", str(bench_root / d["pdf"]), "--worker-out", str(scratch / f"conc_{i}"),
                        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL)
                        for i, (_, d) in enumerate(picked)]
                    outs = await asyncio.gather(*(p.communicate() for p in procs))
                wall = time.perf_counter() - t
                workers = []
                for (c, d), p, (so, _) in zip(picked, procs, outs):
                    line = (so or b"").decode(errors="replace").strip().splitlines()
                    try:
                        w = json.loads(line[-1]) if line else {}
                    except json.JSONDecodeError:
                        w = {"error": f"unparseable worker output: {line[-1][:200] if line else ''}"}
                    workers.append({"doc": f"{c['case']}/{doc_key(d)}", "exit_code": p.returncode, **w})
                seq = sum(next(o["seconds"] for o in ok if o["doc"] == doc_key(d)) for _, d in picked)
                result["concurrency"] = {
                    "n": len(picked), "mode": "worker processes", "docs": [w["doc"] for w in workers],
                    "pages": pages_c, "wall_s": round(wall, 2),
                    "pages_per_s": round(pages_c / wall, 3) if wall else None,
                    "sequential_s": round(seq, 2),
                    "speedup_vs_sequential": round(seq / wall, 2) if wall else None,
                    "workers": workers,
                    "errors": [w.get("error") for w in workers if w.get("error")],
                    **mon.result("concurrency"),
                }
                print(f"[concurrency x{len(picked)}] {wall:.1f}s for {pages_c} pages", flush=True)

    asyncio.run(run_all())
    result["global"] = mon.stop()
    result["vram_per_pid_measured"] = mon.vram_per_pid_seen
    result["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    if scratch.exists():
        shutil.rmtree(scratch, ignore_errors=True)
    write_json(results_path, result)
    print(f"wrote {results_path}")
    # Leave the interpreter without running GPU teardown hooks that can hang.
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()

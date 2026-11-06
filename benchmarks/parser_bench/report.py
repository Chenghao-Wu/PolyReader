"""Merge speed + quality results into ``<bench_root>/reports/parser_benchmark.md``.

    python report.py --bench-root ~/polyreader/bench [--title "..."]
"""

from __future__ import annotations

import argparse
from pathlib import Path

from common import read_json


def _f(x, nd=2, suffix=""):
    if x is None or x == "":
        return "–"
    if isinstance(x, bool):
        return "yes" if x else "no"
    if isinstance(x, float):
        return f"{x:.{nd}f}{suffix}"
    return f"{x}{suffix}"


def _table(header: list[str], rows: list[list]) -> list[str]:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return out + [""]


def _gb(mb):
    return None if mb is None else mb / 1024


def speed_section(speeds: dict[str, dict]) -> list[str]:
    lines = ["## Speed and memory", ""]
    rows = []
    for cfg, s in speeds.items():
        t = s.get("totals", {})
        cs = s.get("cold_start", {})
        w = s.get("warm", {})
        c = s.get("concurrency", {})
        g = s.get("global", {})
        vram = g.get("peak_vram_mb") if s.get("vram_per_pid_measured") else None
        rows.append([
            cfg, s.get("environment", {}).get("hostname", ""),
            _f(t.get("pages")), _f(t.get("seconds"), 1), _f(t.get("s_per_page")), _f(t.get("pages_per_s"), 3),
            _f(s.get("init", {}).get("model_init_s"), 1), _f(cs.get("cold_total_s"), 1),
            _f(w.get("s_per_page")),
            f"{_f(c.get('pages_per_s'), 3)} ({_f(c.get('speedup_vs_sequential'))}×)" if c else "–",
            _f(_gb(g.get("peak_rss_mb")), 1), _f(_gb(vram), 1), _f(_gb(g.get("peak_gpu_dev_delta_mb")), 1),
            f"{t.get('docs_failed', 0)}",
        ])
    lines += _table(
        ["config", "host", "pages", "total s", "s/page", "pages/s", "model init s", "cold start s (first doc)",
         "warm s/page", "concurrent pages/s (speedup)", "peak RSS GB", "peak VRAM GB (per-PID)",
         "GPU mem delta GB", "failed docs"], rows)
    lines += [
        "- *total s / s/page / pages/s*: sequential processing of all documents, excluding model init.",
        "- *cold start*: imports + model init + the first document's first parse (first-call CUDA/kernel costs "
        "land here). MinerU loads its models inside every per-document subprocess, so every MinerU document "
        "time includes model loading.",
        "- *concurrent*: N documents parsed at once by N worker processes (each loads its own models, like N "
        "independent Polyextract runs); wall time includes the workers' model loading. Speedup = sum of the same "
        "documents' sequential times / concurrent wall time.",
        "- *warm*: the same first document in the measured pass (after the cold run); the per-document and "
        "total columns are all measured after the cold run.",
        "- *peak VRAM (per-PID)*: NVML per-process memory of the benchmark process tree; '–' when the driver "
        "reported no per-PID usage (e.g. Vulkan). *GPU mem delta*: device-level used-memory rise on the watched "
        "GPUs (shared machine: may include other users' jobs).",
        "",
        "### Per document", "",
    ]
    rows = []
    for cfg, s in speeds.items():
        for d in s.get("docs", []):
            rows.append([cfg, f"{d['case']}/{d['doc']}", d.get("role", ""), _f(d.get("scanned")),
                         _f(d.get("pdf_pages")), _f(d.get("parsed_pages")), _f(d.get("seconds"), 1),
                         _f(d.get("s_per_page")), _f(_gb(d.get("peak_rss_mb")), 1),
                         _f(_gb(d.get("peak_vram_mb")) if s.get("vram_per_pid_measured") else None, 1),
                         _f(d.get("mean_gpu_util_pct"), 0), _f(d.get("images")), (d.get("error") or "")[:80]])
    lines += _table(["config", "document", "role", "scanned", "PDF pages", "parsed pages", "s", "s/page",
                     "peak RSS GB", "peak VRAM GB", "watched-GPU util %", "images", "error"], rows)
    lines += ["*watched-GPU util %*: mean utilisation of the GPUs used, including other users' jobs "
              "(shared machine) — a contention indicator, not this backend's own load.", ""]
    return lines


def quality_section(quality: dict) -> list[str]:
    lines = ["## Document reconstruction quality (automatic, vs. PDF text layer)", ""]
    rows = []
    for cfg, q in quality.get("configs", {}).items():
        docs = [d for d in q["docs"] if not d.get("missing")]
        if not docs:
            continue
        ref_tok = sum(d["ref_tokens"] for d in docs)
        par_tok = sum(d["parsed_tokens"] for d in docs)
        rec = sum((d["token_recall"] or 0) * d["ref_tokens"] for d in docs) / ref_tok if ref_tok else None
        prec = sum((d["token_precision"] or 0) * d["parsed_tokens"] for d in docs) / par_tok if par_tok else None
        maps = [d["page_mapping_accuracy"] for d in docs if d["page_mapping_accuracy"] is not None]
        orders = [d["order_score"] for d in docs if d["order_score"] is not None]
        sym_ref = sum(sum(d["symbols_ref"].values()) for d in docs)
        sym_kept = sum(sum(min(d["symbols_parsed"].get(k, 0), v) for k, v in d["symbols_ref"].items()) for d in docs)
        rows.append([
            cfg, f"{sum(d['page_count_match'] for d in docs)}/{len(docs)}",
            _f(rec, 3), _f(prec, 3), _f(sum(maps) / len(maps) if maps else None, 3),
            _f(sum(orders) / len(orders) if orders else None, 3),
            sum(d["headings"] for d in docs), sum(d["tables"] for d in docs), sum(d["table_cells"] for d in docs),
            f"{sum(d['equations_display'] for d in docs)}/{sum(d['equations_inline'] for d in docs)}",
            sum(d["image_refs"] for d in docs), sum(d["broken_image_refs"] for d in docs),
            _f(sym_kept / sym_ref if sym_ref else None, 3),
            sum(1 for d in q["docs"] if d.get("missing")),
        ])
    lines += _table(["config", "page count match", "token recall", "token precision", "page mapping",
                     "reading order", "headings", "tables", "table cells", "equations display/inline",
                     "image refs", "broken refs", "symbol retention", "missing docs"], rows)
    lines += ["### Per document", ""]
    rows = []
    for cfg, q in quality.get("configs", {}).items():
        for d in q["docs"]:
            if d.get("missing"):
                rows.append([cfg, f"{d['case']}/{d['doc']}"] + ["missing"] + [""] * 9)
                continue
            rows.append([cfg, f"{d['case']}/{d['doc']}", f"{d['parsed_pages']}/{d['pdf_pages']}",
                         _f(d["token_recall"], 3), _f(d["token_precision"], 3),
                         _f(d["page_mapping_accuracy"], 3), _f(d["order_score"], 3), d["tables"],
                         d["table_cells"], f"{d['equations_display']}/{d['equations_inline']}",
                         f"{d['image_refs']} ({d['broken_image_refs']} broken)", _f(d["symbol_retention"], 3)])
    lines += _table(["config", "document", "pages", "recall", "precision", "page map", "order", "tables",
                     "cells", "eq d/i", "image refs", "symbols"], rows)
    lines += [
        "Notes: the reference is the PDF text layer (PyMuPDF), which keeps running headers/footers and "
        "flattens tables, so recall < 1 and precision < 1 are expected for every backend. For the scanned case "
        "the reference is the original (born-digital) PDF's text layer. *Symbol retention* pools polymer notation "
        "(Đ, Mn, Mw, Tg, Tm, °C, g/mol, mol%, -co-, -b-, ×10, ±, Sn(Oct)2, Greek letters) after flattening LaTeX "
        "(e.g. `M_{n}` → Mn) and HTML sub/superscripts. Table cell accuracy is not scored automatically — see "
        "`reports/tables/<case>/<doc>.md` for side-by-side table dumps.",
        "",
    ]
    return lines


def setup_section(speeds: dict[str, dict]) -> list[str]:
    lines = ["## Setup", ""]
    rows = []
    for cfg, s in speeds.items():
        env = s.get("environment", {})
        pk = env.get("packages", {})
        info = next((d.get("parser_info") for d in s.get("docs", []) if d.get("parser_info")), {}) or {}
        opts = {k: v for k, v in s.get("options", {}).items() if k != "extra_env"}
        watch = [str(g) for g in s.get("watch_gpus", [])]
        # Only the GPUs this configuration used (nvidia-smi rows start with the index).
        gpus = "; ".join(g for g in env.get("gpus", []) if g.split(",")[0].strip() in watch) or \
            "; ".join(env.get("gpus", []))
        extra_env = (s.get("options", {}).get("extra_env") or {})
        if s.get("backend") == "mineru":
            device = (f"llama.cpp Vulkan device {extra_env.get('GGML_VK_VISIBLE_DEVICES', '?')}, "
                      f"ONNX {extra_env.get('MINERU_MODEL_ONNX_DEVICE', 'auto')}; watched GPUs {','.join(watch)}")
        else:
            device = f"CUDA_VISIBLE_DEVICES={s.get('cuda_visible_devices', '')}"
        rows.append([
            cfg, s.get("backend"), s.get("backend_version") or info.get("version", ""),
            info.get("tier", opts.get("tier", "")), env.get("hostname", ""), gpus,
            pk.get("torch", ""), pk.get("marker-pdf", ""), pk.get("surya-ocr", ""), device,
            s.get("started_at", ""),
        ])
    lines += _table(["config", "backend", "version", "tier", "host", "GPU(s) used (index, name, driver, mem total, used + util at start incl. other jobs)",
                     "torch", "marker-pdf", "surya-ocr", "device", "run start"], rows)
    return lines


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bench-root", required=True, type=Path)
    ap.add_argument("--title", default="Marker vs MinerU — parser benchmark (5 polymer papers)")
    args = ap.parse_args()
    root = args.bench_root.expanduser().resolve()

    speeds = {}
    for p in sorted((root / "results").glob("speed_*.json")):
        s = read_json(p)
        speeds[s["config"]] = s
    qpath = root / "results" / "quality.json"
    quality = read_json(qpath) if qpath.exists() else {"configs": {}}

    cases = read_json(root / "cases.json")["cases"]
    lines = [f"# {args.title}", "", "## Corpus", ""]
    lines += _table(["case", "documents"], [[c["case"], ", ".join(
        f"{Path(d['pdf']).name} ({d['role']}{', scanned' if d.get('scanned') else ''})" for d in c["docs"])]
        for c in cases])
    lines += setup_section(speeds) + speed_section(speeds) + quality_section(quality)

    out = root / "reports" / "parser_benchmark.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

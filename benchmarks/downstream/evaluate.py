"""Score polyreader sample JSONs against the manual ground truth.

Usage:
    python evaluate.py --gt ground_truth/ --runs RUN_ROOT/<config> [--runs ...] -o scores.json

``RUN_ROOT/<config>/<case>/`` must contain the ``samples/*.json`` written by
the polyreader CLI (one file per emitted sample) and optionally the
``*_pipeline_report.json``.

Metrics per case and configuration:
- sample identification: precision / recall / F1 of extracted sample IDs
  matched to ground-truth samples (exact or alias match after normalisation);
  extracted samples matching an *optional* ground-truth sample are neither
  rewarded nor penalised;
- field accuracy for Mn, Mw, PDI, Tg, Tm on matched samples: a value counts as
  correct when within tolerance of an accepted ground-truth value (Mn/Mw: 3 %,
  PDI: 0.015, temperatures: 0.6 °C); precision = correct / extracted values,
  recall = correct / ground-truth values (unmatched GT samples count as misses);
- provenance: share of correct values whose provenance pages include the
  ground-truth page.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

FIELDS = ("mn", "mw", "pdi", "tg", "tm")


def norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(text).lower())


def _tokens(text: str) -> set[str]:
    return {t for t in re.split(r"[^a-z0-9]+", str(text).lower()) if t}


def load_extracted(case_dir: Path) -> list[dict]:
    samples = []
    for f in sorted((case_dir / "samples").glob("*.json")):
        try:
            samples.append(json.loads(f.read_text(encoding="utf-8")))
        except (OSError, ValueError):
            continue
    return samples


def _values(sample: dict) -> dict[str, tuple[float, list[int]]]:
    """Extracted numeric values with their provenance pages."""
    out: dict[str, tuple[float, list[int]]] = {}
    mw = sample.get("molecular_weight") or {}
    pages = (mw.get("provenance") or {}).get("page_numbers") or []
    for key in ("mn", "mw", "pdi"):
        if isinstance(mw.get(key), (int, float)):
            out[key] = (float(mw[key]), pages)
    for prop in sample.get("properties") or []:
        name = str(prop.get("property_name", "")).lower().strip()
        value = prop.get("value")
        if not isinstance(value, (int, float)):
            continue
        unit = str(prop.get("unit", "")).strip()
        if unit in ("K", "kelvin"):
            value = value - 273.15
        for sym in ("tg", "tm"):
            if re.match(rf"{sym}(?![a-z])", name) and sym not in out:
                out[sym] = (float(value), (prop.get("provenance") or {}).get("page_numbers") or [])
    return out


def _close(field: str, got: float, accepted: list[float]) -> bool:
    for ref in accepted:
        if field in ("mn", "mw"):
            if abs(got - ref) <= 0.03 * abs(ref):
                return True
        elif field == "pdi":
            if abs(got - ref) <= 0.015:
                return True
        elif abs(got - ref) <= 0.6:
            return True
    return False


def _derivable(field: str, value: float, gt: dict) -> bool:
    """True if ``value`` follows from reported data (Mw = Mn*PDI, PDI = Mw/Mn)."""
    mns, mws, pdi = gt.get("mn") or [], gt.get("mw") or [], gt.get("pdi")
    if field == "mw" and pdi:
        return any(abs(value - mn * pdi) <= 0.03 * mn * pdi for mn in mns)
    if field == "pdi" and mns and mws:
        return any(abs(value - mw / mn) <= 0.02 for mn in mns for mw in mws)
    return False


def _token_list(text: str) -> list[str]:
    return [t for t in re.split(r"[^a-z0-9]+", str(text).lower()) if t]


def _contains_seq(tokens: list[str], seq: list[str]) -> bool:
    n = len(seq)
    return n > 0 and any(tokens[i:i + n] == seq for i in range(len(tokens) - n + 1))


def _match(extracted: list[dict], gt_samples: list[dict]) -> dict[int, int]:
    """Map extracted index -> GT index by sample-ID / alias matching.

    1. exact match of the normalised sample_id with an id/alias;
    2. otherwise the GT id/alias whose tokens occur contiguously in the
       sample_id ("PTMC-b-PCL (Table 2, Run 4)" contains "Run 4"), preferring
       the longest alias. Pure-number aliases only match exactly.
    Required GT samples are matched at most once; optional ones may absorb
    several extracted samples (not scored).
    """
    names = [[g["id"], *g.get("aliases", [])] for g in gt_samples]
    mapping: dict[int, int] = {}
    used: set[int] = set()
    for ei, e in enumerate(extracted):
        sid = norm(e.get("sample_id", ""))
        for gi, ns in enumerate(names):
            if (gi not in used or gt_samples[gi].get("optional")) and sid and sid in {norm(n) for n in ns}:
                mapping[ei] = gi
                if not gt_samples[gi].get("optional"):
                    used.add(gi)
                break
    for ei, e in enumerate(extracted):
        if ei in mapping:
            continue
        tokens = _token_list(e.get("sample_id", ""))
        from_si = "supplementary" in tokens or "si" in tokens
        best, best_len = None, 0
        for gi, ns in enumerate(names):
            if gi in used and not gt_samples[gi].get("optional"):
                continue
            if from_si and gt_samples[gi].get("doc", "main") == "main":
                continue  # "Supplementary Table 1 run 9" is not main-text "Table 1 run 9"
            for n in ns:
                seq = _token_list(n)
                if all(t.isdigit() for t in seq):
                    continue
                if _contains_seq(tokens, seq) and len(seq) > best_len:
                    best, best_len = gi, len(seq)
        if best is not None:
            mapping[ei] = best
            if not gt_samples[best].get("optional"):
                used.add(best)
    return mapping


def score_case(gt: dict, extracted: list[dict]) -> dict:
    gt_samples = gt["samples"]
    mapping = _match(extracted, gt_samples)
    patterns_pre = [re.compile(p, re.IGNORECASE) for p in gt.get("unscored_table_patterns", [])]
    for ei, e in enumerate(extracted):
        sid = str(e.get("sample_id", "")).strip()
        gi = mapping.get(ei)
        if gi is not None and gt_samples[gi].get("doc", "main") == "main" and any(p.fullmatch(sid) for p in patterns_pre) \
                and norm(sid) not in {norm(n) for n in [gt_samples[gi]["id"], *gt_samples[gi].get("aliases", [])]}:
            del mapping[ei]  # matched only loosely and belongs to an unscored table
    required = {i for i, g in enumerate(gt_samples) if not g.get("optional")}
    matched_required = {gi for gi in mapping.values() if gi in required}
    matched_optional = {ei for ei, gi in mapping.items() if gi not in required}
    # Unscored intermediates (e.g. macro-RAFT agents) described by patterns.
    patterns = [re.compile(p, re.IGNORECASE) for p in gt.get("optional_patterns", [])]
    for ei, e in enumerate(extracted):
        if ei not in mapping and any(p.fullmatch(str(e.get("sample_id", "")).strip()) for p in patterns):
            matched_optional.add(ei)
    n_extracted_scored = len(extracted) - len(matched_optional)

    tp = len(matched_required)

    field_stats = {f: {"correct": 0, "extracted": 0, "gt": 0, "page_ok": 0} for f in FIELDS}
    errors = []
    for gi in required:
        for f in FIELDS:
            if f in gt_samples[gi]:
                field_stats[f]["gt"] += 1
    for ei, e in enumerate(extracted):
        gi = mapping.get(ei)
        if ei in matched_optional:
            continue  # optional GT sample / intermediate: not scored
        vals = _values(e)
        for f, (v, pages) in vals.items():
            field_stats[f]["extracted"] += 1
            optional_ref = (gt_samples[gi].get("optional_fields") or {}).get(f) if gi is not None else None
            if gi is not None and f not in gt_samples[gi] and optional_ref is not None:
                tol = (gt_samples[gi].get("optional_tolerance") or {}).get(f)
                if (abs(v - optional_ref) <= tol) if tol is not None else _close(f, v, [optional_ref]):
                    field_stats[f]["extracted"] -= 1  # acceptable, but not scored
                else:
                    errors.append(f"{e.get('sample_id')}: {f}={v} expected {optional_ref} (optional)")
                continue
            if gi is not None and f not in gt_samples[gi] and _derivable(f, v, gt_samples[gi]):
                field_stats[f]["extracted"] -= 1  # e.g. Mw computed as Mn x PDI: consistent, not scored
                continue
            if gi is None or f not in gt_samples[gi]:
                if gi is not None:
                    errors.append(f"{e.get('sample_id')}: unexpected {f}={v}")
                continue
            accepted = gt_samples[gi][f]
            accepted = accepted if isinstance(accepted, list) else [accepted]
            if _close(f, v, accepted):
                field_stats[f]["correct"] += 1
                if set(pages) & set(gt_samples[gi].get("pages", [])):
                    field_stats[f]["page_ok"] += 1
            else:
                errors.append(f"{e.get('sample_id')}: {f}={v} expected {accepted}")

    def prf(c, e, g):
        p = c / e if e else (1.0 if not g else 0.0)
        r = c / g if g else 1.0
        f1 = 2 * p * r / (p + r) if p + r else 0.0
        return round(p, 3), round(r, 3), round(f1, 3)

    fields = {}
    tot = {"correct": 0, "extracted": 0, "gt": 0, "page_ok": 0}
    for f, st in field_stats.items():
        if st["gt"] or st["extracted"]:
            fields[f] = dict(st, **dict(zip(("precision", "recall", "f1"), prf(st["correct"], st["extracted"], st["gt"]))))
        for k in tot:
            tot[k] += st[k]
    overall = dict(zip(("precision", "recall", "f1"), prf(tot["correct"], tot["extracted"], tot["gt"])))
    unmatched = [e.get("sample_id") for ei, e in enumerate(extracted)
                 if ei not in mapping and ei not in matched_optional]
    # Completeness beyond the scored fields: how much each record carries.
    tiers: dict[str, int] = {}
    for e in extracted:
        tiers[e.get("quality_tier", "")] = tiers.get(e.get("quality_tier", ""), 0) + 1
    completeness = {
        "properties_per_sample": round(sum(len(e.get("properties") or []) for e in extracted)
                                       / len(extracted), 2) if extracted else 0.0,
        "processing_steps_per_sample": round(sum(len(e.get("processing_history") or []) for e in extracted)
                                             / len(extracted), 2) if extracted else 0.0,
        "with_synthesis_method": sum(1 for e in extracted if (e.get("synthesis") or {}).get("method")),
        "with_smiles": sum(1 for e in extracted if any(c.get("smiles") for c in e.get("constituents") or [])),
        "tiers": tiers,
    }
    missed = [gt_samples[gi]["id"] for gi in sorted(required - matched_required)]
    return {
        "samples_extracted": len(extracted),
        "samples_scored": n_extracted_scored,
        "samples_gt": len(required),
        "sample_id": dict(zip(("precision", "recall", "f1"),
                              prf(tp, n_extracted_scored, len(required)))) | {"tp": tp},
        "fields": fields,
        "values_overall": overall | tot,
        "provenance_page_ok": round(tot["page_ok"] / tot["correct"], 3) if tot["correct"] else None,
        "completeness": completeness,
        "unmatched_extracted": unmatched,
        "missed_gt": missed,
        "value_errors": errors[:40],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--gt", default=str(Path(__file__).parent / "ground_truth"))
    ap.add_argument("--runs", nargs="+", required=True,
                    help="one directory per configuration; NAME=PATH to name it explicitly")
    ap.add_argument("-o", "--output", default="scores.json")
    args = ap.parse_args()

    gts = {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in sorted(Path(args.gt).glob("*.json"))}
    results: dict[str, dict] = {}
    for run in args.runs:
        name, sep, path = run.partition("=")
        run_dir = Path(path if sep else run)
        # <out>/<config>/<provider> -> name the configuration, not the provider
        config = name if sep else (run_dir.parent.name if run_dir.name in ("deepseek", "openrouter")
                                   else run_dir.name)
        results[config] = {}
        for case, gt in gts.items():
            case_dir = run_dir / case
            if not case_dir.is_dir():
                continue
            entry = score_case(gt, load_extracted(case_dir))
            reports = list(case_dir.glob("*_pipeline_report.json"))
            if reports:
                rep = json.loads(reports[0].read_text(encoding="utf-8"))
                entry["pipeline_error"] = rep.get("error")
                entry["llm_usage"] = rep.get("llm_usage")
            results[config][case] = entry
        # micro-averaged totals across cases
        tot = {"correct": 0, "extracted": 0, "gt": 0}
        id_tot = {"tp": 0, "extracted": 0, "gt": 0}
        for case, e in results[config].items():
            if case.startswith("_"):
                continue
            for k in tot:
                tot[k] += e["values_overall"][k]
            id_tot["tp"] += e["sample_id"]["tp"]
            id_tot["extracted"] += e["samples_scored"]
            id_tot["gt"] += e["samples_gt"]
        p = tot["correct"] / tot["extracted"] if tot["extracted"] else 0.0
        r = tot["correct"] / tot["gt"] if tot["gt"] else 0.0
        ip = id_tot["tp"] / id_tot["extracted"] if id_tot["extracted"] else 0.0
        ir = id_tot["tp"] / id_tot["gt"] if id_tot["gt"] else 0.0
        results[config]["_total"] = {
            "values": {"precision": round(p, 3), "recall": round(r, 3),
                       "f1": round(2 * p * r / (p + r), 3) if p + r else 0.0, **tot},
            "sample_id": {"precision": round(ip, 3), "recall": round(ir, 3),
                          "f1": round(2 * ip * ir / (ip + ir), 3) if ip + ir else 0.0, **id_tot},
        }
    Path(args.output).write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    for config, cases in results.items():
        t = cases["_total"]
        print(f"{config}: sample-ID P/R/F1 = {t['sample_id']['precision']}/{t['sample_id']['recall']}/"
              f"{t['sample_id']['f1']}  values P/R/F1 = {t['values']['precision']}/{t['values']['recall']}/"
              f"{t['values']['f1']}")


if __name__ == "__main__":
    main()

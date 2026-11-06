"""Run extraction-only benchmark arms on existing indexes.

Copies the saved indexes of each case (``<index-root>/<case>/*_tree.json``,
``*_page_contents.json``, ``images_*``) into ``<out>/<arm>/deepseek/<case>/``
and runs the polyreader CLI with ``--skip-index`` plus the arm's flags, so
arms differ only in how extraction runs. Writes ``<out>/<arm>/arm_log.json``
with wall time per case.

    python run_extraction_arms.py --index-root ~/polyreader/bench/downstream/marker/deepseek \\
        --papers ~/polyreader/bench/papers --out ~/polyreader/bench/accel \\
        --arm full_context_r1 -- --extraction-mode full_context
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index-root", required=True, type=Path)
    ap.add_argument("--papers", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--cases", nargs="*", default=None)
    ap.add_argument("--gt", type=Path, default=HERE / "ground_truth")
    ap.add_argument("cli_args", nargs=argparse.REMAINDER, help="extra CLI flags after --")
    args = ap.parse_args()
    extra = [a for a in args.cli_args if a != "--"]

    gts = {p.stem: json.loads(p.read_text()) for p in sorted(args.gt.glob("*.json"))}
    root = args.out.expanduser() / args.arm
    log = []
    for case in args.cases or list(gts):
        docs = gts[case]["documents"]
        pdfs = [args.papers.expanduser() / case / docs[k] for k in ["main"] + [k for k in docs if k != "main"]]
        dst = root / "deepseek" / case
        dst.mkdir(parents=True, exist_ok=True)
        src = args.index_root.expanduser() / case
        for f in list(src.glob("*_tree.json")) + list(src.glob("*_page_contents.json")):
            shutil.copy2(f, dst / f.name)
        for d in src.glob("images_*"):
            if d.is_dir() and not (dst / d.name).exists():
                shutil.copytree(d, dst / d.name)
        cmd = ["polyreader", *map(str, pdfs), "--provider", "deepseek", "-o", str(root),
               "--name", case, "--skip-index", "--pdf-parser", "marker", "--save-log", *extra]
        t0 = time.time()
        proc = subprocess.run(cmd, capture_output=True, text=True)
        (dst / "cli_stdout.log").write_text(proc.stdout + "\n--- stderr ---\n" + proc.stderr)
        entry = {"arm": args.arm, "case": case, "wall_s": round(time.time() - t0, 1),
                 "exit_code": proc.returncode, "flags": extra}
        print(json.dumps(entry), flush=True)
        log.append(entry)
        (root / "arm_log.json").write_text(json.dumps(log, indent=2))


if __name__ == "__main__":
    main()

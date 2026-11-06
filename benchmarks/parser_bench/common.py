"""Shared helpers for the parser benchmark: case manifest, resource monitor, I/O."""

from __future__ import annotations

import json
import os
import platform
import socket
import subprocess
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Case manifest
# ---------------------------------------------------------------------------
#
# bench_root/cases.json:
# {"cases": [{"case": "P1_x", "docs": [{"pdf": "papers/P1_x/a.pdf",
#                                        "reference_pdf": "papers/P1_x/a.pdf",
#                                        "role": "main"}]}]}
# Paths are relative to bench_root.


def load_manifest(bench_root: Path) -> list[dict]:
    with open(bench_root / "cases.json", encoding="utf-8") as f:
        return json.load(f)["cases"]


def select_cases(cases: list[dict], names: list[str] | None) -> list[dict]:
    if not names:
        return cases
    wanted = set(names)
    picked = [c for c in cases if c["case"] in wanted]
    missing = wanted - {c["case"] for c in picked}
    if missing:
        raise SystemExit(f"Unknown cases: {sorted(missing)}")
    return picked


def doc_key(doc: dict) -> str:
    """Stable output-folder name for one document (its PDF stem)."""
    return Path(doc["pdf"]).stem


def pdf_page_count(path: Path) -> int:
    import pymupdf as fitz
    with fitz.open(str(path)) as d:
        return d.page_count


# ---------------------------------------------------------------------------
# JSON I/O
# ---------------------------------------------------------------------------

def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def read_json(path: Path) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_parse_output(doc_dir: Path, page_texts: list[str], meta: dict) -> None:
    """Save page texts in the polyreader page_contents.json format + metadata.

    Images are already in ``doc_dir/images`` (parse_pdf output_dir=doc_dir).
    """
    write_json(doc_dir / "page_contents.json", {
        "total_pages": len(page_texts),
        "page_contents": [{"page": i + 1, "text": t} for i, t in enumerate(page_texts)],
    })
    write_json(doc_dir / "parse_meta.json", meta)


def load_page_texts(doc_dir: Path) -> list[str]:
    data = read_json(doc_dir / "page_contents.json")
    return [p["text"] for p in data["page_contents"]]


# ---------------------------------------------------------------------------
# Environment / versions
# ---------------------------------------------------------------------------

def _pkg_version(name: str) -> str:
    try:
        from importlib.metadata import version
        return version(name)
    except Exception:
        return ""


def environment_info() -> dict:
    info: dict[str, Any] = {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu_count": os.cpu_count(),
        "packages": {p: _pkg_version(p) for p in
                     ("marker-pdf", "surya-ocr", "torch", "pymupdf", "psutil", "nvidia-ml-py")},
        "env": {k: os.environ.get(k, "") for k in
                ("CUDA_VISIBLE_DEVICES", "MINERU_MODEL_ONNX_DEVICE", "MINERU_LLAMA_CPP_BACKEND",
                 "GGML_VK_VISIBLE_DEVICES")},
    }
    try:
        import psutil
        info["ram_total_gb"] = round(psutil.virtual_memory().total / 2**30, 1)
    except Exception:
        pass
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,driver_version,memory.total,memory.used,utilization.gpu",
             "--format=csv,noheader"], capture_output=True, text=True, timeout=30).stdout
        info["gpus"] = [line.strip() for line in out.strip().splitlines()]
    except Exception:
        info["gpus"] = []
    return info


# ---------------------------------------------------------------------------
# Resource monitor: peak RSS + VRAM of this process tree, per phase
# ---------------------------------------------------------------------------

class ResourceMonitor:
    """Polls host RSS and NVIDIA memory of this process and all children.

    Peaks are tracked per named phase (``with mon.phase("x"): ...``) and
    globally. VRAM is the sum of NVML per-process ``usedGpuMemory`` over
    compute and graphics processes belonging to the tree (Vulkan apps show
    up as graphics/compute processes too, when the driver reports them).
    Also records the device-level used-memory delta of ``watch_gpus`` vs. the
    baseline at start, as a fallback when per-PID numbers are unavailable.
    """

    def __init__(self, interval: float = 0.2, watch_gpus: list[int] | None = None):
        import psutil

        self._psutil = psutil
        self.interval = interval
        self.proc = psutil.Process()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._phase: str | None = None
        self.peaks: dict[str, dict[str, float]] = {}
        self._nvml = None
        self._handles: list = []
        self.watch_gpus = watch_gpus or []
        self._baseline_dev: dict[int, float] = {}
        self.vram_per_pid_seen = False
        # Mean utilisation of the watched GPUs per phase (documents contention
        # from other users' jobs on a shared machine): phase -> [sum, n].
        self._util: dict[str, list[float]] = {}
        try:
            import pynvml
            pynvml.nvmlInit()
            self._nvml = pynvml
            n = pynvml.nvmlDeviceGetCount()
            self._handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(n)]
            for g in self.watch_gpus:
                if g < n:
                    self._baseline_dev[g] = pynvml.nvmlDeviceGetMemoryInfo(self._handles[g]).used / 2**20
        except Exception:
            self._nvml = None
        self._thread = threading.Thread(target=self._run, daemon=True)

    # -- sampling -----------------------------------------------------------

    def _tree_pids(self) -> set[int]:
        pids = {self.proc.pid}
        try:
            for c in self.proc.children(recursive=True):
                pids.add(c.pid)
        except self._psutil.Error:
            pass
        return pids

    def _sample(self) -> dict[str, float]:
        rss = 0
        pids = self._tree_pids()
        for pid in pids:
            try:
                rss += self._psutil.Process(pid).memory_info().rss
            except self._psutil.Error:
                pass
        sample = {"rss_mb": rss / 2**20, "vram_mb": 0.0, "gpu_dev_delta_mb": 0.0}
        if self._nvml is not None:
            nv = self._nvml
            vram = 0.0
            for h in self._handles:
                seen: set[int] = set()
                for getter in ("nvmlDeviceGetComputeRunningProcesses", "nvmlDeviceGetGraphicsRunningProcesses"):
                    try:
                        procs = getattr(nv, getter)(h)
                    except Exception:
                        continue
                    for p in procs:
                        if p.pid in pids and p.pid not in seen and p.usedGpuMemory:
                            seen.add(p.pid)
                            vram += p.usedGpuMemory / 2**20
            if vram > 0:
                self.vram_per_pid_seen = True
            sample["vram_mb"] = vram
            delta = 0.0
            for g, base in self._baseline_dev.items():
                try:
                    used = nv.nvmlDeviceGetMemoryInfo(self._handles[g]).used / 2**20
                    delta += max(0.0, used - base)
                except Exception:
                    pass
            sample["gpu_dev_delta_mb"] = delta
            utils = []
            for g in self._baseline_dev:
                try:
                    utils.append(nv.nvmlDeviceGetUtilizationRates(self._handles[g]).gpu)
                except Exception:
                    pass
            if utils:
                sample["_util"] = sum(utils) / len(utils)
        return sample

    def _update(self, key: str, sample: dict[str, float]) -> None:
        cur = self.peaks.setdefault(key, {})
        if "_util" in sample:
            acc = self._util.setdefault(key, [0.0, 0])
            acc[0] += sample["_util"]
            acc[1] += 1
        for k, v in sample.items():
            if k == "_util":
                continue
            if v > cur.get(k, 0.0):
                cur[k] = v

    def _run(self) -> None:
        while not self._stop.is_set():
            s = self._sample()
            with self._lock:
                self._update("_global", s)
                if self._phase:
                    self._update(self._phase, s)
            self._stop.wait(self.interval)

    # -- API ------------------------------------------------------------------

    def start(self) -> "ResourceMonitor":
        self._thread.start()
        return self

    def stop(self) -> dict:
        self._stop.set()
        self._thread.join(timeout=5)
        return self.result("_global")

    def result(self, phase: str) -> dict:
        with self._lock:
            peaks = dict(self.peaks.get(phase, {}))
            util = self._util.get(phase)
        return {
            "mean_gpu_util_pct": round(util[0] / util[1], 1) if util and util[1] else None,
            "peak_rss_mb": round(peaks.get("rss_mb", 0.0), 1),
            "peak_vram_mb": round(peaks.get("vram_mb", 0.0), 1) if self._nvml else None,
            "peak_gpu_dev_delta_mb": round(peaks.get("gpu_dev_delta_mb", 0.0), 1) if self._nvml else None,
        }

    @contextmanager
    def phase(self, name: str):
        with self._lock:
            prev = self._phase
            self._phase = name
            self._update(name, self._sample())
        try:
            yield
        finally:
            with self._lock:
                self._update(name, self._sample())
                self._phase = prev


def now() -> float:
    return time.perf_counter()

# Parser benchmark: Marker vs MinerU

Measures speed/memory and document-reconstruction quality of the PDF parser
backends (`polyreader.parsing`) on a small set of polymer papers, and
saves every parse so the downstream Polyextract comparison can reuse it.

Scripts (run from this directory, with the repository root on `PYTHONPATH`):

| script | what it does |
|---|---|
| `prepare_cases.py` | builds `<bench_root>/cases.json` from `<bench_root>/papers/<case>/*.pdf`; rasterizes `*_source.pdf` into an image-only `*_scanned.pdf` |
| `run_speed.py` | runs ONE backend configuration over all cases in a fresh process; writes `results/speed_<config>.json` and `outputs/<config>/<case>/<pdf stem>/` |
| `quality.py` | scores every saved output against the PDF text layer; writes `results/quality.json` and table dumps in `reports/tables/` |
| `report.py` | merges everything into `reports/parser_benchmark.md` |
| `saved_parse.py` | reuse a saved parse in `process_document()` for the downstream extraction comparison |

## Corpus layout

```
<bench_root>/papers/<case>/<paper>.pdf          # main paper
<bench_root>/papers/<case>/<paper>_suppl.pdf    # SI (filename contains suppl/supporting/_si)
<bench_root>/papers/<case>/<paper>_source.pdf   # optional: becomes <paper>_scanned.pdf (no text layer)
```

## Outputs (per configuration and document)

```
outputs/<config>/<case>/<pdf stem>/page_contents.json   # polyreader index format: {"total_pages", "page_contents": [{"page", "text"}]}
outputs/<config>/<case>/<pdf stem>/parse_meta.json      # parser_info (backend, version, tier), timing, image list
outputs/<config>/<case>/<pdf stem>/images/              # images referenced as ![](<file>) in page texts
```

`page_contents.json` + `images/` are exactly what `PaperIndex` needs, so the
downstream step can build trees from these parses without re-parsing:
`saved_parse.use_saved_parse(doc_dir)` patches `polyreader.indexing.paper_index.parse_pdf` so
`process_document()` reuses the saved parse (call `save_index` inside the
`with` block).

## Running on the workstations

Environment: conda env `polyextract` (marker-pdf 1.10.2, torch, psutil,
nvidia-ml-py, pymupdf). MinerU 4.x lives in its own env (`mineru`) on
XJTLU_SA264_DYVLINGE and is invoked as a subprocess through its executable path.

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate polyextract
export PYTHONPATH=~/polyreader/polyextract   # repository root
cd ~/polyreader/polyextract/benchmarks/parser_bench
B=~/polyreader/bench

python prepare_cases.py --bench-root $B

# Marker (one GPU; CUDA index = nvidia-smi index)
python run_speed.py --bench-root $B --backend marker --config-name marker \
    --cuda-device 1 --watch-gpus 1

# MinerU (Vulkan path on driver 535; raw Vulkan index != nvidia-smi index:
# on DYVLINGE GPUs 0,1,2,3 are Vulkan devices 0,2,3,4; ONNX always uses GPU 0)
python run_speed.py --bench-root $B --backend mineru --config-name mineru_standard \
    --option tier=standard --option executable=$HOME/miniconda3/envs/mineru/bin/mineru-kit \
    --env MINERU_MODEL_ONNX_DEVICE=gpu --env MINERU_LLAMA_CPP_BACKEND=vulkan \
    --env GGML_VK_VISIBLE_DEVICES=2 --watch-gpus 0,1

python quality.py --bench-root $B
python report.py --bench-root $B
```

Run configurations one after another (not concurrently) so they do not compete
for the GPU. On a shared machine record which GPUs were busy: the per-document
`mean_gpu_util_pct` column reports utilisation of the watched GPUs including
other users' jobs.

## Caveats

* The quality reference is the PDF text layer, which keeps running
  headers/footers and flattens tables; use the metrics to compare backends with
  each other, not as absolute accuracy. Table cell accuracy needs manual review
  of `reports/tables/`.
* MinerU runs as a subprocess per document, so its per-document time includes
  model loading; Marker loads models once per process (reported as
  `model_init_s`).
* Concurrency is measured with N worker processes (each loads its own
  models). Two Marker conversions in one process at the same time abort with
  glibc "corrupted double-linked list" — do not call `parse_pdf(backend="marker")`
  concurrently from threads of one process.
* Long SI documents can be trimmed for the benchmark with
  `prepare_cases.py --max-si-pages N` (trimmed copies go to `derived/`).

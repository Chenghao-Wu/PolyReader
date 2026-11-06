# Robustness cycle (Nov 2026): audit, fixes, DeepSeek API, optional MinerU

> Module names in this record (`paperindex.…`, `polyextract.…`) refer to the layout at the time. Both packages were merged into `polyreader` afterwards — see README → Project Structure for where each module lives now.

Implements `plan_v0.md` (robustness improvement + optional MinerU backend). The
extraction architecture, prompts, Pydantic schemas and JSON output formats are
unchanged; Marker stays the default parser.

## 1. Baseline (Stage 1)

Run on the unmodified code (`382048c`), `pytest tests/` on XJTLU_SA264_DYVLINGE
(Python 3.11, torch 2.7.1+cu126, marker-pdf 1.10.2):

| | Baseline | After this cycle |
|---|---|---|
| Passed | 306 | 497 |
| Failed | 10 | 0 |
| Collection errors | 3 | 0 |
| Skipped | 2 | 5 (legacy scripts / opt-in Marker run) |
| Expected failures | 0 | 7 (quality-tier rules, see §5) |

### Outdated tests (fixed or skipped)

| Test | Problem | Action |
|---|---|---|
| `test_document_set::test_available_documents` | `DocumentSet.available_documents()` no longer exists | uses `labels()` |
| `test_schema::test_extraction_result` | `ExtractionResult.paper_doi` replaced by `metadata` | uses `metadata` |
| `test_schema::test_valid_sample_no_flags` | predates the SMILES checks | sample now has SMILES |
| `test_entity_discovery` (import) | `dispatch_entity_tool` moved to `tools.dispatch_tool` | import updated |
| `test_process_document` (import) | needs removed `paperindex.query`, Zhipu key, `test.pdf` | skipped cleanly |
| `test_tree_extraction` (import) | raises without `MOONSHOT_API_KEY` | skipped cleanly |
| `test_end_to_end`, `test_orchestrator` (1 test) | need `ZHIPU_API_KEY` | still skipped |
| 7 quality-tier tests | expect should-have criteria to affect the tier | `xfail`, see §5 |

### Public API compatibility checklist

| API | Status |
|---|---|
| `paperindex.process_document(pdf_path, api_key, base_url, model, **kwargs)` | unchanged; new optional `parse_output_dir`, config keys `pdf_parser_backend`, `mineru`, `request_timeout`, `reasoning_effort` |
| `paperindex.save_index / load_index / print_tree` | unchanged signatures; files written atomically; tree.json gains an optional `"parser"` key; old indexes load |
| `PaperIndex.get_page_content / get_page_image / get_document_structure` | unchanged signatures and result keys |
| `paperindex.core.pdf_parser.parse_pdf(pdf_path, output_dir=None)` | unchanged; new optional `backend=`, backend options; `ParseResult` gains `parser_info` |
| `paperindex.llm.create_llm_client / LLMClient` | unchanged positional/keyword args; new optional `reasoning_effort`, `thinking_disabled`; `timeout=None` = provider default |
| `polyextract.extract / save_samples / PolyextractConfig / DocumentSet` | unchanged; config gains `reasoning_effort`, `thinking_disabled`, `request_timeout`; `DocumentSet.resolve(label)` default behaviour unchanged (new `strict=`) |
| `validate_sample()` output strings | unchanged; codes via new `validate_sample_issues()` |
| CLI flags | all existing flags unchanged; defaults unchanged when `OPENROUTER_API_KEY` is set |
| Sample JSON / pipeline report | unchanged keys; report gains `sample_files`, `llm_usage`, `parser` |

## 2. Confirmed bugs fixed (each has a regression test)

| # | Where | Bug | Fix |
|---|---|---|---|
| 1 | `extraction_loop.py` | Critical-flag retry never fired: it matched codes (`mw_less_than_mn`) against human-readable messages | stable `ValidationIssue` codes; retry on codes; earlier flagged result kept if a retry fails |
| 2 | `schema/validation.py` | MW rescue skipped everything if any of Mn/Mw/PDI was set; ignored kDa / kg/mol | each field recovered independently, duplicates removed, conflicts kept, units converted |
| 3 | `schema/polymer.py` | `"12,300"`-style strings crashed validators with `TypeError`; non-numeric property values invalidated the whole sample | tolerant numeric parsing; non-numeric values kept in `value_str` |
| 4 | `schema/validation.py` | `TGA_residue` / `Tmax` treated as Tg / Tm; Kelvin compared with °C | symbol-exact matching; K converted |
| 5 | `entity_discovery.py`, `guided_extraction.py` | malformed tool-call JSON raised `JSONDecodeError`, aborting discovery (whole paper) | error returned to the LLM as the tool result; loop continues |
| 6 | `entity_discovery.py` | `sample_ids: "P1"` expanded per character; list-shaped `entities` crashed; entities missing from wave lists never extracted | normalisation of the entity map |
| 7 | `tools.py` / `document_set.py` | unknown `doc` label silently read the main paper in main+SI runs | error listing available documents (single-doc behaviour unchanged); file-name labels resolved |
| 8 | `paperindex.get_page_image` | `img_path` joined unchecked → path traversal (`../`, absolute, symlinks) | resolved path must stay inside the images dir, image extensions only |
| 9 | `paperindex.get_page_content` | `"1-100000000"` built a 10⁸-element set | ranges clamped to the document |
| 10 | `paperindex/llm/client.py` | `chat()` retried permanent errors (401/402/400); `chat_with_tools()` had no retry; `"Request timed out."` and 500/504 not classified as transient | type/status-based classification, bounded backoff with Retry-After |
| 11 | `orchestrator.py` | `extract()` mutated the caller's config | works on a copy |
| 12 | `cli.py`, `save_samples` | non-atomic JSON writes; sample IDs that sanitise alike overwrote each other | atomic writes; `_2`, `_3` suffixes |
| 13 | `cli.py --skip-index` | skipped all indexing if the main tree existed, silently dropping an un-indexed SI; reused corrupt/incomplete indexes | per-document reuse after `verify_index()` and parser check |
| 14 | `entity_discovery.py` | transport failures on every attempt returned an empty entity map (paper reported as "no polymers") | raises the transport error |
| 15 | `entity_discovery.py` | a transient error in the max-iterations wrap-up escaped all retries and aborted the paper | handled like other transport errors (next discovery attempt) |
| 16 | `cli.py` | a failure while classifying main vs. SI crashed the CLI without a pipeline report | falls back to positional order, reported in `document_classification_error` |
| 17 | `parsers/marker_parser.py` | two Marker conversions in one process (e.g. concurrent `process_document` calls) crash with heap corruption | conversions serialized with a lock |

## 3. Execution resilience (Stage 5)

- LLM calls: transient errors (connection, timeout, 408/409/425/429, 5xx,
  empty/aborted responses, blank content) retried with 1-2-4 s… backoff capped
  at 60 s, honouring `Retry-After`; permanent errors raised immediately.
  Existing retry counts kept (chat 3, vision 5); tool calls and structured
  output now also retry transient errors (3).
- Per-sample isolation: an exception in one sample is recorded as `fatal` for
  that sample; the others continue. Retries exhausted by transport errors
  report `transport_error` instead of `validation_failed`.
- Files: index, page contents, samples and reports are written atomically
  (temp file + rename); page contents are written before the tree, so an
  existing tree always has its pages; image directories are replaced only after
  the new copy is in place. Missing/corrupt indexes raise `IndexLoadError` /
  are reported by `verify_index()`.

## 4. DeepSeek API

See README → "DeepSeek API". Implemented against the API documentation of
Nov 2026 (DeepSeek-V4.1-Flash, model `deepseek-flash`): explicit thinking toggle,
`reasoning_effort`, JSON Output for structured responses in thinking mode,
`reasoning_content` preserved on tool turns, 600 s timeout, retired model names
mapped, vision only for `deepseek-flash`, token usage (incl. reasoning and cache
hits) reported in the pipeline report.

Verified live (Nov 6, 2026): full CLI runs (Marker or MinerU → index →
discovery → per-sample tool-calling extraction with images → JSON-mode
structured output) on all five benchmark papers with `deepseek-flash` in
thinking mode completed without API errors; one 11-page paper costs ≈2.3 CNY
(≈1.1 M prompt tokens, ~75 % cache hits) at peak-hour prices.

## 5. Open question: quality-tier rules

`classify_quality_tier()` computes the should-have criteria (synthesis method +
condition, processing history) but assigns tiers from the must-have count only
(4 → gold, 3 → silver, 2 → bronze). Seven existing tests expect the should-have
criteria to separate gold from silver. Both came in the same commit, so the
intent is unclear. The behaviour is unchanged (it affects only the tier label,
not which samples are written); the tests are marked `xfail` until decided.

## 6. Marker vs MinerU benchmark (5 papers)

Corpus (106 pages): P1 Nat. Commun. 2024 ROP polyesters + 34-page SI; P2
Polymers 2023 linear/star SIS + SI; P3 Polymers 2023 TMC ROP (dense Mn/Đ
tables); P4 Polymers 2024 carbonate copolymers rasterised to an image-only
"scanned" PDF; P5 PCCP 2024 MD study + SI (negative control: nothing
synthesised). Full reports: `benchmarks/parser_bench/results/` and
`benchmarks/downstream/results/`.

Software: marker-pdf 1.10.2 (surya 0.17.1, torch 2.7.1+cu126); MinerU 4.1.4
(`mineru-kit`, ONNX + llama.cpp on Vulkan, driver 535). Hardware: RTX 3090
(XJTLU_SA264_DYVLINGE, shared with other users' jobs, 45–85 % busy) and an
idle RTX 4090 D (XJTLU_SA264) for Marker.

### Speed and memory

| config | GPU | s/page | cold start (first doc) | peak VRAM / process | peak RAM |
|---|---|---|---|---|---|
| Marker | 3090 (shared) | 7.39 | 231 s | ~9.5 GB | 3–4 GB (10.5 GB on the 34-page SI) |
| Marker | 4090 D (idle) | 2.44 | 82 s | ~9.5 GB | 3–4 GB |
| MinerU standard | 3090 (shared) | 5.89 | 145 s | ~3.5 GB | 2–3 GB |
| MinerU basic | 3090 (shared) | 5.39 | 166 s | ~3.3 GB | 2–3 GB |
| MinerU advanced | 3090 (shared) | 11.16 | 199 s | ~3.5 GB | 2–4 GB |

Absolute 3090 speeds are inflated by other users' jobs; MinerU reloads its
models for every document (subprocess), Marker once per process.

### Document reconstruction (automatic, vs. PDF text layer)

| config | token recall | token precision | page mapping | tables / cells | symbol retention |
|---|---|---|---|---|---|
| Marker | 0.971 | 0.974 | 1.000 | 17 / 1758 | 0.937 |
| MinerU standard | 0.966 | 0.964 | 0.991 | 17 / 1547 | 0.946 |
| MinerU basic | 0.957 | 0.932 | 0.991 | 17 / 1613 | 0.731 |
| MinerU advanced | 0.971 | 0.877 | 0.984 | 50 / 3824 | 0.959 |

All configurations kept the page count of every document and produced no
broken image references. Manual table inspection: MinerU standard/basic
scramble the columns of P2 Table 1 (Marker and MinerU advanced reproduce it);
MinerU advanced over-detects tables in NMR-spectra SI pages; Marker
occasionally misreads math symbols (e.g. Đ as `\theta`). The scanned P4 was
reconstructed well by both (recall 0.98–0.99).

### Downstream Polyextract extraction (identical LLM `deepseek-flash`, prompts, config)

| config | sample-ID P / R / F1 | value P / R / F1 (Mn, Mw, PDI, Tg, Tm) | prompt tokens |
|---|---|---|---|
| Marker | 0.85 / 0.98 / 0.91 | 0.97 / 0.96 / 0.96 | 11.6 M |
| MinerU standard | 0.86 / 1.00 / 0.93 | 0.98 / 0.97 / 0.98 | 11.5 M |

Per paper: P3 and P4 (scanned) were extracted perfectly with both parsers.
The only parser-attributable value error was MinerU's scrambled P2 Table 1
(sample L1 received L3's Mn/Đ); Marker missed sample L3 of P2 and produced one
duplicate record in P1. Both parsers let the LLM read values from figures
(e.g. Tg per run from the SI DSC plot of P1). Both runs extracted the eight
simulated polymers of the negative-control paper P5 — an entity-triage
weakness independent of the parser. With one LLM run per configuration,
differences of one or two samples are within run-to-run variation.

### Recommendation

Keep Marker as the default: equal downstream accuracy on this set, fastest on
an idle GPU, no extra environment. MinerU standard is a sound alternative
where GPU memory is tight (~3.5 GB vs ~9.5 GB) or a CUDA-capable torch is not
available (it runs on Vulkan); avoid MinerU basic (symbol loss) and advanced
(table over-detection, slowest) for extraction. Five papers are not enough to
claim a difference in extraction accuracy; no backend migration is implied.

## 7. Workstation environments

- `polyextract` conda env (Python 3.11, torch 2.7.1+cu126 from the aliyun
  mirror, marker-pdf 1.10.2, editable install of this repo) on
  XJTLU_SA264 and XJTLU_SA264_DYVLINGE; Marker models in
  `~/.cache/datalab/models` on both.
- MinerU runs from the existing `mineru` env on DYVLINGE
  (`--mineru-executable ~/miniconda3/envs/mineru/bin/mineru-kit`, with
  `MINERU_MODEL_ONNX_DEVICE=gpu MINERU_LLAMA_CPP_BACKEND=vulkan
  GGML_VK_VISIBLE_DEVICES=<raw Vulkan index>`; NVIDIA GPU 0,1,2,3 = Vulkan 0,2,3,4).

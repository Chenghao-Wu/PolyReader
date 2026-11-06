# Design proposal: accelerating extraction (and what to take from PageIndex)

Status: Lane F implemented 2026-11-06 as `--extraction-mode full_context|auto`, plus a provider-aware concurrency default. The LLM-free heading-tree indexing was prototyped and benchmarked but **not adopted** (accuracy): indexing keeps the LLM tree builder and node summaries. Results: `docs/accelerated-extraction-results.md`.

## 1. Where the time and tokens go today

Measured on the 5-paper benchmark (`deepseek-flash`, thinking, Marker and
MinerU runs, logs in `~/polyreader/bench/downstream/*/deepseek/`):

| paper | pages (main + SI) | paper size | index s | extract s | samples (tasks) | LLM requests | prompt tokens | page reads / unique pages |
|---|---|---|---|---|---|---|---|---|
| P1 | 12 + 34 | 33k tok | 124 | 538 | 31 | 227 | 6.40 M | 544 / 36 (15×) |
| P2 (MinerU) | 14 + 6 | 19k tok | 91 | 1705 | 28 | 204 | 5.59 M | 439 / 20 (22×) |
| P2 (Marker) | 14 + 6 | 19k tok | 94 | 528 | 22 | 134 | 2.69 M | 134 / 20 (7×) |
| P3 | 11 | 13k tok | 53 | 231 | 16 | 86 | 1.13 M | 129 / 11 (12×) |
| P4 | 11 | 13k tok | 75 | 154 | 9 | 49 | 0.68 M | 80 / 9 (9×) |
| P5 | 12 + 6 | 16k tok | 81 | 180 | 8 | 45 | 0.75 M | 78 / 14 (6×) |

- Entity discovery takes 20–48 s; **per-sample extraction is 65–95 % of wall time**.
- Every sample runs its own agent loop: 5–7 tool rounds (mean) plus a final
  structured call. Samples of one table re-read the same pages; each page is
  read 6–22× per paper.
- Each round re-sends the whole conversation, so prompt tokens reach
  **45–290× the size of the paper** (P1: 6.4 M tokens for a 33k-token paper).
  ~75 % are prompt-cache hits, which hides part of the cost, not the latency.
- `get_document_structure` is called 9–53× per paper: the per-sample
  extraction prompt does not contain the structure (only the discovery prompt
  does), so every sample fetches it once or more.
- Indexing (LLM tree building + node summaries + metadata) is 15–35 % of wall
  time; for papers this small the tree is barely needed.

The documents are small: 13–33k tokens including SI, against a 1 M-token
context window (`deepseek-flash`). Navigating them page by page per sample
is the dominant cost.

## 2. What PageIndex (VectifyAI/PageIndex, MIT, v0.5.3) offers

| PageIndex component | What it does | Relevance |
|---|---|---|
| Classic tree builder + node summaries + 3 navigation tools | the design `polyreader.indexing` (formerly `paperindex`) was derived from (same `node_id`/`start_index`/`end_index` tree format, same `get_document_structure`/`get_page_content` tools) | already in use |
| "Structure first only above 20 pages" (`STRUCTURE_FIRST_PAGE_THRESHOLD = 20`); for ≤ 5 pages "extract all content" | their own agent skips tree navigation for short documents | **adopt the idea** — it is exactly our case |
| PageIndex Flash (`pageindex.flash`) | LLM-free tree from PDF layout (pdfium char-level parsing, heading detection, bookmarks) | useful for born-digital long documents; local mode cannot read scans (`toc_source: "unreadable"`) |
| `tree_optimize` merge (deterministic) / expand (LLM) | reshapes the tree to minimise worst-case pages scanned per lookup; keeps merged titles as `key_items` | useful for the agentic lane on long documents |
| `md_to_tree` | tree from Markdown headings | adapt per page to Marker/MinerU output (works for scans) |
| Tool output budget (100k chars, structure pagination) | bounds tool results | small hardening |
| Cloud: OCR, block-level citations, search, MCP | hosted service | not adopted: documents leave the premises, paid |
| Chat agent (OpenAI Agents SDK via LiteLLM) | one question → answer loop | not suited to batch record extraction |

Dependency cost: `import pageindex` (also for `pageindex.flash`) eagerly loads
the SDK client, which requires `litellm>=1.105.1,<1.106`, `openai-agents`,
`mcp`, `openai>=2.45`. Recommendation: **do not take the SDK as a runtime
dependency**; adopt the ideas, and offer Flash only as an optional extra if
long born-digital documents become common.

PageIndex does not address our bottleneck (many records per small document);
its main contribution here is the evidence that short documents should be read
whole, not navigated.

## 3. Proposed design: two-lane extraction

```text
              ┌────────────── doc set tokens ≤ FULL_CONTEXT_BUDGET (e.g. 80k) ──────────────┐
PDF → parse → │ Lane F (new; extraction_mode auto | full_context)                          │ → validate → reconcile → outputs (unchanged)
              │  F0 pack paper once  F1 discovery+evidence (1 call)  F2 group  F3 batch extract │
              └────────────── otherwise, or per-sample fallback ──────────────────────────────┘
                Lane A (existing agentic per-sample extraction, with cheap speed-ups)
```

### Lane F — full-context, retrieve once, extract many

- **F0 Pack.** Build a deterministic `PaperContext` string from the page texts
  of all documents: `<<doc=main page=3>>` markers + page text, plus a
  table/figure inventory (captions, image file names). Identical bytes for
  every call of a paper → a stable prompt prefix that the provider caches
  (DeepSeek caches prefixes automatically; OpenRouter models via their own
  caching).
- **F1 Discovery (1 call, JSON).** Same triage rules and topology vocabulary
  as today's discovery prompt, but the paper is in context (no tools). Output =
  today's entity map **plus evidence pointers** per sample:
  `{"doc", "pages", "kind": table|text|figure, "ref": "Table 1, row S3"}`.
- **F2 Group.** Batch samples that share evidence (same entity/table), 3–5 per
  batch. Small batches keep output latency low; the cached prefix makes many
  parallel batches cheap.
- **F3 Batch extraction (parallel, JSON).** Prefix + the existing schema
  guidance + "extract samples [S1..S5] from Table 1 (p.3)" →
  `BatchResult{samples: list[PolymerSample]}`. Figure evidence: attach those
  images (vision) to the batch call, or allow ≤ 2 `get_page_image` turns.
  Base-polymer context for derived entities is passed exactly as today.
- **F4 Validate / fall back.** Each returned sample goes through the existing
  post-processing (`rescue_mw_from_properties`, `validate_sample_issues`,
  SMILES canonicalisation, tier). Samples that are missing from a batch
  result, fail validation or carry critical flags are re-extracted with the
  **existing per-sample agentic path** (Lane A). Lane F can therefore only add
  speed, not lose coverage.
- **Indexing in Lane F.** Unchanged (LLM tree builder + node summaries). A
  Markdown-heading tree without summaries was prototyped (≈10× faster
  indexing) but not adopted for accuracy reasons.

### Lane A — existing agentic path, cheap speed-ups (also useful alone)

1. Raise default concurrency (`max_concurrent` 3 → 8–12). DeepSeek allows
   2,500 concurrent requests per account; wall time on 20–30-sample papers
   drops ~2–3× with no accuracy risk.
2. Put the document structure into the per-sample prompt and drop the
   `get_document_structure` tool (saves 9–53 calls per paper). Not implemented:
   it changes the default lane's prompt, and Lane F removes the need.
3. Per-entity evidence sharing: once one sample of an entity is done, its
   provenance pages are pre-loaded for its siblings (fewer rounds). Not
   implemented (Lane F subsumes it).
4. For long documents: deterministic Markdown-heading tree + PageIndex-style
   merge (worst-case pages-to-scan cost) instead of LLM tree building. Not
   adopted (accuracy).

### Interfaces and compatibility

- New config / CLI: `extraction_mode = auto | full_context | agentic` (until
  validated, default `agentic` = today's behaviour; switch the default to
  `auto` only after the gate in §5 passes), `full_context_budget` (tokens),
  `batch_size`.
- Unchanged: `PolymerSample` schema, sample JSON, pipeline report keys (gains
  `extraction_mode` and per-lane counts in the trace), `extract()` signature,
  DocumentSet, validation and reconciliation.
- Prompt content (triage gate, schema guidance, topology vocabulary) is
  reused verbatim; only the delivery changes (paper in context instead of
  tools). This is an orchestration change — it was out of scope for the
  robustness cycle and is proposed here as a separate step.

## 4. Expected effect (estimates, to be measured)

Per paper, Lane F needs ~1 discovery call + ⌈samples / batch⌉ batch calls +
a few fall-backs, each with a ~15–40k-token mostly cached prefix:

| paper | requests now → Lane F | prompt tokens now → Lane F | wall time now → Lane F |
|---|---|---|---|
| P1 (31 samples) | 227 → ~12 | 6.4 M → ~0.5 M | 11 min → ~3 min |
| P2 (22–28) | 134–204 → ~10 | 2.7–5.6 M → ~0.3 M | 10–30 min → ~3 min |
| P3 (16) | 86 → ~7 | 1.1 M → ~0.15 M | 4.7 min → ~2 min |

Completion tokens (thinking + JSON) shrink less (~1.5–2×), so cost should
fall roughly 3–8× and wall time 2–10×. Indexing time drops by the LLM tree
building and summaries (40–130 s per paper).

Risks: multi-record calls can drop or swap rows ("lost in the middle"),
which is mitigated by small batches, explicit row lists from the evidence
pointers, and the per-sample fallback. Long-context reasoning quality must be
shown on our data, not assumed.

## 5. Validation gate

Rerun the 5-paper benchmark (same Marker parses, `deepseek-flash`) for
`agentic` vs `full_context`, twice each to see run-to-run variation, and
switch the default only if Lane F keeps
- value F1 ≥ 0.95 and sample-ID recall ≥ 0.98 (today 0.96 / 0.98),
- no more false positives on the negative control (P5),
- provenance page accuracy ≥ 0.95,
while reporting requests, prompt/completion tokens, cost and wall time.
Estimated API cost of the gate: ~10–15 CNY (the Lane F arm is cheap; the
agentic arm reruns dominate).

## 6. Effort

Lane F: packer + evidence-aware discovery + batch schema/grouping +
fallback wiring + Markdown-heading tree ≈ 600–800 lines with tests. Lane A
speed-ups ≈ 100 lines. Optional Flash integration (extra) ≈ 100 lines.

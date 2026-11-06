# Accelerated extraction: results (2026-11-06)

Implements `docs/design-accelerated-extraction.md`: `--extraction-mode
full_context|auto` (whole paper in the prompt, discovery with evidence
pointers, batched per-entity extraction with per-sample validation and
fallback) and a DeepSeek concurrency default of 8. The default mode is
unchanged (`agentic`). A fast indexing option (section tree from Markdown
headings, no node summaries; `fast_e2e*` rows below) was evaluated and **not
adopted** for accuracy reasons: indexing keeps the LLM tree builder and node
summaries.

## Setup

Same 5 papers, Marker parses and ground truth as `robustness-cycle-2026-11.md`
§6; `deepseek-flash` (thinking) for every arm. Extraction-only arms reuse the
same LLM-built indexes (index time 427 s for all 5 papers); the `fast_e2e*`
arms (evaluated, not adopted) indexed with the heading-tree prototype instead. Arms ran in
parallel on DYVLINGE; wall times are the sum over the 5 papers. Scores:
`benchmarks/downstream/results/2026-11-06_accel_scores.json`.

## Results

| run | mode | sample-ID P / R / F1 | value P / R / F1 | requests | prompt tokens | completion tokens | extraction s | index s | P5 false positives |
|---|---|---|---|---|---|---|---|---|---|
| agentic_r1 | agentic, conc. 3 | 0.85 / 0.98 / 0.91 | 0.97 / 0.95 / 0.96 | 541 | 11.6 M | 490 k | 1631 | 427 | 8 |
| agentic_r2 | agentic, conc. 3 | 0.86 / 1.00 / 0.93 | 1.00 / 1.00 / 1.00 | 420 | 8.3 M | 365 k | 1631 | 427 | 8 |
| agentic_c8 | agentic, conc. 8 | 0.86 / 1.00 / 0.93 | 1.00 / 1.00 / 1.00 | 405 | 8.0 M | 366 k | 1272 | 427 | 8 |
| full_context_r1 | full_context | 0.86 / 1.00 / 0.93 | 1.00 / 0.97 / 0.99 | 44 | 1.0 M | 378 k | 658 | 427 | 8 |
| full_context_r2 | full_context | 0.86 / 1.00 / 0.93 | 1.00 / 0.97 / 0.99 | 46 | 1.1 M | 422 k | 704 | 427 | 8 |
| full_context_r3* | full_context | 1.00 / 0.92 / 0.96 | 1.00 / 0.93 / 0.96 | 33 | 0.8 M | 304 k | 604 | 427 | 0 |
| full_context_r4* | full_context | 0.82 / 1.00 / 0.90 | 1.00 / 1.00 / 1.00 | 49 | 1.2 M | 405 k | 602 | 427 | 8 |
| fast_e2e | heading tree (not adopted) + full_context | 1.00 / 1.00 / 1.00 | 1.00 / 0.97 / 0.99 | 34 | 0.8 M | 263 k | 548 | 50 | 0 |
| fast_e2e2* | heading tree (not adopted) + full_context | 1.00 / 1.00 / 1.00 | 1.00 / 1.00 / 1.00 | 58 | 1.2 M | 327 k | 634 | 52 | 0 |

`*` with characterization figures attached to batch calls (final code).
Sample-ID precision is dominated by the negative-control paper P5 (8 simulated
polymers extracted or not); it is not a parser or lane effect.

Measured spend: the three final full-context/fast runs together cost 9.5 CNY
(≈ 3.2 CNY per 5-paper run, ≈ 0.6 CNY per paper); the first batch of arms
(2 agentic + 3 full-context) cost 33.9 CNY, i.e. ≈ 12 CNY per agentic 5-paper
extraction run. Indexing with the LLM tree builder adds 427 s for the 5
papers.

### Efficiency (full_context / fast vs agentic)

- LLM requests: 405–541 → 33–58 per 5 papers (**~10× fewer**).
- Prompt tokens: 8.0–11.6 M → 0.8–1.2 M (**~9× fewer**); completion tokens
  (mostly thinking) only ~1.2× fewer.
- Cost: ≈ 12 → ≈ 3.2 CNY per 5 papers (**~3.8× cheaper**).
- Wall time, extraction: 1272–1631 s → 602–704 s (**2.3–2.7× faster**);
  end-to-end with the standard LLM indexing (427 s) 1699–2058 s →
  1029–1131 s (**1.5–2.0×**).
- Concurrency 8 alone (agentic_c8): 1631 → 1272 s; tokens unchanged.

### Accuracy gate (design §5)

| criterion | agentic (3 runs) | full_context (4 runs; + 2 heading-tree runs) |
|---|---|---|
| value F1 ≥ 0.95 | 0.96, 1.00, 1.00 | 0.99, 0.99, 0.96, 1.00, 0.99, 1.00 |
| sample-ID recall ≥ 0.98 | 0.98, 1.00, 1.00 | 1.00, 1.00, **0.92**, 1.00, 1.00, 1.00 |
| P5 false positives (no more than agentic) | 8, 8, 8 | 8, 8, 0, 8, 0, 0 |
| provenance pages correct ≥ 0.95 | 1.00 | 0.98–1.00 |
| completeness (properties / sample, gold %) | 5.3–6.8, 59–82 % | 4.2–5.5, 71–99 % |

Five of six full-context runs pass every criterion (3 of the 4 runs on the
standard indexes). In full_context_r3,
discovery left out four P1 runs with Mn 0.9–1.3 kg/mol (the paper calls
products below 2 kg/mol "oligo(EVP)ROP"); the agentic runs kept them. The
P5 negative control is triaged inconsistently in both modes.

### Issues found and fixed while benchmarking

- Batch outputs were terse (P3: 2 vs 46 secondary properties) → the batch
  task now requires each sample object to be as complete as a single-sample
  extraction; completeness metrics were added to the evaluator.
- Discovery invented sample IDs ("T2R1") → it must use the paper's labels.
- Values only legible in figures (P1 SI DSC Tg labels) were missed → batch
  calls attach characterization figures (DSC/TGA/DMA/stress–strain/SAXS first,
  then GPC/MALDI) whose captions mention the entity, a target or its table.

## Recommendation

Use `--extraction-mode auto` for typical articles (+ SI), with the standard
indexing: same accuracy as the agentic lane within run-to-run variation,
~10× fewer requests, ~4× lower extraction cost, ~2.5× faster extraction
(~1.5–2× end to end). Documents above the context budget use `agentic`
automatically. The default is still `agentic`;
switching it is a one-line change once you are comfortable with the remaining
risk (discovery occasionally drops borderline samples, as in r3).

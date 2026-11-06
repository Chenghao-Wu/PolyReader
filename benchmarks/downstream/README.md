# Downstream comparison: does the PDF parser change what Polyextract extracts?

Runs the unchanged Polyextract pipeline (same LLM, prompts, schemas,
configuration, validation) on the Markdown each parser backend produced in
`../parser_bench`, and scores the emitted samples against a manually
transcribed ground truth.

| file | purpose |
|---|---|
| `ground_truth/<case>.json` | sample IDs (+ aliases), accepted Mn/Mw values, PDI, Tg/Tm and source pages, transcribed from the papers' tables, text and (where stated) figures. `optional` samples, `optional_fields`, `optional_patterns` and `unscored_table_patterns` mark data that is correct if extracted but not required (e.g. intermediates, values only readable from a plot). Each file's `source` says what was transcribed. |
| `run_downstream.py` | for each configuration and case: index every PDF from the saved parse (`saved_parse.py`), then run `polyreader ... --skip-index --pdf-parser <backend>` |
| `evaluate.py` | sample-ID precision/recall/F1 (ID/alias matching) and value precision/recall/F1 for Mn, Mw, PDI, Tg, Tm (Mn/Mw ±3 %, PDI ±0.015, T ±0.6 °C); Mw = Mn×PDI is accepted as derived |
| `results/` | scores of the Nov 2026 run |

```bash
export DEEPSEEK_API_KEY=...
python run_downstream.py --parse-root ~/polyreader/bench/outputs --papers ~/polyreader/bench/papers \
    --configs marker mineru_standard -o ~/polyreader/bench/downstream
python evaluate.py --runs ~/polyreader/bench/downstream/marker/deepseek \
    ~/polyreader/bench/downstream/mineru_standard/deepseek -o scores.json
```

Cost note: with `deepseek-flash` (thinking) one configuration over the five
papers used ~11.5 M prompt tokens (~75 % cache hits) and ~0.5 M completion
tokens; the paper with a 34-page SI alone was about half of that.

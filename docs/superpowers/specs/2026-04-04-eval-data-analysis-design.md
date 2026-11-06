# Evaluation Data Organization & Analysis

## Overview

Normalize three heterogeneous evaluation spreadsheets (xinyan, yuqi, yuwei) into
unified CSVs, then produce publication-ready analysis and pipeline feedback.

## Data Sources

| File | Annotator | Papers | Format |
|------|-----------|--------|--------|
| `xinyan.xlsx` | Xinyan | 23 | Fraction strings ("13/14"), alternating recall/precision columns, 29 cols |
| `yuqi.xlsx` | Yuqi | ~15 | Dot-fraction strings ("5./5."), per-sample rows, annotator notes, 36 cols |
| `yuwei.xlsx` | Yuwei | 21 | Binary 1/0 per sample, merged 2-level headers, 109 cols, 2 sheets |

No paper overlap between annotators. No cross-annotator comparison needed.
Data source is spreadsheets only (no linkage to `output/` extraction JSONs).

## Project Structure

```
paper/
├── data/
│   ├── xinyan.xlsx
│   ├── yuqi.xlsx
│   └── yuwei.xlsx
├── scripts/
│   ├── parsers/
│   │   ├── __init__.py
│   │   ├── base.py          # PaperPropertyRecord dataclass + shared types
│   │   ├── xinyan.py         # Xinyan fraction-pair parser
│   │   ├── yuqi.py           # Yuqi dot-fraction + multi-row parser
│   │   └── yuwei.py          # Yuwei binary 1/0 merged-header parser
│   ├── unify.py              # Runs all parsers, produces CSVs
│   └── utils.py              # Fraction parsing, metric computation, failure classification
├── notebooks/
│   ├── analysis.ipynb        # Metrics, breakdowns, failure patterns
│   └── figures.ipynb         # Publication-ready visualizations
└── output/
    ├── layer1_paper_summary.csv
    ├── layer2_yuwei_samples.csv
    ├── failure_analysis.csv
    └── figures/              # PNG visualizations
```

## Data Schema

### Layer 1 — Per-Paper Summary (`layer1_paper_summary.csv`)

One row per paper per property per annotator.

| Column | Type | Description |
|--------|------|-------------|
| `annotator` | str | "xinyan", "yuqi", "yuwei" |
| `paper_id` | str | Paper identifier (normalized per annotator) |
| `property` | str | Standardized property name |
| `true_count` | int | Ground truth count |
| `pred_count` | int | Predicted count |
| `matched_count` | int | Correct predictions |
| `precision` | float | matched / pred (null if pred=0) |
| `recall` | float | matched / true (null if true=0) |
| `f1` | float | Harmonic mean of precision and recall |
| `notes` | str | Annotator notes/comments (if any) |

Standardized property names:

```
sample, smiles, constituents, mole_fraction, topology,
mn, mw, pdi, tg, tm, density, youngs_modulus,
synthesis, processing_history, distribution, properties
```

Properties not present in an annotator's spreadsheet produce no rows.

### Layer 2 — Yuwei Per-Sample Detail (`layer2_yuwei_samples.csv`)

One row per sample per property (long format).

| Column | Type | Description |
|--------|------|-------------|
| `paper_id` | str | Paper identifier |
| `sample_name` | str | Sample identifier |
| `quality_tier` | str | "Gold" or "Silver" from Notes column |
| `property` | str | Property name |
| `true` | int | 1/0 — field exists in ground truth |
| `pred` | int | 1/0 — system extracted this field |
| `matched` | int | 1/0 — extraction is correct |
| `page_correct` | int | 1/0 — page-level source correct (null if N/A) |
| `variant` | str | "excl" or "incl" for Mn/Mw/PDI, null otherwise |

### Failure Analysis (`failure_analysis.csv`)

| Column | Type | Description |
|--------|------|-------------|
| `annotator` | str | Source annotator |
| `paper_id` | str | Paper identifier |
| `property` | str | Property that failed |
| `failure_type` | str | "missed" / "wrong" / "hallucinated" |
| `annotator_note` | str | Notes from yuqi or xinyan if available |

Failure classification logic:
- **missed**: true=1, pred=0 (system didn't extract an existing value)
- **wrong**: true=1, pred=1, matched=0 (system extracted but got it wrong)
- **hallucinated**: true=0, pred=1 (system extracted a nonexistent value)

## Parser Design

### Base (`parsers/base.py`)

```python
@dataclass
class PaperPropertyRecord:
    annotator: str
    paper_id: str
    property: str
    true_count: int
    pred_count: int
    matched_count: int
    notes: str | None = None
```

Each parser exposes `parse(filepath) -> list[PaperPropertyRecord]`.

### Xinyan Parser (`parsers/xinyan.py`)

- Skip row 0 (recall/precision sub-header)
- Column pairs: named column = recall fraction, next "Unnamed" column = precision fraction
  - e.g., "SMILES" col has recall, the following "Unnamed: 3" col has precision for SMILES
- Parse "13/14" → (matched=13, total=14); for recall the total is true_count, for precision the total is pred_count
- Handle compound fractions like "7/7, 13/14" (multiple sub-properties) by summing
- Map column names → standardized property names
- Chinese annotations preserved in notes
- Last unnamed column (col 28) contains general notes (e.g., "derived entities: 0")

### Yuqi Parser (`parsers/yuqi.py`)

- Forward-fill Paper column (blank for continuation rows)
- Parse dot-fractions "5./5." → (matched=5, total=5)
- Aggregate per-sample rows to per-paper totals by summing counts
- Extract "Missing information" and "Supplementary instruction" into notes

### Yuwei Parser (`parsers/yuwei.py`)

Two outputs:
1. **Layer 1**: Read Summary rows → extract True/Pred/Matched counts → PaperPropertyRecord list
2. **Layer 2**: Read Sample rows → melt wide format to long format with (paper_id, sample_name, quality_tier, property, true, pred, matched, page_correct, variant)

Handle merged 2-level header: row 1 = group names, row 2 = field names.
Map Excl/Incl variants for Mn/Mw/PDI.

### Utilities (`utils.py`)

- `parse_fraction("13/14") -> (13, 14)`
- `parse_dot_fraction("5./5.") -> (5, 5)`
- `compute_metrics(true_count, pred_count, matched_count) -> (precision, recall, f1)`
- `classify_failure(true, pred, matched) -> "missed" | "wrong" | "hallucinated" | None`

## Unifier Script (`unify.py`)

1. Call each parser's `parse()` function
2. Compute precision/recall/F1 from counts
3. Generate failure_analysis records from individual failures
4. Write Layer 1, Layer 2, and failure analysis CSVs to `paper/output/`

Run with: `conda run -n paperindex python paper/scripts/unify.py`

## Analysis Notebook (`analysis.ipynb`)

### Section 1 — Overall System Performance
- Aggregate P/R/F1 across all annotators and papers
- Weighted by sample count
- Overall score and per-annotator breakdown

### Section 2 — Per-Property Breakdown
- P/R/F1 for each standardized property
- Rank from best to worst
- Identify weakest properties (pipeline improvement targets)

### Section 3 — Per-Paper Breakdown
- P/R/F1 per paper across all properties
- Identify hardest papers (lowest F1)
- Correlate with number of samples where visible

### Section 4 — Failure Pattern Analysis
- Distribution of missed/wrong/hallucinated overall and per-property
- Mine yuqi's "Missing information" for recurring failure reasons
- Mine xinyan's Chinese annotations for failure context
- Identify systematic failure patterns

### Section 5 — Yuwei Deep Dive (Layer 2)
- Page-level accuracy per property
- Excl vs. Incl comparison for Mn/Mw/PDI
- Gold vs. Silver quality tier comparison

## Figures Notebook (`figures.ipynb`)

| # | Figure | Type | Purpose |
|---|--------|------|---------|
| 1 | Overall performance table | LaTeX table | Summary for paper |
| 2 | Per-property bar chart | Grouped bars (P/R/F1) | Property-level comparison |
| 3 | Property x paper heatmap | Heatmap (F1 values) | Identify problem intersections |
| 4 | Radar/spider plot | Radar chart | Per-property performance profile |
| 5 | Failure type distribution | Stacked bar chart | Missed/wrong/hallucinated per property |
| 6 | Page accuracy bar chart | Bar chart | Yuwei page-level correctness |
| 7 | Excl vs. Incl comparison | Paired bar chart | MW variant impact |
| 8 | Quality tier comparison | Grouped bars | Gold vs. Silver performance |

All figures saved as PNG to `paper/output/figures/`.
Use scientific-schematics skill for publication-quality figure generation.

## Technical Requirements

- Python environment: `paperindex` conda env
- Dependencies: pandas, openpyxl, matplotlib, numpy
- No external data sources; spreadsheets only

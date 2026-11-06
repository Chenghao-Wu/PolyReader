# Evaluation Data Analysis Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Normalize three heterogeneous evaluation spreadsheets into unified CSVs and produce publication-ready analysis with pipeline feedback.

**Architecture:** Per-annotator parser modules produce `PaperPropertyRecord` lists. A unifier script calls all parsers and writes Layer 1 (per-paper summary), Layer 2 (yuwei per-sample detail), and failure analysis CSVs. Two Jupyter notebooks consume the CSVs for analysis and figure generation.

**Tech Stack:** Python 3, pandas, openpyxl, matplotlib, numpy. Conda env: `paperindex`.

**Spec:** `docs/superpowers/specs/2026-04-04-eval-data-analysis-design.md`

---

## File Structure

```
paper/
├── scripts/
│   ├── parsers/
│   │   ├── __init__.py        # Re-exports parse functions
│   │   ├── base.py            # PaperPropertyRecord + YuweiSampleRecord dataclasses
│   │   ├── xinyan.py          # parse_xinyan(filepath) -> list[PaperPropertyRecord]
│   │   ├── yuqi.py            # parse_yuqi(filepath) -> list[PaperPropertyRecord]
│   │   └── yuwei.py           # parse_yuwei(filepath) -> (list[PaperPropertyRecord], list[YuweiSampleRecord])
│   ├── utils.py               # Fraction parsing, metric computation, failure classification
│   └── unify.py               # CLI entry point: runs parsers, writes CSVs
├── notebooks/
│   ├── analysis.ipynb         # Metrics, breakdowns, failure patterns
│   └── figures.ipynb          # Publication-ready visualizations
└── output/                    # Generated artifacts (gitignored)
    ├── layer1_paper_summary.csv
    ├── layer2_yuwei_samples.csv
    ├── failure_analysis.csv
    └── figures/
paper/tests/
└── test_parsers.py            # All parser + utils tests
```

---

### Task 1: Utils — Fraction Parsing and Metric Computation

**Files:**
- Create: `paper/scripts/utils.py`
- Create: `paper/tests/test_parsers.py`

- [ ] **Step 1: Write tests for fraction parsing utilities**

```python
# paper/tests/test_parsers.py
import pytest
from paper.scripts.utils import parse_fraction, parse_dot_fraction, compute_metrics, classify_failure


class TestParseFraction:
    def test_simple(self):
        assert parse_fraction("13/14") == (13, 14)

    def test_backslash(self):
        assert parse_fraction("14\\14") == (14, 14)

    def test_whitespace(self):
        assert parse_fraction(" 13/14 ") == (13, 14)

    def test_zero(self):
        assert parse_fraction("0/5") == (0, 5)

    def test_not_applicable(self):
        assert parse_fraction("/") is None

    def test_nan(self):
        assert parse_fraction(float("nan")) is None

    def test_none(self):
        assert parse_fraction(None) is None

    def test_numeric_one(self):
        # xinyan uses bare 1 in some precision columns
        assert parse_fraction(1) is None

    def test_compound_fraction(self):
        # "7/7, 13/14" -> sum both: (7+13, 7+14) = (20, 21)
        assert parse_fraction("7/7, 13/14") == (20, 21)

    def test_total_row_format(self):
        # "312/346=90.2%" -> (312, 346)
        assert parse_fraction("312/346=90.2%") == (312, 346)


class TestParseDotFraction:
    def test_simple(self):
        assert parse_dot_fraction("2./3.") == (2, 3)

    def test_perfect(self):
        assert parse_dot_fraction("5./5.") == (5, 5)

    def test_zero(self):
        assert parse_dot_fraction("0./0.") == (0, 0)

    def test_whitespace(self):
        assert parse_dot_fraction(" 10./10. ") == (10, 10)

    def test_nan(self):
        assert parse_dot_fraction(float("nan")) is None

    def test_none(self):
        assert parse_dot_fraction(None) is None

    def test_regular_fraction_fallback(self):
        # some yuqi cells use regular fraction format
        assert parse_dot_fraction("72/72") == (72, 72)


class TestComputeMetrics:
    def test_perfect(self):
        p, r, f1 = compute_metrics(true_count=10, pred_count=10, matched_count=10)
        assert p == 1.0
        assert r == 1.0
        assert f1 == 1.0

    def test_no_predictions(self):
        p, r, f1 = compute_metrics(true_count=5, pred_count=0, matched_count=0)
        assert p is None
        assert r == 0.0
        assert f1 == 0.0

    def test_no_ground_truth(self):
        p, r, f1 = compute_metrics(true_count=0, pred_count=3, matched_count=0)
        assert p == 0.0
        assert r is None
        assert f1 == 0.0

    def test_partial(self):
        p, r, f1 = compute_metrics(true_count=10, pred_count=8, matched_count=6)
        assert p == pytest.approx(0.75)
        assert r == pytest.approx(0.6)
        assert f1 == pytest.approx(2 * 0.75 * 0.6 / (0.75 + 0.6))


class TestClassifyFailure:
    def test_missed(self):
        assert classify_failure(true=1, pred=0, matched=0) == "missed"

    def test_wrong(self):
        assert classify_failure(true=1, pred=1, matched=0) == "wrong"

    def test_hallucinated(self):
        assert classify_failure(true=0, pred=1, matched=0) == "hallucinated"

    def test_correct(self):
        assert classify_failure(true=1, pred=1, matched=1) is None

    def test_absent(self):
        assert classify_failure(true=0, pred=0, matched=0) is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `conda run -n paperindex python -m pytest paper/tests/test_parsers.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'paper.scripts.utils'`

- [ ] **Step 3: Implement utils.py**

```python
# paper/scripts/utils.py
"""Fraction parsing, metric computation, and failure classification utilities."""

from __future__ import annotations

import math
import re


def parse_fraction(value) -> tuple[int, int] | None:
    """Parse fraction string like '13/14', '14\\14', '7/7, 13/14', '312/346=90.2%'.

    Returns (numerator, denominator) or None if not parseable.
    """
    if value is None:
        return None
    if isinstance(value, (int, float)):
        if math.isnan(value) if isinstance(value, float) else False:
            return None
        return None  # bare numbers like 1 are not fractions

    s = str(value).strip()
    if not s or s == "/":
        return None

    # Strip trailing "=XX.X%" from total rows
    s = re.sub(r"=[\d.]+%$", "", s)

    # Handle compound fractions "7/7, 13/14" by splitting on comma
    parts = [p.strip() for p in s.split(",") if p.strip()]

    total_num = 0
    total_den = 0
    for part in parts:
        # Normalize backslash to forward slash
        part = part.replace("\\", "/")
        # Remove any parenthetical annotations like "(未能提取缩写)"
        part = re.sub(r"\(.*?\)", "", part).strip()
        m = re.match(r"^(\d+)\s*/\s*(\d+)$", part)
        if not m:
            return None
        total_num += int(m.group(1))
        total_den += int(m.group(2))

    return (total_num, total_den)


def parse_dot_fraction(value) -> tuple[int, int] | None:
    """Parse dot-fraction string like '2./3.' or '5./5.'.

    Falls back to regular fraction parsing for values like '72/72'.
    Returns (numerator, denominator) or None if not parseable.
    """
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None

    s = str(value).strip()
    if not s:
        return None

    # Try dot-fraction format first: "N./N."
    m = re.match(r"^(\d+)\.\s*/\s*(\d+)\.$", s)
    if m:
        return (int(m.group(1)), int(m.group(2)))

    # Fallback to regular fraction
    return parse_fraction(s)


def compute_metrics(
    true_count: int, pred_count: int, matched_count: int
) -> tuple[float | None, float | None, float]:
    """Compute precision, recall, F1 from counts.

    Returns (precision, recall, f1). Precision is None if pred_count=0.
    Recall is None if true_count=0. F1 is 0.0 if either is None/zero.
    """
    precision = matched_count / pred_count if pred_count > 0 else None
    recall = matched_count / true_count if true_count > 0 else None

    p = precision if precision is not None else 0.0
    r = recall if recall is not None else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    return (precision, recall, f1)


def classify_failure(true: int, pred: int, matched: int) -> str | None:
    """Classify a failure type from binary true/pred/matched values.

    Returns 'missed', 'wrong', 'hallucinated', or None (no failure).
    """
    if true == 1 and pred == 0:
        return "missed"
    if true == 1 and pred == 1 and matched == 0:
        return "wrong"
    if true == 0 and pred == 1:
        return "hallucinated"
    return None
```

- [ ] **Step 4: Create `__init__.py` for paper/scripts and paper/tests**

```python
# paper/scripts/__init__.py
# (empty)
```

```python
# paper/tests/__init__.py
# (empty)
```

```python
# paper/scripts/parsers/__init__.py
# (empty — will be populated in later tasks)
```

Also create a minimal `conftest.py` so pytest can find the modules:

```python
# paper/tests/conftest.py
import sys
from pathlib import Path

# Add project root to path so 'paper.scripts' is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `conda run -n paperindex python -m pytest paper/tests/test_parsers.py -v`
Expected: All 22 tests PASS.

- [ ] **Step 6: Commit**

```bash
git add paper/scripts/__init__.py paper/scripts/utils.py paper/scripts/parsers/__init__.py paper/tests/__init__.py paper/tests/conftest.py paper/tests/test_parsers.py
git commit -m "feat(eval): add fraction parsing and metric utilities with tests"
```

---

### Task 2: Base Dataclasses

**Files:**
- Create: `paper/scripts/parsers/base.py`

- [ ] **Step 1: Create base dataclasses**

```python
# paper/scripts/parsers/base.py
"""Shared data structures for evaluation parsers."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PaperPropertyRecord:
    """One property's evaluation for one paper by one annotator.

    Used in Layer 1 (per-paper summary).
    """
    annotator: str
    paper_id: str
    property: str
    true_count: int
    pred_count: int
    matched_count: int
    notes: str | None = None


@dataclass
class YuweiSampleRecord:
    """One property's evaluation for one sample in yuwei's per-sample detail.

    Used in Layer 2 (yuwei per-sample).
    """
    paper_id: str
    sample_name: str
    quality_tier: str | None
    property: str
    true: int
    pred: int
    matched: int
    page_correct: int | None = None
    variant: str | None = None  # "excl" or "incl" for Mn/Mw/PDI
```

- [ ] **Step 2: Commit**

```bash
git add paper/scripts/parsers/base.py
git commit -m "feat(eval): add PaperPropertyRecord and YuweiSampleRecord dataclasses"
```

---

### Task 3: Xinyan Parser

**Files:**
- Create: `paper/scripts/parsers/xinyan.py`
- Modify: `paper/tests/test_parsers.py`

**Context — xinyan.xlsx structure:**
- 2-row header: row 0 = property names, row 1 = recall/precision sub-headers
- Data starts row 2, ends row 21 (20 papers: paper1..paper20)
- Row 22 = "total", row 23 = "total (exclude derived entities)" — skip both
- Column 0 = paper ID
- Columns 1-27: alternating named (recall) / unnamed (precision) pairs
  - Col 1: "sample" (recall only, no precision pair)
  - Cols 2-3: SMILES (recall/precision)
  - Cols 4-5: constituents (recall/precision)
  - Cols 6-7: topology (recall/precision)
  - Cols 8-9: Mn (recall/precision)
  - Cols 10-11: Mw (recall/precision)
  - Cols 12-13: PDI (recall/precision)
  - Cols 14-15: Tg (recall/precision)
  - Cols 16-17: Tm (recall/precision)
  - Cols 18-19: density (recall/precision)
  - Cols 20-21: Young's modulus (recall/precision)
  - Cols 22-23: Source (recall/precision)
  - Cols 24-25: synthesis (recall/precision)
  - Cols 26-27: processing history (recall/precision)
- Column 28 = free-text notes
- Values: "13/14" or "14\14" (backslash), "7/7, 13/14" (compound), "/" (N/A), bare 1
- Chinese annotations in parentheses: "13/14(未能提取缩写)" — strip parenthetical before parsing

**Property name mapping:**

```python
XINYAN_COL_MAP = {
    1: ("sample", False),       # recall only, no precision pair
    2: ("smiles", True),        # (recall_col, has_precision_pair)
    4: ("constituents", True),
    6: ("topology", True),
    8: ("mn", True),
    10: ("mw", True),
    12: ("pdi", True),
    14: ("tg", True),
    16: ("tm", True),
    18: ("density", True),
    20: ("youngs_modulus", True),
    22: ("source", True),
    24: ("synthesis", True),
    26: ("processing_history", True),
}
```

- [ ] **Step 1: Write tests for xinyan parser**

Add to `paper/tests/test_parsers.py`:

```python
import pandas as pd
from pathlib import Path
from paper.scripts.parsers.xinyan import parse_xinyan

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


class TestXinyanParser:
    def test_returns_list_of_records(self):
        records = parse_xinyan(DATA_DIR / "xinyan.xlsx")
        assert len(records) > 0
        assert all(r.annotator == "xinyan" for r in records)

    def test_paper_ids_are_paper1_through_paper20(self):
        records = parse_xinyan(DATA_DIR / "xinyan.xlsx")
        paper_ids = sorted(set(r.paper_id for r in records))
        # Should have paper1..paper20, no "total" rows
        assert "total" not in paper_ids
        assert "total (exclude derived entities)" not in paper_ids
        assert len(paper_ids) == 20

    def test_has_expected_properties(self):
        records = parse_xinyan(DATA_DIR / "xinyan.xlsx")
        properties = set(r.property for r in records)
        assert "tg" in properties
        assert "mn" in properties
        assert "synthesis" in properties
        assert "smiles" in properties

    def test_counts_are_non_negative(self):
        records = parse_xinyan(DATA_DIR / "xinyan.xlsx")
        for r in records:
            assert r.true_count >= 0, f"{r.paper_id}/{r.property}: true_count={r.true_count}"
            assert r.pred_count >= 0, f"{r.paper_id}/{r.property}: pred_count={r.pred_count}"
            assert r.matched_count >= 0

    def test_sample_property_has_recall_only(self):
        records = parse_xinyan(DATA_DIR / "xinyan.xlsx")
        sample_records = [r for r in records if r.property == "sample"]
        for r in sample_records:
            # sample col has no precision pair, so pred_count should equal matched_count
            # (we only have recall data: matched/total)
            assert r.true_count >= r.matched_count
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `conda run -n paperindex python -m pytest paper/tests/test_parsers.py::TestXinyanParser -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'paper.scripts.parsers.xinyan'`

- [ ] **Step 3: Implement xinyan parser**

```python
# paper/scripts/parsers/xinyan.py
"""Parser for xinyan.xlsx evaluation spreadsheet."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from paper.scripts.parsers.base import PaperPropertyRecord
from paper.scripts.utils import parse_fraction

# Map: recall column index -> (standardized_name, has_precision_pair)
_COL_MAP = {
    1: ("sample", False),
    2: ("smiles", True),
    4: ("constituents", True),
    6: ("topology", True),
    8: ("mn", True),
    10: ("mw", True),
    12: ("pdi", True),
    14: ("tg", True),
    16: ("tm", True),
    18: ("density", True),
    20: ("youngs_modulus", True),
    22: ("source", True),
    24: ("synthesis", True),
    26: ("processing_history", True),
}


def parse_xinyan(filepath: Path) -> list[PaperPropertyRecord]:
    """Parse xinyan.xlsx into PaperPropertyRecord list.

    Reads the 2-row header format with alternating recall/precision columns.
    Skips total rows at the bottom.
    """
    df = pd.read_excel(filepath, header=None, sheet_name="Sheet1")

    # Data rows start at row 2 (after 2 header rows)
    # Skip total rows: look for "total" in col 0
    records: list[PaperPropertyRecord] = []

    for row_idx in range(2, len(df)):
        paper_id = df.iloc[row_idx, 0]
        if pd.isna(paper_id):
            continue
        paper_id = str(paper_id).strip()
        if paper_id.startswith("total"):
            continue

        # Notes from last column (28)
        raw_notes = df.iloc[row_idx, 28]
        notes = str(raw_notes).strip() if pd.notna(raw_notes) else None

        for recall_col, (prop_name, has_precision) in _COL_MAP.items():
            recall_val = df.iloc[row_idx, recall_col]
            recall_parsed = parse_fraction(recall_val)

            if has_precision:
                precision_col = recall_col + 1
                precision_val = df.iloc[row_idx, precision_col]
                precision_parsed = parse_fraction(precision_val)
            else:
                precision_parsed = None

            if recall_parsed is None and precision_parsed is None:
                continue

            # recall fraction: matched_recall / true_count
            # precision fraction: matched_precision / pred_count
            # matched_recall and matched_precision should be the same (both = correct predictions)
            if recall_parsed is not None:
                matched_count = recall_parsed[0]
                true_count = recall_parsed[1]
            else:
                matched_count = 0
                true_count = 0

            if precision_parsed is not None:
                pred_count = precision_parsed[1]
                # Cross-check: precision matched should equal recall matched
                # Use recall matched as canonical if both exist
                if recall_parsed is None:
                    matched_count = precision_parsed[0]
            else:
                # No precision data — for sample col and any missing precision
                # Best estimate: pred_count = matched_count (assume no hallucinations)
                pred_count = matched_count

            records.append(PaperPropertyRecord(
                annotator="xinyan",
                paper_id=paper_id,
                property=prop_name,
                true_count=true_count,
                pred_count=pred_count,
                matched_count=matched_count,
                notes=notes,
            ))

    return records
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `conda run -n paperindex python -m pytest paper/tests/test_parsers.py::TestXinyanParser -v`
Expected: All 5 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add paper/scripts/parsers/xinyan.py paper/tests/test_parsers.py
git commit -m "feat(eval): add xinyan parser with tests"
```

---

### Task 4: Yuqi Parser

**Files:**
- Create: `paper/scripts/parsers/yuqi.py`
- Modify: `paper/tests/test_parsers.py`

**Context — yuqi.xlsx structure:**
- 2-row header: row 0 = property names, row 1 = recall/precision
- Data rows 2-22, 5 papers (Ranks 1-5)
- Multi-row papers: first row has Rank/Paper/Year, continuations have NaN
- Col 0: Rank, Col 1: Paper title, Col 2: Year
- Col 3: Sample recall (paper-level, e.g. "72/72")
- Col 4: Sample name
- Col 5: Per-sample Recall total, Col 6: Paper-level Total recall
- Col 7: Per-sample Precision total, Col 8: Paper-level Total precision
- Cols 9-30: Property recall/precision pairs (same pattern as xinyan)
- Col 31: Missing information
- Cols 32-35: Supplementary instruction (overflow)
- Values: "2./3.", "5./5." (dot-fraction format)

**Property column mapping:**

```python
_COL_MAP = {
    9: "constituents",    # (recall at col, precision at col+1)
    11: "topology",
    13: "synthesis",
    15: "distribution",
    17: "processing_history",
    19: "properties",
    21: "mn",
    23: "mw",
    25: "pdi",
    27: "tg",
    29: "tm",
}
```

- [ ] **Step 1: Write tests for yuqi parser**

Add to `paper/tests/test_parsers.py`:

```python
from paper.scripts.parsers.yuqi import parse_yuqi


class TestYuqiParser:
    def test_returns_list_of_records(self):
        records = parse_yuqi(DATA_DIR / "yuqi.xlsx")
        assert len(records) > 0
        assert all(r.annotator == "yuqi" for r in records)

    def test_has_5_papers(self):
        records = parse_yuqi(DATA_DIR / "yuqi.xlsx")
        paper_ids = set(r.paper_id for r in records)
        assert len(paper_ids) == 5

    def test_has_expected_properties(self):
        records = parse_yuqi(DATA_DIR / "yuqi.xlsx")
        properties = set(r.property for r in records)
        assert "tg" in properties
        assert "mn" in properties
        assert "synthesis" in properties
        assert "constituents" in properties

    def test_counts_are_non_negative(self):
        records = parse_yuqi(DATA_DIR / "yuqi.xlsx")
        for r in records:
            assert r.true_count >= 0
            assert r.pred_count >= 0
            assert r.matched_count >= 0

    def test_sample_property_aggregated_per_paper(self):
        records = parse_yuqi(DATA_DIR / "yuqi.xlsx")
        sample_records = [r for r in records if r.property == "sample"]
        # Each paper should have exactly one sample record (aggregated)
        assert len(sample_records) == 5

    def test_notes_contain_missing_info(self):
        records = parse_yuqi(DATA_DIR / "yuqi.xlsx")
        records_with_notes = [r for r in records if r.notes]
        # At least some papers should have annotator notes
        assert len(records_with_notes) > 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `conda run -n paperindex python -m pytest paper/tests/test_parsers.py::TestYuqiParser -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement yuqi parser**

```python
# paper/scripts/parsers/yuqi.py
"""Parser for yuqi.xlsx evaluation spreadsheet."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from paper.scripts.parsers.base import PaperPropertyRecord
from paper.scripts.utils import parse_dot_fraction, parse_fraction

# Map: recall column index -> standardized property name
# Each property has recall at col N, precision at col N+1
_COL_MAP = {
    9: "constituents",
    11: "topology",
    13: "synthesis",
    15: "distribution",
    17: "processing_history",
    19: "properties",
    21: "mn",
    23: "mw",
    25: "pdi",
    27: "tg",
    29: "tm",
}


def _collect_notes(row: pd.Series) -> str | None:
    """Collect notes from Missing information (col 31) and Supplementary instruction (cols 32-35)."""
    parts = []
    for col in [31, 32, 33, 34, 35]:
        val = row.iloc[col] if col < len(row) else None
        if pd.notna(val):
            parts.append(str(val).strip())
    return " | ".join(parts) if parts else None


def parse_yuqi(filepath: Path) -> list[PaperPropertyRecord]:
    """Parse yuqi.xlsx into PaperPropertyRecord list.

    Groups multi-row papers by forward-filling the Rank column.
    Aggregates per-sample scores to per-paper totals.
    """
    df = pd.read_excel(filepath, header=None, sheet_name="Sheet1")

    # Data rows start at row 2
    # Forward-fill Rank (col 0) to group samples by paper
    data = df.iloc[2:].copy()
    data = data.reset_index(drop=True)

    # Drop rows where col 4 (sample name) is NaN — these are empty trailing rows
    data = data[data.iloc[:, 4].notna()]

    # Forward-fill Rank (col 0) and Paper (col 1) for multi-row papers
    data.iloc[:, 0] = data.iloc[:, 0].ffill()
    data.iloc[:, 1] = data.iloc[:, 1].ffill()

    records: list[PaperPropertyRecord] = []

    for rank, group in data.groupby(data.iloc[:, 0]):
        paper_title = str(group.iloc[0, 1]).strip().replace("\n", " ")[:80]
        paper_id = f"rank{int(rank)}"

        # Collect notes from all rows for this paper
        all_notes = []
        for _, row in group.iterrows():
            n = _collect_notes(row)
            if n:
                all_notes.append(n)
        paper_notes = " || ".join(all_notes) if all_notes else None

        # Sample-level record from col 3 (Sample recall: "72/72" or "4./4.")
        sample_recall_val = group.iloc[0, 3]
        sample_parsed = parse_dot_fraction(sample_recall_val)
        if sample_parsed is None:
            sample_parsed = parse_fraction(sample_recall_val)
        if sample_parsed is not None:
            records.append(PaperPropertyRecord(
                annotator="yuqi",
                paper_id=paper_id,
                property="sample",
                true_count=sample_parsed[1],
                pred_count=sample_parsed[1],  # sample recall only; assume pred=true
                matched_count=sample_parsed[0],
                notes=paper_notes,
            ))

        # Per-property: aggregate across all sample rows
        for recall_col, prop_name in _COL_MAP.items():
            precision_col = recall_col + 1
            total_matched_r = 0
            total_true = 0
            total_matched_p = 0
            total_pred = 0
            has_data = False

            for _, row in group.iterrows():
                r_parsed = parse_dot_fraction(row.iloc[recall_col])
                p_parsed = parse_dot_fraction(row.iloc[precision_col])

                if r_parsed is not None:
                    total_matched_r += r_parsed[0]
                    total_true += r_parsed[1]
                    has_data = True
                if p_parsed is not None:
                    total_matched_p += p_parsed[0]
                    total_pred += p_parsed[1]
                    has_data = True

            if not has_data:
                continue

            # Use recall matched as canonical matched count
            matched = total_matched_r if total_true > 0 else total_matched_p

            records.append(PaperPropertyRecord(
                annotator="yuqi",
                paper_id=paper_id,
                property=prop_name,
                true_count=total_true,
                pred_count=total_pred,
                matched_count=matched,
                notes=paper_notes,
            ))

    return records
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `conda run -n paperindex python -m pytest paper/tests/test_parsers.py::TestYuqiParser -v`
Expected: All 6 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add paper/scripts/parsers/yuqi.py paper/tests/test_parsers.py
git commit -m "feat(eval): add yuqi parser with tests"
```

---

### Task 5: Yuwei Parser

**Files:**
- Create: `paper/scripts/parsers/yuwei.py`
- Modify: `paper/tests/test_parsers.py`

**Context — yuwei.xlsx structure:**
- Sheet "Evaluation", 200 rows x 109 cols
- 2-row header: row 0 = group names, row 1 = column names
- Data starts row 2
- Col 0: Paper ID (only on first Sample row per paper, NaN on rest)
- Col 1: Sample name (NaN on Summary rows)
- Col 2: Row Type ("Sample" or "Summary")
- Col 3: Notes ("Gold", "Silver", "Bronze", "auto-calculated from sample rows above", etc.)
- Cols 4-6: Sample-level True/Pred/Matched (binary on Sample rows)
- Cols 7-12: Sample-level aggregate counts + metrics (Summary rows only)
- Cols 13-26: Composition block (14 cols)
- Cols 27-34: Tg block (8 cols: True/Pred/Matched/PageCorrect + P/R/F1/PageAcc)
- Cols 35-42: Tm block
- Cols 43-56: Mn block (14 cols: Excl True/Pred/Matched + Incl True/Pred/Matched + PageCorrect + Excl P/R/F1 + Incl P/R/F1 + PageAcc)
- Cols 57-70: Mw block (same as Mn)
- Cols 71-84: PDI block (same as Mn)
- Cols 85-92: Density block (8 cols)
- Cols 93-100: YM block (8 cols)
- Cols 101-108: Synthesis block (8 cols)
- Last row (199): OVERALL summary

**Property block definitions for Sample rows:**

```python
# Simple properties: (true_col, pred_col, matched_col, page_correct_col)
_SIMPLE_BLOCKS = {
    "sample": (4, 5, 6, None),
    "constituents": (13, 14, 15, None),
    "smiles": (None, None, 16, None),          # only "matched" column
    "mole_fraction": (17, 18, 19, None),
    "tg": (27, 28, 29, 30),
    "tm": (35, 36, 37, 38),
    "density": (85, 86, 87, 88),
    "youngs_modulus": (93, 94, 95, 96),
    "synthesis": (101, 102, 103, 104),
}

# Excl/Incl properties: (excl_true, excl_pred, excl_matched, incl_true, incl_pred, incl_matched, page_correct)
_EXCL_INCL_BLOCKS = {
    "mn": (43, 44, 45, 46, 47, 48, 49),
    "mw": (57, 58, 59, 60, 61, 62, 63),
    "pdi": (71, 72, 73, 74, 75, 76, 77),
}
```

- [ ] **Step 1: Write tests for yuwei parser**

Add to `paper/tests/test_parsers.py`:

```python
from paper.scripts.parsers.yuwei import parse_yuwei
from paper.scripts.parsers.base import YuweiSampleRecord


class TestYuweiParser:
    def test_returns_two_lists(self):
        layer1, layer2 = parse_yuwei(DATA_DIR / "yuwei.xlsx")
        assert isinstance(layer1, list)
        assert isinstance(layer2, list)
        assert len(layer1) > 0
        assert len(layer2) > 0

    def test_layer1_records_are_paper_property_records(self):
        layer1, _ = parse_yuwei(DATA_DIR / "yuwei.xlsx")
        assert all(r.annotator == "yuwei" for r in layer1)

    def test_layer1_excludes_overall(self):
        layer1, _ = parse_yuwei(DATA_DIR / "yuwei.xlsx")
        paper_ids = set(r.paper_id for r in layer1)
        assert "OVERALL" not in paper_ids

    def test_layer1_has_20_papers(self):
        layer1, _ = parse_yuwei(DATA_DIR / "yuwei.xlsx")
        paper_ids = set(r.paper_id for r in layer1)
        assert len(paper_ids) == 20

    def test_layer1_has_expected_properties(self):
        layer1, _ = parse_yuwei(DATA_DIR / "yuwei.xlsx")
        properties = set(r.property for r in layer1)
        assert "tg" in properties
        assert "mn" in properties
        assert "synthesis" in properties

    def test_layer2_records_are_yuwei_sample_records(self):
        _, layer2 = parse_yuwei(DATA_DIR / "yuwei.xlsx")
        assert all(isinstance(r, YuweiSampleRecord) for r in layer2)

    def test_layer2_has_quality_tiers(self):
        _, layer2 = parse_yuwei(DATA_DIR / "yuwei.xlsx")
        tiers = set(r.quality_tier for r in layer2 if r.quality_tier)
        assert "Gold" in tiers or "Silver" in tiers

    def test_layer2_mn_has_excl_and_incl_variants(self):
        _, layer2 = parse_yuwei(DATA_DIR / "yuwei.xlsx")
        mn_records = [r for r in layer2 if r.property == "mn"]
        variants = set(r.variant for r in mn_records)
        assert "excl" in variants
        assert "incl" in variants

    def test_layer2_tg_has_page_correct(self):
        _, layer2 = parse_yuwei(DATA_DIR / "yuwei.xlsx")
        tg_records = [r for r in layer2 if r.property == "tg"]
        has_page = [r for r in tg_records if r.page_correct is not None]
        assert len(has_page) > 0

    def test_layer2_values_are_binary(self):
        _, layer2 = parse_yuwei(DATA_DIR / "yuwei.xlsx")
        for r in layer2:
            assert r.true in (0, 1), f"true={r.true} for {r.paper_id}/{r.sample_name}/{r.property}"
            assert r.pred in (0, 1), f"pred={r.pred} for {r.paper_id}/{r.sample_name}/{r.property}"
            assert r.matched in (0, 1), f"matched={r.matched} for {r.paper_id}/{r.sample_name}/{r.property}"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `conda run -n paperindex python -m pytest paper/tests/test_parsers.py::TestYuweiParser -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement yuwei parser**

```python
# paper/scripts/parsers/yuwei.py
"""Parser for yuwei.xlsx evaluation spreadsheet."""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

from paper.scripts.parsers.base import PaperPropertyRecord, YuweiSampleRecord

# Simple properties: (true_col, pred_col, matched_col, page_correct_col)
# None means column doesn't exist for that field
_SIMPLE_BLOCKS = {
    "sample": (4, 5, 6, None),
    "constituents": (13, 14, 15, None),
    "smiles": (None, None, 16, None),
    "mole_fraction": (17, 18, 19, None),
    "tg": (27, 28, 29, 30),
    "tm": (35, 36, 37, 38),
    "density": (85, 86, 87, 88),
    "youngs_modulus": (93, 94, 95, 96),
    "synthesis": (101, 102, 103, 104),
}

# Excl/Incl properties: (excl_true, excl_pred, excl_matched, incl_true, incl_pred, incl_matched, page_correct)
_EXCL_INCL_BLOCKS = {
    "mn": (43, 44, 45, 46, 47, 48, 49),
    "mw": (57, 58, 59, 60, 61, 62, 63),
    "pdi": (71, 72, 73, 74, 75, 76, 77),
}


def _safe_int(val) -> int | None:
    """Convert a cell value to int, returning None if NaN."""
    if val is None:
        return None
    if isinstance(val, float):
        if math.isnan(val):
            return None
        return int(val)
    return int(val)


def _extract_quality_tier(notes_val) -> str | None:
    """Extract quality tier from Notes column value."""
    if pd.isna(notes_val):
        return None
    s = str(notes_val).strip()
    if s.startswith("Gold"):
        return "Gold"
    if s.startswith("Silver"):
        return "Silver"
    if s.startswith("Bronze"):
        return "Bronze"
    return None


def parse_yuwei(filepath: Path) -> tuple[list[PaperPropertyRecord], list[YuweiSampleRecord]]:
    """Parse yuwei.xlsx into Layer 1 and Layer 2 records.

    Returns:
        (layer1_records, layer2_records) where layer1 is per-paper summary
        and layer2 is per-sample detail.
    """
    df = pd.read_excel(filepath, header=None, sheet_name="Evaluation")

    # Data starts at row 2 (after 2 header rows)
    data = df.iloc[2:].copy().reset_index(drop=True)

    # Forward-fill Paper ID (col 0) for grouping
    data.iloc[:, 0] = data.iloc[:, 0].ffill()

    layer1: list[PaperPropertyRecord] = []
    layer2: list[YuweiSampleRecord] = []

    for paper_id_raw, group in data.groupby(data.iloc[:, 0]):
        paper_id = str(paper_id_raw).strip()
        if paper_id == "OVERALL":
            continue

        sample_rows = group[group.iloc[:, 2] == "Sample"]
        summary_rows = group[group.iloc[:, 2] == "Summary"]

        # --- Layer 1: Build from summing Sample rows ---
        # (More reliable than reading Summary rows which may have formula artifacts)
        for prop_name, (t_col, p_col, m_col, pg_col) in _SIMPLE_BLOCKS.items():
            true_sum = 0
            pred_sum = 0
            matched_sum = 0
            has_data = False

            for _, row in sample_rows.iterrows():
                t = _safe_int(row.iloc[t_col]) if t_col is not None else None
                p = _safe_int(row.iloc[p_col]) if p_col is not None else None
                m = _safe_int(row.iloc[m_col]) if m_col is not None else None

                if t is not None:
                    true_sum += t
                    has_data = True
                if p is not None:
                    pred_sum += p
                if m is not None:
                    matched_sum += m
                    has_data = True

                # For smiles (only matched col), set true=pred=matched
                if prop_name == "smiles" and m is not None:
                    true_sum = matched_sum  # will be overwritten each iteration, but final value is sum
                    pred_sum = matched_sum

            if not has_data:
                continue

            # Fix smiles: recompute sums properly
            if prop_name == "smiles":
                true_sum = matched_sum
                pred_sum = matched_sum

            layer1.append(PaperPropertyRecord(
                annotator="yuwei",
                paper_id=paper_id,
                property=prop_name,
                true_count=true_sum,
                pred_count=pred_sum,
                matched_count=matched_sum,
            ))

        # Excl/Incl properties — create separate records for excl and incl
        for prop_name, (et, ep, em, it, ip, im, pg) in _EXCL_INCL_BLOCKS.items():
            for variant, t_col, p_col, m_col in [("excl", et, ep, em), ("incl", it, ip, im)]:
                true_sum = 0
                pred_sum = 0
                matched_sum = 0
                has_data = False

                for _, row in sample_rows.iterrows():
                    t = _safe_int(row.iloc[t_col])
                    p = _safe_int(row.iloc[p_col])
                    m = _safe_int(row.iloc[m_col])
                    if t is not None:
                        true_sum += t
                        has_data = True
                    if p is not None:
                        pred_sum += p
                    if m is not None:
                        matched_sum += m

                if not has_data:
                    continue

                layer1.append(PaperPropertyRecord(
                    annotator="yuwei",
                    paper_id=paper_id,
                    property=f"{prop_name}_{variant}",
                    true_count=true_sum,
                    pred_count=pred_sum,
                    matched_count=matched_sum,
                ))

        # --- Layer 2: Per-sample detail ---
        for _, row in sample_rows.iterrows():
            sample_name = str(row.iloc[1]).strip() if pd.notna(row.iloc[1]) else "unknown"
            quality_tier = _extract_quality_tier(row.iloc[3])

            # Simple properties
            for prop_name, (t_col, p_col, m_col, pg_col) in _SIMPLE_BLOCKS.items():
                t = _safe_int(row.iloc[t_col]) if t_col is not None else None
                p = _safe_int(row.iloc[p_col]) if p_col is not None else None
                m = _safe_int(row.iloc[m_col]) if m_col is not None else None
                pg = _safe_int(row.iloc[pg_col]) if pg_col is not None else None

                # Skip if all fields are None
                if t is None and p is None and m is None:
                    continue

                # For smiles (matched-only), mirror to true/pred
                if prop_name == "smiles":
                    t = m if m is not None else 0
                    p = m if m is not None else 0

                layer2.append(YuweiSampleRecord(
                    paper_id=paper_id,
                    sample_name=sample_name,
                    quality_tier=quality_tier,
                    property=prop_name,
                    true=t if t is not None else 0,
                    pred=p if p is not None else 0,
                    matched=m if m is not None else 0,
                    page_correct=pg,
                    variant=None,
                ))

            # Excl/Incl properties
            for prop_name, (et, ep, em, it, ip, im, pg_col) in _EXCL_INCL_BLOCKS.items():
                pg = _safe_int(row.iloc[pg_col])

                for variant, t_col, p_col, m_col in [("excl", et, ep, em), ("incl", it, ip, im)]:
                    t = _safe_int(row.iloc[t_col])
                    p = _safe_int(row.iloc[p_col])
                    m = _safe_int(row.iloc[m_col])

                    if t is None and p is None and m is None:
                        continue

                    layer2.append(YuweiSampleRecord(
                        paper_id=paper_id,
                        sample_name=sample_name,
                        quality_tier=quality_tier,
                        property=prop_name,
                        true=t if t is not None else 0,
                        pred=p if p is not None else 0,
                        matched=m if m is not None else 0,
                        page_correct=pg,
                        variant=variant,
                    ))

    return layer1, layer2
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `conda run -n paperindex python -m pytest paper/tests/test_parsers.py::TestYuweiParser -v`
Expected: All 10 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add paper/scripts/parsers/yuwei.py paper/tests/test_parsers.py
git commit -m "feat(eval): add yuwei parser with tests"
```

---

### Task 6: Unifier Script

**Files:**
- Create: `paper/scripts/unify.py`
- Modify: `paper/scripts/parsers/__init__.py`
- Modify: `paper/tests/test_parsers.py`

- [ ] **Step 1: Update parsers `__init__.py` with re-exports**

```python
# paper/scripts/parsers/__init__.py
from paper.scripts.parsers.xinyan import parse_xinyan
from paper.scripts.parsers.yuqi import parse_yuqi
from paper.scripts.parsers.yuwei import parse_yuwei

__all__ = ["parse_xinyan", "parse_yuqi", "parse_yuwei"]
```

- [ ] **Step 2: Write tests for the unifier**

Add to `paper/tests/test_parsers.py`:

```python
import subprocess
import csv


class TestUnifier:
    def test_unify_produces_csvs(self, tmp_path):
        """Run unify.py and check that output CSVs are created."""
        result = subprocess.run(
            [
                "conda", "run", "-n", "paperindex", "python",
                str(Path(__file__).resolve().parent.parent / "scripts" / "unify.py"),
                "--output-dir", str(tmp_path),
            ],
            capture_output=True, text=True, cwd=str(Path(__file__).resolve().parent.parent.parent),
        )
        assert result.returncode == 0, f"unify.py failed:\n{result.stderr}"

        assert (tmp_path / "layer1_paper_summary.csv").exists()
        assert (tmp_path / "layer2_yuwei_samples.csv").exists()
        assert (tmp_path / "failure_analysis.csv").exists()

    def test_layer1_csv_has_expected_columns(self, tmp_path):
        subprocess.run(
            [
                "conda", "run", "-n", "paperindex", "python",
                str(Path(__file__).resolve().parent.parent / "scripts" / "unify.py"),
                "--output-dir", str(tmp_path),
            ],
            capture_output=True, text=True, cwd=str(Path(__file__).resolve().parent.parent.parent),
        )
        with open(tmp_path / "layer1_paper_summary.csv") as f:
            reader = csv.DictReader(f)
            row = next(reader)
            assert "annotator" in row
            assert "paper_id" in row
            assert "property" in row
            assert "precision" in row
            assert "recall" in row
            assert "f1" in row

    def test_layer1_has_all_three_annotators(self, tmp_path):
        subprocess.run(
            [
                "conda", "run", "-n", "paperindex", "python",
                str(Path(__file__).resolve().parent.parent / "scripts" / "unify.py"),
                "--output-dir", str(tmp_path),
            ],
            capture_output=True, text=True, cwd=str(Path(__file__).resolve().parent.parent.parent),
        )
        with open(tmp_path / "layer1_paper_summary.csv") as f:
            reader = csv.DictReader(f)
            annotators = set(row["annotator"] for row in reader)
        assert annotators == {"xinyan", "yuqi", "yuwei"}

    def test_failure_analysis_has_valid_types(self, tmp_path):
        subprocess.run(
            [
                "conda", "run", "-n", "paperindex", "python",
                str(Path(__file__).resolve().parent.parent / "scripts" / "unify.py"),
                "--output-dir", str(tmp_path),
            ],
            capture_output=True, text=True, cwd=str(Path(__file__).resolve().parent.parent.parent),
        )
        with open(tmp_path / "failure_analysis.csv") as f:
            reader = csv.DictReader(f)
            types = set(row["failure_type"] for row in reader)
        assert types.issubset({"missed", "wrong", "hallucinated"})
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `conda run -n paperindex python -m pytest paper/tests/test_parsers.py::TestUnifier -v`
Expected: FAIL — unify.py doesn't exist yet

- [ ] **Step 4: Implement unify.py**

```python
# paper/scripts/unify.py
"""Unifier script: runs all parsers, computes metrics, writes CSVs."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from pathlib import Path

from paper.scripts.parsers import parse_xinyan, parse_yuqi, parse_yuwei
from paper.scripts.parsers.base import PaperPropertyRecord, YuweiSampleRecord
from paper.scripts.utils import classify_failure, compute_metrics

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"


def build_layer1(records: list[PaperPropertyRecord]) -> list[dict]:
    """Add precision/recall/F1 to each record, return list of dicts."""
    rows = []
    for r in records:
        p, rec, f1 = compute_metrics(r.true_count, r.pred_count, r.matched_count)
        rows.append({
            "annotator": r.annotator,
            "paper_id": r.paper_id,
            "property": r.property,
            "true_count": r.true_count,
            "pred_count": r.pred_count,
            "matched_count": r.matched_count,
            "precision": p,
            "recall": rec,
            "f1": f1,
            "notes": r.notes or "",
        })
    return rows


def build_layer2(records: list[YuweiSampleRecord]) -> list[dict]:
    """Convert YuweiSampleRecord list to list of dicts."""
    return [asdict(r) for r in records]


def build_failure_analysis(
    all_records: list[PaperPropertyRecord],
    yuwei_samples: list[YuweiSampleRecord],
) -> list[dict]:
    """Build failure analysis from Layer 1 records and Layer 2 sample detail.

    For xinyan/yuqi: infer failure counts from aggregate true/pred/matched.
    For yuwei: use per-sample binary data for precise failure classification.
    """
    failures = []

    # Yuwei: per-sample failures from Layer 2
    for s in yuwei_samples:
        ftype = classify_failure(s.true, s.pred, s.matched)
        if ftype is not None:
            prop = s.property
            if s.variant:
                prop = f"{prop}_{s.variant}"
            failures.append({
                "annotator": "yuwei",
                "paper_id": s.paper_id,
                "property": prop,
                "failure_type": ftype,
                "annotator_note": "",
            })

    # Xinyan/Yuqi: aggregate-level failure counts
    for r in all_records:
        if r.annotator == "yuwei":
            continue

        # Missed: true - matched (true items that weren't correctly predicted)
        missed_count = r.true_count - r.matched_count
        for _ in range(max(0, missed_count)):
            failures.append({
                "annotator": r.annotator,
                "paper_id": r.paper_id,
                "property": r.property,
                "failure_type": "missed",
                "annotator_note": r.notes or "",
            })

        # Hallucinated: pred - matched (predictions with no true counterpart)
        # But only if pred > true (otherwise it's "wrong", not hallucinated)
        wrong_count = min(r.pred_count, r.true_count) - r.matched_count
        for _ in range(max(0, wrong_count)):
            failures.append({
                "annotator": r.annotator,
                "paper_id": r.paper_id,
                "property": r.property,
                "failure_type": "wrong",
                "annotator_note": r.notes or "",
            })

        hallucinated_count = max(0, r.pred_count - r.true_count)
        for _ in range(hallucinated_count):
            failures.append({
                "annotator": r.annotator,
                "paper_id": r.paper_id,
                "property": r.property,
                "failure_type": "hallucinated",
                "annotator_note": r.notes or "",
            })

    return failures


def write_csv(rows: list[dict], filepath: Path) -> None:
    """Write list of dicts to CSV."""
    if not rows:
        filepath.write_text("")
        return
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Unify evaluation data into CSVs")
    parser.add_argument(
        "--data-dir", type=Path, default=DATA_DIR,
        help="Directory containing xlsx files",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
        help="Directory to write output CSVs",
    )
    args = parser.parse_args()

    print(f"Reading data from {args.data_dir}")

    # Parse all three annotators
    xinyan_records = parse_xinyan(args.data_dir / "xinyan.xlsx")
    yuqi_records = parse_yuqi(args.data_dir / "yuqi.xlsx")
    yuwei_layer1_records, yuwei_layer2_records = parse_yuwei(args.data_dir / "yuwei.xlsx")

    print(f"  xinyan: {len(xinyan_records)} records")
    print(f"  yuqi: {len(yuqi_records)} records")
    print(f"  yuwei layer1: {len(yuwei_layer1_records)} records")
    print(f"  yuwei layer2: {len(yuwei_layer2_records)} records")

    # Build unified outputs
    all_layer1_records = xinyan_records + yuqi_records + yuwei_layer1_records
    layer1_rows = build_layer1(all_layer1_records)
    layer2_rows = build_layer2(yuwei_layer2_records)
    failure_rows = build_failure_analysis(
        xinyan_records + yuqi_records, yuwei_layer2_records
    )

    # Write CSVs
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(layer1_rows, args.output_dir / "layer1_paper_summary.csv")
    write_csv(layer2_rows, args.output_dir / "layer2_yuwei_samples.csv")
    write_csv(failure_rows, args.output_dir / "failure_analysis.csv")

    print(f"\nOutput written to {args.output_dir}")
    print(f"  layer1_paper_summary.csv: {len(layer1_rows)} rows")
    print(f"  layer2_yuwei_samples.csv: {len(layer2_rows)} rows")
    print(f"  failure_analysis.csv: {len(failure_rows)} rows")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `conda run -n paperindex python -m pytest paper/tests/test_parsers.py::TestUnifier -v`
Expected: All 4 tests PASS.

- [ ] **Step 6: Run the unifier end-to-end and inspect output**

Run: `conda run -n paperindex python paper/scripts/unify.py`
Check: Verify the output CSVs exist in `paper/output/` and spot-check a few rows.

- [ ] **Step 7: Commit**

```bash
git add paper/scripts/parsers/__init__.py paper/scripts/unify.py paper/tests/test_parsers.py
git commit -m "feat(eval): add unifier script — produces layer1, layer2, failure CSVs"
```

---

### Task 7: Analysis Notebook

**Files:**
- Create: `paper/notebooks/analysis.ipynb`

**Dependencies:** Requires `paper/output/layer1_paper_summary.csv`, `layer2_yuwei_samples.csv`, `failure_analysis.csv` to exist. Run `unify.py` first.

- [ ] **Step 1: Create the analysis notebook**

Create `paper/notebooks/analysis.ipynb` with the following cells:

**Cell 1 — Setup:**

```python
import pandas as pd
import numpy as np
from pathlib import Path

OUTPUT_DIR = Path("../output")
layer1 = pd.read_csv(OUTPUT_DIR / "layer1_paper_summary.csv")
layer2 = pd.read_csv(OUTPUT_DIR / "layer2_yuwei_samples.csv")
failures = pd.read_csv(OUTPUT_DIR / "failure_analysis.csv")

print(f"Layer 1: {len(layer1)} rows, {layer1['annotator'].nunique()} annotators, {layer1['paper_id'].nunique()} papers")
print(f"Layer 2: {len(layer2)} rows, {layer2['paper_id'].nunique()} papers")
print(f"Failures: {len(failures)} rows")
```

**Cell 2 — Section 1: Overall System Performance:**

```python
# Overall precision/recall/F1 weighted by counts
# Exclude sample-level and variant properties for clean aggregation
core_props = ["constituents", "topology", "mn", "mw", "pdi", "tg", "tm",
              "density", "youngs_modulus", "synthesis", "processing_history"]

# For yuwei, use excl variants for mn/mw/pdi
l1 = layer1.copy()
# Filter to core properties (including mn_excl etc. for yuwei)
l1_core = l1[
    l1["property"].isin(core_props) |
    l1["property"].isin([f"{p}_excl" for p in ["mn", "mw", "pdi"]])
].copy()

# For yuwei excl variants, rename to base property name for aggregation
l1_core["property"] = l1_core["property"].str.replace("_excl$", "", regex=True)
# Drop yuwei's base mn/mw/pdi (keep only excl)
mask = (l1_core["annotator"] == "yuwei") & l1_core["property"].isin(["mn", "mw", "pdi"])
# Actually, yuwei won't have bare mn/mw/pdi since parser creates mn_excl/mn_incl
# So this is fine as-is

overall = l1_core.groupby("annotator").agg(
    true_total=("true_count", "sum"),
    pred_total=("pred_count", "sum"),
    matched_total=("matched_count", "sum"),
).reset_index()
overall["precision"] = overall["matched_total"] / overall["pred_total"]
overall["recall"] = overall["matched_total"] / overall["true_total"]
overall["f1"] = 2 * overall["precision"] * overall["recall"] / (overall["precision"] + overall["recall"])

print("=== Overall System Performance (weighted by counts) ===")
print(overall.to_string(index=False, float_format="%.3f"))
print()

# Grand total
gt = overall[["true_total", "pred_total", "matched_total"]].sum()
p = gt["matched_total"] / gt["pred_total"]
r = gt["matched_total"] / gt["true_total"]
f = 2 * p * r / (p + r)
print(f"GRAND TOTAL: P={p:.3f}  R={r:.3f}  F1={f:.3f}")
```

**Cell 3 — Section 2: Per-Property Breakdown:**

```python
# Per-property P/R/F1 across all annotators
prop_agg = l1_core.groupby("property").agg(
    true_total=("true_count", "sum"),
    pred_total=("pred_count", "sum"),
    matched_total=("matched_count", "sum"),
).reset_index()
prop_agg["precision"] = prop_agg["matched_total"] / prop_agg["pred_total"]
prop_agg["recall"] = prop_agg["matched_total"] / prop_agg["true_total"]
prop_agg["f1"] = 2 * prop_agg["precision"] * prop_agg["recall"] / (prop_agg["precision"] + prop_agg["recall"])
prop_agg = prop_agg.sort_values("f1", ascending=True)

print("=== Per-Property Performance (sorted by F1, worst first) ===")
print(prop_agg.to_string(index=False, float_format="%.3f"))
```

**Cell 4 — Section 3: Per-Paper Breakdown:**

```python
# Per-paper P/R/F1
paper_agg = l1_core.groupby(["annotator", "paper_id"]).agg(
    true_total=("true_count", "sum"),
    pred_total=("pred_count", "sum"),
    matched_total=("matched_count", "sum"),
    n_properties=("property", "nunique"),
).reset_index()
paper_agg["precision"] = paper_agg["matched_total"] / paper_agg["pred_total"]
paper_agg["recall"] = paper_agg["matched_total"] / paper_agg["true_total"]
paper_agg["f1"] = 2 * paper_agg["precision"] * paper_agg["recall"] / (paper_agg["precision"] + paper_agg["recall"])
paper_agg = paper_agg.sort_values("f1", ascending=True)

print("=== Hardest Papers (bottom 10 by F1) ===")
print(paper_agg.head(10).to_string(index=False, float_format="%.3f"))
print()
print("=== Easiest Papers (top 10 by F1) ===")
print(paper_agg.tail(10).to_string(index=False, float_format="%.3f"))
```

**Cell 5 — Section 4: Failure Pattern Analysis:**

```python
# Failure type distribution
print("=== Failure Type Distribution ===")
print(failures["failure_type"].value_counts().to_string())
print()

# Per-property failure breakdown
prop_failures = failures.groupby(["property", "failure_type"]).size().unstack(fill_value=0)
prop_failures["total"] = prop_failures.sum(axis=1)
prop_failures = prop_failures.sort_values("total", ascending=False)
print("=== Failures by Property ===")
print(prop_failures.to_string())
print()

# Mine yuqi's missing information notes
yuqi_failures = failures[
    (failures["annotator"] == "yuqi") & (failures["annotator_note"].str.len() > 0)
]
if len(yuqi_failures) > 0:
    print("=== Yuqi Annotator Notes (sample) ===")
    for note in yuqi_failures["annotator_note"].unique()[:10]:
        print(f"  - {note[:120]}")
```

**Cell 6 — Section 5: Yuwei Deep Dive:**

```python
# Page-level accuracy per property
page_props = layer2[layer2["page_correct"].notna()].copy()
page_acc = page_props.groupby("property").agg(
    page_correct_sum=("page_correct", "sum"),
    page_total=("page_correct", "count"),
).reset_index()
page_acc["page_accuracy"] = page_acc["page_correct_sum"] / page_acc["page_total"]
page_acc = page_acc.sort_values("page_accuracy")
print("=== Page-Level Accuracy (Yuwei) ===")
print(page_acc.to_string(index=False, float_format="%.3f"))
print()

# Excl vs Incl comparison
for prop in ["mn", "mw", "pdi"]:
    excl = layer2[(layer2["property"] == prop) & (layer2["variant"] == "excl")]
    incl = layer2[(layer2["property"] == prop) & (layer2["variant"] == "incl")]
    if len(excl) > 0 and len(incl) > 0:
        e_f1 = 2 * excl["matched"].sum() / (excl["true"].sum() + excl["pred"].sum()) if (excl["true"].sum() + excl["pred"].sum()) > 0 else 0
        i_f1 = 2 * incl["matched"].sum() / (incl["true"].sum() + incl["pred"].sum()) if (incl["true"].sum() + incl["pred"].sum()) > 0 else 0
        print(f"{prop.upper()}: Excl F1={e_f1:.3f}, Incl F1={i_f1:.3f}, delta={i_f1-e_f1:+.3f}")
print()

# Quality tier comparison
tier_agg = layer2[layer2["quality_tier"].notna()].groupby("quality_tier").agg(
    true_sum=("true", "sum"),
    pred_sum=("pred", "sum"),
    matched_sum=("matched", "sum"),
).reset_index()
tier_agg["precision"] = tier_agg["matched_sum"] / tier_agg["pred_sum"]
tier_agg["recall"] = tier_agg["matched_sum"] / tier_agg["true_sum"]
tier_agg["f1"] = 2 * tier_agg["precision"] * tier_agg["recall"] / (tier_agg["precision"] + tier_agg["recall"])
print("=== Quality Tier Comparison (Yuwei) ===")
print(tier_agg.to_string(index=False, float_format="%.3f"))
```

- [ ] **Step 2: Run the notebook end-to-end**

Run: `conda run -n paperindex jupyter nbconvert --to notebook --execute paper/notebooks/analysis.ipynb --output analysis.ipynb`
Expected: Notebook executes without errors, output cells populated.

- [ ] **Step 3: Commit**

```bash
git add paper/notebooks/analysis.ipynb
git commit -m "feat(eval): add analysis notebook — metrics, breakdowns, failure patterns"
```

---

### Task 8: Figures Notebook

**Files:**
- Create: `paper/notebooks/figures.ipynb`

**Dependencies:** Same CSVs as Task 7.

- [ ] **Step 1: Create the figures notebook**

Create `paper/notebooks/figures.ipynb` with the following cells:

**Cell 1 — Setup:**

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path

matplotlib.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

OUTPUT_DIR = Path("../output")
FIG_DIR = OUTPUT_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

layer1 = pd.read_csv(OUTPUT_DIR / "layer1_paper_summary.csv")
layer2 = pd.read_csv(OUTPUT_DIR / "layer2_yuwei_samples.csv")
failures = pd.read_csv(OUTPUT_DIR / "failure_analysis.csv")

# Prepare core properties view (same logic as analysis notebook)
core_props = ["constituents", "topology", "mn", "mw", "pdi", "tg", "tm",
              "density", "youngs_modulus", "synthesis", "processing_history"]
l1 = layer1.copy()
l1_core = l1[
    l1["property"].isin(core_props) |
    l1["property"].isin([f"{p}_excl" for p in ["mn", "mw", "pdi"]])
].copy()
l1_core["property"] = l1_core["property"].str.replace("_excl$", "", regex=True)
```

**Cell 2 — Figure 1: Overall Performance Table (LaTeX):**

```python
overall = l1_core.groupby("annotator").agg(
    true_total=("true_count", "sum"),
    pred_total=("pred_count", "sum"),
    matched_total=("matched_count", "sum"),
).reset_index()
overall["Precision"] = overall["matched_total"] / overall["pred_total"]
overall["Recall"] = overall["matched_total"] / overall["true_total"]
overall["F1"] = 2 * overall["Precision"] * overall["Recall"] / (overall["Precision"] + overall["Recall"])

gt = overall[["true_total", "pred_total", "matched_total"]].sum()
p, r = gt["matched_total"] / gt["pred_total"], gt["matched_total"] / gt["true_total"]
total_row = pd.DataFrame([{
    "annotator": "Overall",
    "true_total": int(gt["true_total"]),
    "pred_total": int(gt["pred_total"]),
    "matched_total": int(gt["matched_total"]),
    "Precision": p, "Recall": r, "F1": 2*p*r/(p+r),
}])
table = pd.concat([overall, total_row], ignore_index=True)

print("% LaTeX table")
print(r"\begin{table}[h]")
print(r"\centering")
print(r"\caption{Overall extraction performance by annotator}")
print(r"\begin{tabular}{lccc}")
print(r"\toprule")
print(r"Annotator & Precision & Recall & F1 \\")
print(r"\midrule")
for _, row in table.iterrows():
    name = row["annotator"].capitalize()
    print(f"{name} & {row['Precision']:.3f} & {row['Recall']:.3f} & {row['F1']:.3f} \\\\")
print(r"\bottomrule")
print(r"\end{tabular}")
print(r"\end{table}")
```

**Cell 3 — Figure 2: Per-Property Bar Chart:**

```python
prop_agg = l1_core.groupby("property").agg(
    true_total=("true_count", "sum"),
    pred_total=("pred_count", "sum"),
    matched_total=("matched_count", "sum"),
).reset_index()
prop_agg["Precision"] = prop_agg["matched_total"] / prop_agg["pred_total"]
prop_agg["Recall"] = prop_agg["matched_total"] / prop_agg["true_total"]
prop_agg["F1"] = 2 * prop_agg["Precision"] * prop_agg["Recall"] / (prop_agg["Precision"] + prop_agg["Recall"])
prop_agg = prop_agg.sort_values("F1", ascending=False)

fig, ax = plt.subplots(figsize=(12, 5))
x = np.arange(len(prop_agg))
w = 0.25
ax.bar(x - w, prop_agg["Precision"], w, label="Precision", color="#2196F3")
ax.bar(x, prop_agg["Recall"], w, label="Recall", color="#FF9800")
ax.bar(x + w, prop_agg["F1"], w, label="F1", color="#4CAF50")
ax.set_xticks(x)
ax.set_xticklabels(prop_agg["property"], rotation=45, ha="right")
ax.set_ylabel("Score")
ax.set_title("Extraction Performance by Property")
ax.legend()
ax.set_ylim(0, 1.05)
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(FIG_DIR / "fig2_per_property_bar.png")
plt.show()
```

**Cell 4 — Figure 3: Property x Paper Heatmap:**

```python
# Pivot: paper_id x property -> F1
paper_prop = l1_core.groupby(["paper_id", "property"]).agg(
    true_total=("true_count", "sum"),
    pred_total=("pred_count", "sum"),
    matched_total=("matched_count", "sum"),
).reset_index()
paper_prop["f1"] = paper_prop.apply(
    lambda r: 2*r["matched_total"]/(r["true_total"]+r["pred_total"])
    if (r["true_total"]+r["pred_total"]) > 0 else np.nan, axis=1
)
heatmap_data = paper_prop.pivot(index="paper_id", columns="property", values="f1")

fig, ax = plt.subplots(figsize=(14, max(8, len(heatmap_data) * 0.35)))
im = ax.imshow(heatmap_data.values, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)
ax.set_xticks(range(len(heatmap_data.columns)))
ax.set_xticklabels(heatmap_data.columns, rotation=45, ha="right")
ax.set_yticks(range(len(heatmap_data.index)))
ax.set_yticklabels(heatmap_data.index, fontsize=8)
ax.set_title("F1 Score: Paper x Property")
plt.colorbar(im, ax=ax, shrink=0.8, label="F1")
fig.tight_layout()
fig.savefig(FIG_DIR / "fig3_heatmap.png")
plt.show()
```

**Cell 5 — Figure 4: Radar/Spider Plot:**

```python
props = prop_agg["property"].tolist()
values = prop_agg["F1"].tolist()
N = len(props)
angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
values_plot = values + [values[0]]
angles_plot = angles + [angles[0]]

fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
ax.plot(angles_plot, values_plot, "o-", linewidth=2, color="#4CAF50")
ax.fill(angles_plot, values_plot, alpha=0.15, color="#4CAF50")
ax.set_xticks(angles)
ax.set_xticklabels(props, size=9)
ax.set_ylim(0, 1)
ax.set_title("Per-Property F1 Profile", pad=20)
fig.tight_layout()
fig.savefig(FIG_DIR / "fig4_radar.png")
plt.show()
```

**Cell 6 — Figure 5: Failure Type Distribution:**

```python
prop_failures = failures.groupby(["property", "failure_type"]).size().unstack(fill_value=0)
# Sort by total failures
prop_failures["total"] = prop_failures.sum(axis=1)
prop_failures = prop_failures.sort_values("total", ascending=True).drop(columns="total")

fig, ax = plt.subplots(figsize=(10, 6))
prop_failures.plot(kind="barh", stacked=True, ax=ax,
                   color={"missed": "#F44336", "wrong": "#FF9800", "hallucinated": "#9C27B0"})
ax.set_xlabel("Number of Failures")
ax.set_title("Failure Distribution by Property")
ax.legend(title="Failure Type")
ax.grid(axis="x", alpha=0.3)
fig.tight_layout()
fig.savefig(FIG_DIR / "fig5_failure_distribution.png")
plt.show()
```

**Cell 7 — Figure 6: Page Accuracy Bar Chart:**

```python
page_props = layer2[layer2["page_correct"].notna()].copy()
page_acc = page_props.groupby("property").agg(
    correct=("page_correct", "sum"),
    total=("page_correct", "count"),
).reset_index()
page_acc["accuracy"] = page_acc["correct"] / page_acc["total"]
page_acc = page_acc.sort_values("accuracy", ascending=False)

fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.bar(range(len(page_acc)), page_acc["accuracy"], color="#2196F3")
ax.set_xticks(range(len(page_acc)))
ax.set_xticklabels(page_acc["property"], rotation=45, ha="right")
ax.set_ylabel("Page Accuracy")
ax.set_title("Page-Level Source Accuracy (Yuwei)")
ax.set_ylim(0, 1.05)
ax.grid(axis="y", alpha=0.3)
for bar, acc in zip(bars, page_acc["accuracy"]):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
            f"{acc:.2f}", ha="center", va="bottom", fontsize=9)
fig.tight_layout()
fig.savefig(FIG_DIR / "fig6_page_accuracy.png")
plt.show()
```

**Cell 8 — Figure 7: Excl vs Incl Comparison:**

```python
comparison = []
for prop in ["mn", "mw", "pdi"]:
    for variant in ["excl", "incl"]:
        subset = layer2[(layer2["property"] == prop) & (layer2["variant"] == variant)]
        if len(subset) > 0:
            t, p, m = subset["true"].sum(), subset["pred"].sum(), subset["matched"].sum()
            prec = m / p if p > 0 else 0
            rec = m / t if t > 0 else 0
            f1 = 2*prec*rec/(prec+rec) if (prec+rec) > 0 else 0
            comparison.append({"property": prop.upper(), "variant": variant, "F1": f1})

comp_df = pd.DataFrame(comparison)
comp_pivot = comp_df.pivot(index="property", columns="variant", values="F1")

fig, ax = plt.subplots(figsize=(8, 5))
x = np.arange(len(comp_pivot))
w = 0.3
ax.bar(x - w/2, comp_pivot["excl"], w, label="Exclude derived", color="#2196F3")
ax.bar(x + w/2, comp_pivot["incl"], w, label="Include derived", color="#FF9800")
ax.set_xticks(x)
ax.set_xticklabels(comp_pivot.index)
ax.set_ylabel("F1 Score")
ax.set_title("Molecular Weight: Excl vs Incl Derived Values")
ax.legend()
ax.set_ylim(0, 1.05)
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(FIG_DIR / "fig7_excl_vs_incl.png")
plt.show()
```

**Cell 9 — Figure 8: Quality Tier Comparison:**

```python
tier_data = []
for tier in ["Gold", "Silver", "Bronze"]:
    subset = layer2[layer2["quality_tier"] == tier]
    if len(subset) > 0:
        t, p, m = subset["true"].sum(), subset["pred"].sum(), subset["matched"].sum()
        prec = m / p if p > 0 else 0
        rec = m / t if t > 0 else 0
        f1 = 2*prec*rec/(prec+rec) if (prec+rec) > 0 else 0
        tier_data.append({"tier": tier, "Precision": prec, "Recall": rec, "F1": f1})

tier_df = pd.DataFrame(tier_data)

fig, ax = plt.subplots(figsize=(8, 5))
x = np.arange(len(tier_df))
w = 0.25
ax.bar(x - w, tier_df["Precision"], w, label="Precision", color="#2196F3")
ax.bar(x, tier_df["Recall"], w, label="Recall", color="#FF9800")
ax.bar(x + w, tier_df["F1"], w, label="F1", color="#4CAF50")
ax.set_xticks(x)
ax.set_xticklabels(tier_df["tier"])
ax.set_ylabel("Score")
ax.set_title("Performance by Quality Tier (Yuwei)")
ax.legend()
ax.set_ylim(0, 1.05)
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(FIG_DIR / "fig8_quality_tier.png")
plt.show()
```

- [ ] **Step 2: Run the notebook end-to-end**

Run: `conda run -n paperindex jupyter nbconvert --to notebook --execute paper/notebooks/figures.ipynb --output figures.ipynb`
Expected: Notebook executes without errors. PNGs generated in `paper/output/figures/`.

- [ ] **Step 3: Verify figures exist**

Run: `ls -la paper/output/figures/`
Expected: 7 PNG files (fig2 through fig8).

- [ ] **Step 4: Commit**

```bash
git add paper/notebooks/figures.ipynb
git commit -m "feat(eval): add figures notebook — 7 publication-ready visualizations"
```

---

### Task 9: Gitignore and Final Verification

**Files:**
- Create or modify: `paper/.gitignore`

- [ ] **Step 1: Add gitignore for output directory**

```
# paper/.gitignore
output/
```

- [ ] **Step 2: Run the full pipeline end-to-end**

```bash
conda run -n paperindex python paper/scripts/unify.py
conda run -n paperindex jupyter nbconvert --to notebook --execute paper/notebooks/analysis.ipynb --output analysis.ipynb
conda run -n paperindex jupyter nbconvert --to notebook --execute paper/notebooks/figures.ipynb --output figures.ipynb
```

Expected: All three commands succeed. Output CSVs and PNGs are generated.

- [ ] **Step 3: Run all tests**

Run: `conda run -n paperindex python -m pytest paper/tests/test_parsers.py -v`
Expected: All tests PASS.

- [ ] **Step 4: Commit**

```bash
git add paper/.gitignore
git commit -m "chore(eval): add gitignore for generated output"
```

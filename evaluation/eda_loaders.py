"""
Load tabular views of evaluation / training data for EDA scripts.

Keeps plotting out of import paths used by tests (no matplotlib required).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def load_golden_eval_jsonl(path: Path) -> pd.DataFrame:
    """
    Load ``data/golden/eval_set.jsonl`` into a single DataFrame (mixed tasks).

    Rows include ``task`` plus task-specific columns (NaN where absent).
    """
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return pd.DataFrame(rows)


def load_labeled_tickets_csv(path: Path) -> pd.DataFrame:
    """Load CSV with at least ``text`` and ``category`` (for supervised training EDA)."""
    df = pd.read_csv(path)
    if "text" not in df.columns or "category" not in df.columns:
        raise ValueError("CSV must contain columns: text, category")
    out = df[["text", "category"]].copy()
    out["text"] = out["text"].astype(str)
    out["category"] = out["category"].astype(str)
    out["char_len"] = out["text"].str.len()
    out["word_len"] = out["text"].str.split().str.len()
    return out

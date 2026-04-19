"""Tests for evaluation.eda_loaders (no matplotlib)."""

from __future__ import annotations

from pathlib import Path

import pytest

from evaluation.eda_loaders import load_golden_eval_jsonl, load_labeled_tickets_csv


def test_load_golden_eval_jsonl_skips_blank_lines(tmp_path: Path) -> None:
    p = tmp_path / "e.jsonl"
    p.write_text(
        "\n"
        '{"task": "triage", "id": "x", "ticket_text": "a", '
        '"gold_priority": "low", "gold_category": "billing"}\n\n',
        encoding="utf-8",
    )
    df = load_golden_eval_jsonl(p)
    assert len(df) == 1


def test_load_golden_eval_jsonl_shapes(tmp_path: Path) -> None:
    p = tmp_path / "eval.jsonl"
    p.write_text(
        '{"task": "triage", "id": "a", "ticket_text": "hello", '
        '"gold_priority": "low", "gold_category": "billing"}\n',
        encoding="utf-8",
    )
    df = load_golden_eval_jsonl(p)
    assert len(df) == 1
    assert df.iloc[0]["task"] == "triage"
    assert df.iloc[0]["gold_category"] == "billing"


def test_load_labeled_tickets_csv_ok(tmp_path: Path) -> None:
    p = tmp_path / "t.csv"
    p.write_text("text,category\nfoo bar,billing\n", encoding="utf-8")
    df = load_labeled_tickets_csv(p)
    assert list(df.columns) == ["text", "category", "char_len", "word_len"]
    assert df.iloc[0]["category"] == "billing"
    assert df.iloc[0]["char_len"] == 7


def test_load_labeled_tickets_csv_missing_columns(tmp_path: Path) -> None:
    p = tmp_path / "bad.csv"
    p.write_text("a,b\n1,2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="text, category"):
        load_labeled_tickets_csv(p)

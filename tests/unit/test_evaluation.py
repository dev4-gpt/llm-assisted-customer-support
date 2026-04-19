"""Unit tests for evaluation package (no LLM / API)."""

from __future__ import annotations

import pytest

from evaluation.correlation import judge_proxy_correlation, pearson_r, spearman_rho
from evaluation.metrics import classification_report_dict, rouge_l_f1_placeholder
from evaluation.splits import stratified_split_indices


def test_stratified_split_covers_all_indices() -> None:
    labels = ["a", "a", "a", "b", "b", "c"]
    train, val, test = stratified_split_indices(labels, seed=0)
    all_idx = set(train) | set(val) | set(test)
    assert all_idx == set(range(len(labels)))


def test_classification_report_basic() -> None:
    y_true = ["x", "x", "y"]
    y_pred = ["x", "y", "y"]
    rep = classification_report_dict(y_true, y_pred)
    assert rep["support"] == 3
    assert "per_label" in rep
    assert rep["accuracy"] == pytest.approx(2 / 3)


def test_classification_report_length_mismatch() -> None:
    with pytest.raises(ValueError):
        classification_report_dict(["a"], ["a", "b"])


def test_rouge_placeholder_bounded() -> None:
    s = rouge_l_f1_placeholder("hello world", "hello there world")
    assert 0.0 <= s <= 1.0


def test_pearson_perfect_positive() -> None:
    assert pearson_r([1.0, 2.0, 3.0], [2.0, 4.0, 6.0]) == pytest.approx(1.0)


def test_spearman_inverse() -> None:
    rho = spearman_rho([3.0, 2.0, 1.0], [1.0, 2.0, 3.0])
    assert rho == pytest.approx(-1.0)


def test_judge_proxy_correlation_keys() -> None:
    out = judge_proxy_correlation([0.2, 0.8, 0.5], [1.0, 5.0, 3.0])
    assert "pearson_r" in out and "spearman_rho" in out
    assert out["n"] == 3.0

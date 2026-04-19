"""
Classification and summarization metrics for offline evaluation.

ROUGE integration is optional: install `rouge-score` if you want real scores.
This module provides a deterministic placeholder so the pipeline is documented
even when optional deps are missing.
"""

from __future__ import annotations

from typing import Any


def classification_report_dict(
    y_true: list[str],
    y_pred: list[str],
) -> dict[str, Any]:
    """
    Micro-averaged precision / recall / F1 over the label set present in y_true ∪ y_pred.

    For multi-label tasks, flatten to per-label binary vectors or use sklearn separately.
    """
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")

    labels = sorted(set(y_true) | set(y_pred))
    per_label: dict[str, dict[str, float]] = {}

    for lab in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred, strict=True) if t == lab and p == lab)
        fp = sum(1 for t, p in zip(y_true, y_pred, strict=True) if t != lab and p == lab)
        fn = sum(1 for t, p in zip(y_true, y_pred, strict=True) if t == lab and p != lab)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
        per_label[lab] = {"precision": prec, "recall": rec, "f1": f1}

    correct = sum(1 for t, p in zip(y_true, y_pred, strict=True) if t == p)
    accuracy = correct / len(y_true) if y_true else 0.0

    return {
        "accuracy": accuracy,
        "per_label": per_label,
        "support": len(y_true),
    }


def rouge_l_f1_placeholder(reference: str, candidate: str) -> float:
    """
    Cheap token-overlap proxy standing in for ROUGE-L F1 when `rouge-score` is not installed.

    Replace with `rouge_score` in experiments:
        pip install rouge-score
    """
    ref_toks = reference.lower().split()
    cand_toks = candidate.lower().split()
    if not ref_toks or not cand_toks:
        return 0.0
    ref_set = set(ref_toks)
    cand_set = set(cand_toks)
    overlap = len(ref_set & cand_set)
    prec = overlap / len(cand_set)
    rec = overlap / len(ref_set)
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)


def rouge_l_f1(reference: str, candidate: str) -> float:
    """
    ROUGE-L F1 when `rouge-score` is available; otherwise falls back to placeholder.
    """
    try:
        from rouge_score import rouge_scorer

        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        scores = scorer.score(reference, candidate)
        return float(scores["rougeL"].fmeasure)
    except ImportError:
        return rouge_l_f1_placeholder(reference, candidate)

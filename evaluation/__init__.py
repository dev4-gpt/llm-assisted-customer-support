"""
Offline evaluation utilities for triage, summarization, and LLM-judge calibration.

These modules are intentionally lightweight so they run without the API server.
Install optional deps for sklearn-heavy scripts: pip install -e ".[eval]"
"""

from __future__ import annotations

from evaluation.correlation import judge_proxy_correlation
from evaluation.metrics import classification_report_dict, rouge_l_f1, rouge_l_f1_placeholder
from evaluation.splits import stratified_split_indices

__all__ = [
    "classification_report_dict",
    "judge_proxy_correlation",
    "rouge_l_f1",
    "rouge_l_f1_placeholder",
    "stratified_split_indices",
]

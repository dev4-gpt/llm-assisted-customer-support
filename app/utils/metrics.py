"""
app/utils/metrics.py
─────────────────────
Prometheus metrics definitions.

All counters/histograms are defined here so they're registered once and
importable anywhere without circular-dependency issues.
"""

from __future__ import annotations

from prometheus_client import Counter, Histogram

# ── LLM ──────────────────────────────────────────────────────────────────────
LLM_REQUESTS = Counter(
    "support_triage_llm_requests_total",
    "Total LLM API requests",
    ["schema", "prompt_version"],
)

LLM_ERRORS = Counter(
    "support_triage_llm_errors_total",
    "Total LLM API errors",
    ["schema", "error_type", "prompt_version"],
)

LLM_LATENCY_SECONDS = Histogram(
    "support_triage_llm_latency_seconds",
    "LLM request latency in seconds",
    ["schema", "prompt_version"],
    buckets=[0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0],
)

# ── Endpoints ─────────────────────────────────────────────────────────────────
HTTP_REQUESTS = Counter(
    "support_triage_http_requests_total",
    "Total HTTP requests",
    ["endpoint", "method", "status"],
)

HTTP_LATENCY_SECONDS = Histogram(
    "support_triage_http_latency_seconds",
    "HTTP request latency",
    ["endpoint"],
    buckets=[0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
)

# ── Business ─────────────────────────────────────────────────────────────────
TRIAGE_PRIORITY = Counter(
    "support_triage_priority_total",
    "Tickets triaged by priority",
    ["priority"],
)

TRIAGE_CATEGORY = Counter(
    "support_triage_category_total",
    "Tickets triaged by category",
    ["category"],
)

QUALITY_SCORE = Histogram(
    "support_triage_quality_score",
    "Distribution of quality scores",
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)

QUALITY_PASS = Counter(
    "support_triage_quality_pass_total",
    "Quality evaluation outcomes",
    ["result"],  # "pass" | "fail"
)

PIPELINE_WORKFLOW = Counter(
    "support_triage_pipeline_workflow_total",
    "Pipeline workflow outcomes",
    ["result"],  # "pass" | "fail"
)

"""
app/api/v1/routers.py
──────────────────────
All HTTP endpoints for v1 of the API.

Design patterns:
  • Response models are always declared for schema generation in /docs.
  • Cache is checked before hitting the LLM to avoid redundant API costs.
  • Metrics are emitted after every successful response.
  • Endpoints return ORJSONResponse for faster serialisation.
"""

from __future__ import annotations

import time

from fastapi import APIRouter
from fastapi.responses import ORJSONResponse

from app.core.dependencies import (
    CacheDep,
    PipelineDep,
    QualityDep,
    RAGDep,
    SettingsDep,
    SummarizeDep,
    TriageDep,
)
from app.core.logging import get_logger
from app.models.domain import (
    HealthResponse,
    HealthStatus,
    PipelineRequest,
    PipelineResult,
    QualityRequest,
    QualityResult,
    RAGContextRequest,
    RAGContextResponse,
    SummarizeRequest,
    SummarizeResult,
    TriageRequest,
    TriageResult,
)
from app.utils.metrics import (
    HTTP_LATENCY_SECONDS,
    HTTP_REQUESTS,
    PIPELINE_WORKFLOW,
    QUALITY_PASS,
    QUALITY_SCORE,
    TRIAGE_CATEGORY,
    TRIAGE_PRIORITY,
)

logger = get_logger(__name__)

router = APIRouter()

APP_VERSION = "1.0.0"


# ── Health ────────────────────────────────────────────────────────────────────


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    tags=["ops"],
)
def health(cache: CacheDep, settings: SettingsDep) -> ORJSONResponse:
    checks = {
        "redis": "ok" if cache.ping() else "degraded",
        "config": "ok",
    }
    all_ok = all(v == "ok" for v in checks.values())
    status = HealthStatus.HEALTHY if all_ok else HealthStatus.DEGRADED

    return ORJSONResponse(content={"status": status, "version": APP_VERSION, "checks": checks})


# ── Triage ────────────────────────────────────────────────────────────────────


@router.post(
    "/triage",
    response_model=TriageResult,
    summary="Triage a support ticket",
    tags=["triage"],
)
def triage(
    request: TriageRequest,
    service: TriageDep,
    cache: CacheDep,
) -> ORJSONResponse:
    start = time.perf_counter()
    payload = request.model_dump()

    cached = cache.get("triage", payload)
    if cached:
        return ORJSONResponse(content=cached)

    result = service.triage(request)
    result_dict = result.model_dump()

    cache.set("triage", payload, result_dict)

    # Metrics
    TRIAGE_PRIORITY.labels(priority=result.priority.value).inc()
    TRIAGE_CATEGORY.labels(category=result.category.value).inc()
    duration = time.perf_counter() - start
    HTTP_LATENCY_SECONDS.labels(endpoint="/triage").observe(duration)
    HTTP_REQUESTS.labels(endpoint="/triage", method="POST", status=200).inc()

    return ORJSONResponse(content=result_dict)


# ── Quality ───────────────────────────────────────────────────────────────────


@router.post(
    "/quality",
    response_model=QualityResult,
    summary="Evaluate agent response quality",
    tags=["quality"],
)
def quality(
    request: QualityRequest,
    service: QualityDep,
    cache: CacheDep,
) -> ORJSONResponse:
    start = time.perf_counter()
    payload = request.model_dump()

    cached = cache.get("quality", payload)
    if cached:
        return ORJSONResponse(content=cached)

    result = service.evaluate(request)
    result_dict = result.model_dump()

    cache.set("quality", payload, result_dict)

    # Metrics
    QUALITY_SCORE.observe(result.score)
    QUALITY_PASS.labels(result="pass" if result.passed else "fail").inc()
    duration = time.perf_counter() - start
    HTTP_LATENCY_SECONDS.labels(endpoint="/quality").observe(duration)
    HTTP_REQUESTS.labels(endpoint="/quality", method="POST", status=200).inc()

    return ORJSONResponse(content=result_dict)


# ── Pipeline ──────────────────────────────────────────────────────────────────


@router.post(
    "/pipeline",
    response_model=PipelineResult,
    summary="Run full triage + quality pipeline",
    tags=["pipeline"],
)
def pipeline(
    request: PipelineRequest,
    service: PipelineDep,
    cache: CacheDep,
) -> ORJSONResponse:
    start = time.perf_counter()
    payload = request.model_dump()

    cached = cache.get("pipeline", payload)
    if cached:
        return ORJSONResponse(content=cached)

    result = service.run(request)
    result_dict = result.model_dump()

    cache.set("pipeline", payload, result_dict)

    # Metrics
    TRIAGE_PRIORITY.labels(priority=result.triage.priority.value).inc()
    TRIAGE_CATEGORY.labels(category=result.triage.category.value).inc()
    QUALITY_SCORE.observe(result.quality.score)
    QUALITY_PASS.labels(result="pass" if result.quality.passed else "fail").inc()
    PIPELINE_WORKFLOW.labels(result="pass" if result.workflow_passed else "fail").inc()
    duration = time.perf_counter() - start
    HTTP_LATENCY_SECONDS.labels(endpoint="/pipeline").observe(duration)
    HTTP_REQUESTS.labels(endpoint="/pipeline", method="POST", status=200).inc()

    return ORJSONResponse(content=result_dict)


# ── Summarization ───────────────────────────────────────────────────────────────


@router.post(
    "/summarize",
    response_model=SummarizeResult,
    summary="Summarize a multi-turn support thread",
    tags=["summarization"],
)
def summarize_thread(
    request: SummarizeRequest,
    service: SummarizeDep,
    cache: CacheDep,
) -> ORJSONResponse:
    start = time.perf_counter()
    payload = request.model_dump()

    cached = cache.get("summarize", payload)
    if cached:
        return ORJSONResponse(content=cached)

    result = service.summarize(request)
    result_dict = result.model_dump()
    cache.set("summarize", payload, result_dict)

    duration = time.perf_counter() - start
    HTTP_LATENCY_SECONDS.labels(endpoint="/summarize").observe(duration)
    HTTP_REQUESTS.labels(endpoint="/summarize", method="POST", status=200).inc()

    return ORJSONResponse(content=result_dict)


# ── RAG (policy snippets) ─────────────────────────────────────────────────────


@router.post(
    "/rag/context",
    response_model=RAGContextResponse,
    summary="Retrieve top policy snippets (lexical or embedding RAG)",
    tags=["rag"],
)
def rag_context(
    request: RAGContextRequest,
    service: RAGDep,
) -> ORJSONResponse:
    start = time.perf_counter()
    result = service.retrieve(request)
    result_dict = result.model_dump()

    duration = time.perf_counter() - start
    HTTP_LATENCY_SECONDS.labels(endpoint="/rag/context").observe(duration)
    HTTP_REQUESTS.labels(endpoint="/rag/context", method="POST", status=200).inc()

    return ORJSONResponse(content=result_dict)

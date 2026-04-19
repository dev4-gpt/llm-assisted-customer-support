"""API key middleware for mutating /api/v1 routes."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from app.core.dependencies import (
    get_cache,
    get_pipeline_service,
    get_quality_service,
    get_summarization_service,
    get_triage_service,
)


def _clear_app_caches() -> None:
    from app.core.config import get_settings
    from app.core.dependencies import (
        get_cache as gc,
    )
    from app.core.dependencies import (
        get_llm_client,
        get_rag_service,
    )
    from app.core.dependencies import (
        get_pipeline_service as gps,
    )
    from app.core.dependencies import (
        get_quality_service as gqs,
    )
    from app.core.dependencies import (
        get_summarization_service as gss,
    )
    from app.core.dependencies import (
        get_triage_service as gts,
    )

    for fn in (
        get_settings,
        get_llm_client,
        gc,
        get_rag_service,
        gts,
        gqs,
        gps,
        gss,
    ):
        fn.cache_clear()


@pytest.fixture()
def mock_triage_result():
    from app.models.domain import Category, IntentScore, Priority, RoutedTeam, TriageResult

    return TriageResult(
        priority=Priority.HIGH,
        category=Category.BILLING,
        intents=[IntentScore(label="billing", score=0.9)],
        sentiment_score=-0.2,
        routed_team=RoutedTeam.BILLING_SPECIALISTS,
        rationale="r",
        confidence=0.9,
    )


@pytest.fixture()
def mock_quality_result():
    from app.models.domain import QualityChecks, QualityResult

    return QualityResult(
        score=0.8,
        passed=True,
        checks=QualityChecks(
            empathetic_tone=True,
            actionable_next_step=True,
            policy_safety=True,
            resolved_or_escalated=True,
        ),
        coaching_feedback="ok",
        flagged_phrases=[],
    )


@pytest.fixture()
def mock_pipeline_result(mock_triage_result, mock_quality_result):
    from app.models.domain import PipelineResult

    return PipelineResult(
        triage=mock_triage_result,
        quality=mock_quality_result,
        recommended_sla_minutes=60,
        workflow_passed=True,
    )


@pytest.fixture()
def locked_client(monkeypatch, mock_triage_result, mock_quality_result, mock_pipeline_result):
    monkeypatch.setenv("API_KEYS", "unit-test-secret-key")
    _clear_app_caches()

    from app.main import create_app
    from app.models.domain import SummarizeResult

    app = create_app()

    mock_triage_svc = MagicMock()
    mock_triage_svc.triage.return_value = mock_triage_result
    mock_quality_svc = MagicMock()
    mock_quality_svc.evaluate.return_value = mock_quality_result
    mock_pipeline_svc = MagicMock()
    mock_pipeline_svc.run.return_value = mock_pipeline_result
    mock_summarize_svc = MagicMock()
    mock_summarize_svc.summarize.return_value = SummarizeResult(
        summary="s", key_points=["a"], confidence=0.5
    )
    mock_cache = MagicMock()
    mock_cache.get.return_value = None
    mock_cache.ping.return_value = True

    app.dependency_overrides[get_triage_service] = lambda: mock_triage_svc
    app.dependency_overrides[get_quality_service] = lambda: mock_quality_svc
    app.dependency_overrides[get_pipeline_service] = lambda: mock_pipeline_svc
    app.dependency_overrides[get_summarization_service] = lambda: mock_summarize_svc
    app.dependency_overrides[get_cache] = lambda: mock_cache

    with TestClient(app) as c:
        yield c

    app.dependency_overrides.clear()
    _clear_app_caches()


def test_post_rejected_without_api_key(locked_client: TestClient) -> None:
    r = locked_client.post(
        "/api/v1/triage",
        json={"ticket_text": "This is long enough for validation to pass here."},
    )
    assert r.status_code == 401


def test_post_allowed_with_api_key(locked_client: TestClient) -> None:
    r = locked_client.post(
        "/api/v1/triage",
        json={"ticket_text": "This is long enough for validation to pass here."},
        headers={"X-API-Key": "unit-test-secret-key"},
    )
    assert r.status_code == 200


def test_audit_log_written(
    monkeypatch, tmp_path, mock_triage_result, mock_quality_result, mock_pipeline_result
):
    logf = tmp_path / "audit.jsonl"
    monkeypatch.setenv("API_KEYS", "unit-test-secret-key")
    monkeypatch.setenv("AUDIT_LOG_PATH", str(logf))
    _clear_app_caches()

    from app.main import create_app
    from app.models.domain import SummarizeResult

    app = create_app()
    mock_triage_svc = MagicMock()
    mock_triage_svc.triage.return_value = mock_triage_result
    mock_quality_svc = MagicMock()
    mock_quality_svc.evaluate.return_value = mock_quality_result
    mock_pipeline_svc = MagicMock()
    mock_pipeline_svc.run.return_value = mock_pipeline_result
    mock_summarize_svc = MagicMock()
    mock_summarize_svc.summarize.return_value = SummarizeResult(
        summary="s", key_points=["a"], confidence=0.5
    )
    mock_cache = MagicMock()
    mock_cache.get.return_value = None
    mock_cache.ping.return_value = True

    app.dependency_overrides[get_triage_service] = lambda: mock_triage_svc
    app.dependency_overrides[get_quality_service] = lambda: mock_quality_svc
    app.dependency_overrides[get_pipeline_service] = lambda: mock_pipeline_svc
    app.dependency_overrides[get_summarization_service] = lambda: mock_summarize_svc
    app.dependency_overrides[get_cache] = lambda: mock_cache

    with TestClient(app) as client:
        client.post(
            "/api/v1/triage",
            json={"ticket_text": "This is long enough for validation to pass here."},
            headers={"X-API-Key": "unit-test-secret-key"},
        )

    app.dependency_overrides.clear()
    _clear_app_caches()

    assert logf.is_file()
    line = logf.read_text(encoding="utf-8").strip().splitlines()[0]
    entry = json.loads(line)
    assert entry["path"] == "/api/v1/triage"
    assert entry["status_code"] == 200

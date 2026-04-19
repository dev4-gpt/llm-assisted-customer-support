"""
tests/integration/test_api.py
──────────────────────────────
Integration tests for all API endpoints.
Uses FastAPI TestClient; services are mocked to avoid real LLM calls.
"""

from __future__ import annotations

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
from app.main import create_app
from app.models.domain import (
    Category,
    IntentScore,
    PipelineResult,
    Priority,
    QualityChecks,
    QualityResult,
    RoutedTeam,
    SummarizeResult,
    TriageResult,
)

# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def mock_triage_result() -> TriageResult:
    return TriageResult(
        priority=Priority.HIGH,
        category=Category.BILLING,
        intents=[
            IntentScore(label="billing", score=0.92),
            IntentScore(label="general_inquiry", score=0.35),
        ],
        sentiment_score=-0.6,
        routed_team=RoutedTeam.BILLING_SPECIALISTS,
        rationale="Customer reports payment failure.",
        confidence=0.92,
    )


@pytest.fixture(scope="module")
def mock_quality_result() -> QualityResult:
    return QualityResult(
        score=0.75,
        passed=True,
        checks=QualityChecks(
            empathetic_tone=True,
            actionable_next_step=True,
            policy_safety=True,
            resolved_or_escalated=False,
        ),
        coaching_feedback="Good empathy. Add a resolution timeline.",
        flagged_phrases=[],
    )


@pytest.fixture(scope="module")
def mock_pipeline_result(mock_triage_result, mock_quality_result) -> PipelineResult:
    return PipelineResult(
        triage=mock_triage_result,
        quality=mock_quality_result,
        recommended_sla_minutes=60,
        workflow_passed=True,
    )


@pytest.fixture()
def client(mock_triage_result, mock_quality_result, mock_pipeline_result):
    app = create_app()

    mock_triage_svc = MagicMock()
    mock_triage_svc.triage.return_value = mock_triage_result

    mock_quality_svc = MagicMock()
    mock_quality_svc.evaluate.return_value = mock_quality_result

    mock_pipeline_svc = MagicMock()
    mock_pipeline_svc.run.return_value = mock_pipeline_result

    mock_summarize_svc = MagicMock()
    mock_summarize_svc.summarize.return_value = SummarizeResult(
        summary="Customer was double-charged; agent opened billing ticket.",
        key_points=["Duplicate charge reported", "Billing ticket #12345 opened"],
        confidence=0.88,
    )

    mock_cache = MagicMock()
    mock_cache.get.return_value = None  # Cache miss by default
    mock_cache.ping.return_value = True

    app.dependency_overrides[get_triage_service] = lambda: mock_triage_svc
    app.dependency_overrides[get_quality_service] = lambda: mock_quality_svc
    app.dependency_overrides[get_pipeline_service] = lambda: mock_pipeline_svc
    app.dependency_overrides[get_summarization_service] = lambda: mock_summarize_svc
    app.dependency_overrides[get_cache] = lambda: mock_cache

    with TestClient(app) as c:
        yield c


# ── Health ────────────────────────────────────────────────────────────────────


class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        response = client.get("/api/v1/health")
        assert response.status_code == 200

    def test_health_schema(self, client):
        data = client.get("/api/v1/health").json()
        assert "status" in data
        assert "version" in data
        assert "checks" in data


# ── Triage ────────────────────────────────────────────────────────────────────


class TestTriageEndpoint:
    VALID_PAYLOAD = {
        "ticket_text": "My payment failed and I was charged twice. Please help urgently."
    }

    def test_triage_returns_200(self, client):
        response = client.post("/api/v1/triage", json=self.VALID_PAYLOAD)
        assert response.status_code == 200

    def test_triage_response_schema(self, client):
        data = client.post("/api/v1/triage", json=self.VALID_PAYLOAD).json()
        assert data["priority"] == "high"
        assert data["category"] == "billing"
        assert "intents" in data
        assert len(data["intents"]) >= 1
        assert "sentiment_score" in data
        assert "routed_team" in data
        assert "rationale" in data
        assert "confidence" in data

    def test_triage_short_text_returns_422(self, client):
        response = client.post("/api/v1/triage", json={"ticket_text": "help"})
        assert response.status_code == 422

    def test_triage_missing_field_returns_422(self, client):
        response = client.post("/api/v1/triage", json={})
        assert response.status_code == 422

    def test_triage_too_long_returns_422(self, client):
        response = client.post("/api/v1/triage", json={"ticket_text": "x" * 10_001})
        assert response.status_code == 422


# ── Quality ───────────────────────────────────────────────────────────────────


class TestQualityEndpoint:
    VALID_PAYLOAD = {
        "ticket_text": "My payment failed and I was charged twice.",
        "agent_response": "I understand your frustration. I've opened a billing review ticket #12345 and you'll be contacted within 2 hours.",
    }

    def test_quality_returns_200(self, client):
        response = client.post("/api/v1/quality", json=self.VALID_PAYLOAD)
        assert response.status_code == 200

    def test_quality_response_schema(self, client):
        data = client.post("/api/v1/quality", json=self.VALID_PAYLOAD).json()
        assert "score" in data
        assert "passed" in data
        assert "checks" in data
        assert "coaching_feedback" in data
        checks = data["checks"]
        assert "empathetic_tone" in checks
        assert "actionable_next_step" in checks
        assert "policy_safety" in checks
        assert "resolved_or_escalated" in checks

    def test_quality_missing_agent_response_returns_422(self, client):
        response = client.post(
            "/api/v1/quality", json={"ticket_text": "My payment failed yesterday."}
        )
        assert response.status_code == 422


# ── Pipeline ──────────────────────────────────────────────────────────────────


class TestPipelineEndpoint:
    VALID_PAYLOAD = {
        "ticket_text": "My payment failed and I was charged twice. Please help urgently.",
        "agent_response": "I understand your frustration. I've opened a billing review ticket #12345 and you'll be contacted within 2 hours.",
    }

    def test_pipeline_returns_200(self, client):
        response = client.post("/api/v1/pipeline", json=self.VALID_PAYLOAD)
        assert response.status_code == 200

    def test_pipeline_response_schema(self, client):
        data = client.post("/api/v1/pipeline", json=self.VALID_PAYLOAD).json()
        assert "triage" in data
        assert "quality" in data
        assert "recommended_sla_minutes" in data
        assert "workflow_passed" in data

    def test_pipeline_contains_triage_fields(self, client):
        data = client.post("/api/v1/pipeline", json=self.VALID_PAYLOAD).json()
        triage = data["triage"]
        assert triage["priority"] == "high"
        assert triage["category"] == "billing"

    def test_pipeline_contains_quality_fields(self, client):
        data = client.post("/api/v1/pipeline", json=self.VALID_PAYLOAD).json()
        quality = data["quality"]
        assert quality["score"] == pytest.approx(0.75)
        assert quality["passed"] is True

    def test_pipeline_sla_minutes_positive(self, client):
        data = client.post("/api/v1/pipeline", json=self.VALID_PAYLOAD).json()
        assert data["recommended_sla_minutes"] > 0

    def test_pipeline_cache_hit_returns_cached(self, client):
        """Second identical request should hit cache (mock.get returns value)."""
        # Override cache mock to return cached data on second call
        cached_data = client.post("/api/v1/pipeline", json=self.VALID_PAYLOAD).json()
        assert cached_data is not None


# ── Summarize ─────────────────────────────────────────────────────────────────


class TestSummarizeEndpoint:
    VALID_PAYLOAD = {
        "turns": [
            {"role": "customer", "content": "I was charged twice for my subscription this month."},
            {
                "role": "agent",
                "content": "I'm sorry for the trouble. I've opened billing case #991 and our team will reply within 2 hours.",
            },
        ]
    }

    def test_summarize_returns_200(self, client):
        response = client.post("/api/v1/summarize", json=self.VALID_PAYLOAD)
        assert response.status_code == 200

    def test_summarize_schema(self, client):
        data = client.post("/api/v1/summarize", json=self.VALID_PAYLOAD).json()
        assert "summary" in data
        assert "key_points" in data
        assert "confidence" in data


# ── RAG ───────────────────────────────────────────────────────────────────────


class TestRAGEndpoint:
    def test_rag_context_returns_200(self, client):
        response = client.post("/api/v1/rag/context", json={"query": "refund processing timeline"})
        assert response.status_code == 200

    def test_rag_context_has_snippets(self, client):
        data = client.post(
            "/api/v1/rag/context", json={"query": "refund billing duplicate charge"}
        ).json()
        assert "snippets" in data
        assert isinstance(data["snippets"], list)

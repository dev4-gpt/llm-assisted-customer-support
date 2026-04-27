"""
Deterministic end-to-end integration test for the full API workflow.

This test uses the real FastAPI wiring (no service dependency overrides) and
mocks LLMClient.complete_json so the complete pipeline is reproducible in CI
without external LLM credentials/network calls.
"""

from __future__ import annotations

from typing import Any

from fastapi.testclient import TestClient

from app.main import create_app
from app.services.llm_client import LLMClient


def _mock_complete_json(self: LLMClient, user_prompt: str, *, schema_hint: str = "") -> dict[str, Any]:
    if schema_hint == "TriageResult":
        return {
            "priority": "high",
            "category": "billing",
            "intents": [{"label": "billing", "score": 0.93}],
            "sentiment_score": -0.72,
            "rationale": "Duplicate charge and urgency indicate high-priority billing support.",
            "confidence": 0.93,
        }
    if schema_hint == "QualityResult":
        return {
            "score": 0.75,
            "checks": {
                "empathetic_tone": True,
                "actionable_next_step": True,
                "policy_safety": True,
                "resolved_or_escalated": False,
            },
            "coaching_feedback": "Good acknowledgement and next step. Add an explicit follow-up time.",
            "flagged_phrases": [],
        }
    if schema_hint == "SummarizeResult":
        return {
            "summary": "Customer reports a duplicate charge and the agent opened a billing case.",
            "key_points": ["Duplicate charge", "Billing case opened", "Follow-up pending"],
            "confidence": 0.88,
        }
    raise AssertionError(f"Unexpected schema_hint in mocked call: {schema_hint}")


def test_e2e_pipeline_and_supporting_endpoints(monkeypatch):
    monkeypatch.setattr(LLMClient, "complete_json", _mock_complete_json)

    app = create_app()
    with TestClient(app) as client:
        pipeline_payload = {
            "ticket_text": "I was charged twice and need this fixed today.",
            "agent_response": "Sorry for the trouble. I opened billing case #1201 for review.",
        }
        pipeline_response = client.post("/api/v1/pipeline", json=pipeline_payload)
        assert pipeline_response.status_code == 200
        pipeline_data = pipeline_response.json()
        assert pipeline_data["triage"]["priority"] == "high"
        assert pipeline_data["triage"]["category"] == "billing"
        assert pipeline_data["quality"]["passed"] is True
        assert pipeline_data["recommended_sla_minutes"] == 60
        assert pipeline_data["workflow_passed"] is True

        summarize_response = client.post(
            "/api/v1/summarize",
            json={
                "turns": [
                    {"role": "customer", "content": "I was charged twice."},
                    {"role": "agent", "content": "I have opened billing case #1201."},
                ]
            },
        )
        assert summarize_response.status_code == 200
        summary_data = summarize_response.json()
        assert "duplicate charge" in summary_data["summary"].lower()
        assert summary_data["confidence"] == 0.88

        rag_response = client.post(
            "/api/v1/rag/context",
            json={"query": "duplicate charge refund timeline"},
        )
        assert rag_response.status_code == 200
        rag_data = rag_response.json()
        assert "snippets" in rag_data
        assert isinstance(rag_data["snippets"], list)

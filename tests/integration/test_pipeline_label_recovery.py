from __future__ import annotations

from unittest.mock import MagicMock

from fastapi.testclient import TestClient

from app.core.dependencies import get_cache, get_llm_client
from app.main import create_app


class _FakeLLM:
    def complete_json(self, _prompt: str, *, schema_hint: str = "") -> dict:
        if schema_hint == "TriageResult":
            return {
                "priority": "high",
                "category": "refund",
                "intents": [{"label": "refund", "score": 0.93}],
                "sentiment_score": -0.4,
                "rationale": "Customer requests a refund for duplicate charge.",
                "confidence": 0.93,
            }
        if schema_hint == "QualityResult":
            return {
                "score": 0.75,
                "checks": {
                    "empathetic_tone": True,
                    "actionable_next_step": True,
                    "policy_safety": True,
                    "resolved_or_escalated": True,
                },
                "coaching_feedback": "Good response with clear next action.",
                "flagged_phrases": [],
            }
        raise AssertionError(f"Unexpected schema_hint: {schema_hint}")


def test_pipeline_recovers_invalid_labels_to_allowed_taxonomy():
    app = create_app()
    mock_cache = MagicMock()
    mock_cache.get.return_value = None
    mock_cache.ping.return_value = True

    app.dependency_overrides[get_llm_client] = lambda: _FakeLLM()
    app.dependency_overrides[get_cache] = lambda: mock_cache

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/pipeline",
            json={
                "ticket_text": "I was charged twice and want a refund immediately.",
                "agent_response": "I opened a billing case and will update you shortly.",
            },
        )
        assert response.status_code == 200
        payload = response.json()
        assert payload["triage"]["category"] == "billing"
        intent_labels = {item["label"] for item in payload["triage"]["intents"]}
        assert "billing" in intent_labels

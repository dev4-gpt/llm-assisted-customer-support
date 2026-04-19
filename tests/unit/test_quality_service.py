"""
tests/unit/test_quality_service.py
────────────────────────────────────
Unit tests for QualityService.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from app.core.exceptions import ValidationError
from app.models.domain import QualityRequest, RAGContextResponse
from app.services.quality_service import QualityService


@pytest.fixture()
def settings():
    from app.core.config import Settings

    return Settings(
        app_secret_key="test-secret-key-1234",
        quality_pass_threshold=0.70,
        quality_policy_context_top_k=0,
    )


@pytest.fixture()
def mock_llm():
    return MagicMock()


@pytest.fixture()
def mock_rag():
    m = MagicMock()
    m.retrieve.return_value = RAGContextResponse(snippets=[])
    return m


@pytest.fixture()
def service(mock_llm, settings, mock_rag):
    return QualityService(mock_llm, settings, mock_rag)


SAMPLE_REQUEST = QualityRequest(
    ticket_text="My payment failed and I need help immediately.",
    agent_response="I understand how frustrating this is. I've flagged your account for billing review and you'll hear back within 2 hours. Ticket #12345 has been created.",
)


class TestQualityService:
    def test_perfect_response_passes(self, service, mock_llm):
        mock_llm.complete_json.return_value = {
            "score": 1.0,
            "checks": {
                "empathetic_tone": True,
                "actionable_next_step": True,
                "policy_safety": True,
                "resolved_or_escalated": True,
            },
            "coaching_feedback": "Excellent response. All quality dimensions met.",
            "flagged_phrases": [],
        }

        result = service.evaluate(SAMPLE_REQUEST)

        assert result.passed is True
        assert result.score == pytest.approx(1.0)
        assert result.checks.empathetic_tone is True
        assert result.checks.resolved_or_escalated is True
        assert result.flagged_phrases == []

    def test_low_quality_response_fails(self, service, mock_llm):
        mock_llm.complete_json.return_value = {
            "score": 0.25,
            "checks": {
                "empathetic_tone": False,
                "actionable_next_step": False,
                "policy_safety": True,
                "resolved_or_escalated": False,
            },
            "coaching_feedback": "Response lacks empathy and provides no next steps.",
            "flagged_phrases": ["not our problem"],
        }

        result = service.evaluate(SAMPLE_REQUEST)

        assert result.passed is False
        assert result.score < 0.70
        assert "not our problem" in result.flagged_phrases

    def test_score_at_threshold_passes(self, service, mock_llm):
        mock_llm.complete_json.return_value = {
            "score": 0.70,
            "checks": {
                "empathetic_tone": True,
                "actionable_next_step": True,
                "policy_safety": True,
                "resolved_or_escalated": False,
            },
            "coaching_feedback": "Almost there.",
            "flagged_phrases": [],
        }

        result = service.evaluate(SAMPLE_REQUEST)

        assert result.passed is True

    def test_score_just_below_threshold_fails(self, service, mock_llm):
        mock_llm.complete_json.return_value = {
            "score": 0.69,
            "checks": {
                "empathetic_tone": True,
                "actionable_next_step": True,
                "policy_safety": True,
                "resolved_or_escalated": False,
            },
            "coaching_feedback": "Close but needs improvement.",
            "flagged_phrases": [],
        }

        result = service.evaluate(SAMPLE_REQUEST)

        assert result.passed is False

    def test_boolean_string_normalisation(self, service, mock_llm):
        """LLMs sometimes return 'true'/'false' strings instead of booleans."""
        mock_llm.complete_json.return_value = {
            "score": 0.75,
            "checks": {
                "empathetic_tone": "true",
                "actionable_next_step": "true",
                "policy_safety": "false",
                "resolved_or_escalated": "true",
            },
            "coaching_feedback": "Policy section needs review.",
            "flagged_phrases": [],
        }

        result = service.evaluate(SAMPLE_REQUEST)

        assert result.checks.empathetic_tone is True
        assert result.checks.policy_safety is False

    def test_invalid_score_range_raises_validation_error(self, service, mock_llm):
        mock_llm.complete_json.return_value = {
            "score": 1.5,  # Out of range
            "checks": {
                "empathetic_tone": True,
                "actionable_next_step": True,
                "policy_safety": True,
                "resolved_or_escalated": True,
            },
            "coaching_feedback": "Great.",
            "flagged_phrases": [],
        }

        with pytest.raises(ValidationError, match="out of range"):
            service.evaluate(SAMPLE_REQUEST)

    def test_policy_snippets_injected_when_enabled(self, mock_llm, mock_rag):
        from app.core.config import Settings
        from app.models.domain import RAGSnippet

        settings = Settings(
            app_secret_key="test-secret-key-1234",
            quality_pass_threshold=0.70,
            quality_policy_context_top_k=2,
        )
        mock_rag.retrieve.return_value = RAGContextResponse(
            snippets=[
                RAGSnippet(id="p1", title="Refunds", body="Refunds within 5 days.", score=0.9)
            ]
        )
        mock_llm.complete_json.return_value = {
            "score": 0.8,
            "checks": {
                "empathetic_tone": True,
                "actionable_next_step": True,
                "policy_safety": True,
                "resolved_or_escalated": True,
            },
            "coaching_feedback": "OK",
            "flagged_phrases": [],
        }
        svc = QualityService(mock_llm, settings, mock_rag)
        svc.evaluate(SAMPLE_REQUEST)
        mock_rag.retrieve.assert_called_once()
        call_kw = mock_llm.complete_json.call_args[0][0]
        assert "RELEVANT POLICY SNIPPETS" in call_kw
        assert "Refunds" in call_kw

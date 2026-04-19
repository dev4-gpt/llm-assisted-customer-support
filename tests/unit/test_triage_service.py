"""
tests/unit/test_triage_service.py
──────────────────────────────────
Unit tests for TriageService.
All LLM calls are mocked – no real API calls are made.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from app.core.exceptions import LLMParseError, ValidationError
from app.models.domain import Category, Priority, RAGContextResponse, RoutedTeam, TriageRequest
from app.services.triage_service import TriageService


@pytest.fixture()
def settings():
    from app.core.config import Settings

    return Settings(
        app_secret_key="test-secret-key-1234",
        triage_policy_context_top_k=0,
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
    return TriageService(mock_llm, settings, mock_rag)


class TestTriageService:
    def test_critical_billing_ticket(self, service, mock_llm):
        mock_llm.complete_json.return_value = {
            "priority": "critical",
            "category": "billing",
            "sentiment_score": -0.55,
            "rationale": "Customer reports double charge. Urgent financial issue.",
            "confidence": 0.95,
        }

        result = service.triage(TriageRequest(ticket_text="I was charged twice! Refund now!"))

        assert result.priority == Priority.CRITICAL
        assert result.category == Category.BILLING
        assert result.routed_team == RoutedTeam.CRITICAL_RESPONSE
        assert result.sentiment_score == pytest.approx(-0.55, abs=0.001)

    def test_high_priority_auth_escalates_on_negative_sentiment(self, service, mock_llm):
        mock_llm.complete_json.return_value = {
            "priority": "high",
            "category": "authentication",
            "sentiment_score": -0.65,  # Strictly below escalation cutoff of -0.6
            "rationale": "Customer locked out.",
            "confidence": 0.88,
        }

        result = service.triage(TriageRequest(ticket_text="I cannot log in at all!"))

        assert result.routed_team == RoutedTeam.ESCALATIONS

    def test_medium_technical_bug_routes_to_tier2(self, service, mock_llm):
        mock_llm.complete_json.return_value = {
            "priority": "medium",
            "category": "technical_bug",
            "sentiment_score": -0.2,
            "rationale": "Bug report with moderate impact.",
            "confidence": 0.80,
        }

        result = service.triage(TriageRequest(ticket_text="The export button is broken"))

        assert result.routed_team == RoutedTeam.TIER2_ENGINEERING

    def test_feature_request_routes_to_product_team(self, service, mock_llm):
        mock_llm.complete_json.return_value = {
            "priority": "low",
            "category": "feature_request",
            "sentiment_score": 0.5,
            "rationale": "New feature suggestion.",
            "confidence": 0.90,
        }

        result = service.triage(TriageRequest(ticket_text="Would love dark mode please!"))

        assert result.routed_team == RoutedTeam.PRODUCT_TEAM

    def test_invalid_priority_raises_validation_error(self, service, mock_llm):
        mock_llm.complete_json.return_value = {
            "priority": "super_urgent",  # Invalid
            "category": "billing",
            "sentiment_score": -0.5,
            "rationale": "Test.",
            "confidence": 0.5,
        }

        with pytest.raises(ValidationError, match="invalid priority"):
            service.triage(TriageRequest(ticket_text="My payment failed completely"))

    def test_missing_field_raises_parse_error(self, service, mock_llm):
        mock_llm.complete_json.return_value = {
            "priority": "high",
            # Missing: category, sentiment_score, rationale, confidence
        }

        with pytest.raises(LLMParseError):
            service.triage(TriageRequest(ticket_text="My payment failed completely"))

    def test_sentiment_at_boundary_does_not_escalate(self, service, mock_llm):
        """Sentiment exactly at cutoff (-0.6) should NOT escalate."""
        mock_llm.complete_json.return_value = {
            "priority": "high",
            "category": "technical_bug",
            "sentiment_score": -0.6,  # At boundary, not below
            "rationale": "Boundary test.",
            "confidence": 0.75,
        }

        result = service.triage(TriageRequest(ticket_text="Things are broken and I am unhappy"))

        # -0.6 is equal to cutoff; escalation requires strictly below
        assert result.routed_team == RoutedTeam.TIER2_ENGINEERING

    def test_critical_negative_sentiment_escalates(self, service, mock_llm):
        mock_llm.complete_json.return_value = {
            "priority": "critical",
            "category": "billing",
            "sentiment_score": -0.9,
            "rationale": "Angry customer, financial issue.",
            "confidence": 0.9,
        }

        result = service.triage(TriageRequest(ticket_text="I was charged twice and I am furious."))

        assert result.priority == Priority.CRITICAL
        assert result.routed_team == RoutedTeam.ESCALATIONS

    def test_multi_intent_returns_sorted_intents(self, service, mock_llm):
        mock_llm.complete_json.return_value = {
            "priority": "medium",
            "category": "billing",
            "intents": [
                {"label": "general_inquiry", "score": 0.4},
                {"label": "billing", "score": 0.9},
            ],
            "sentiment_score": -0.2,
            "rationale": "Refund and account question.",
            "confidence": 0.85,
        }

        result = service.triage(
            TriageRequest(ticket_text="I need a refund and also update my address.")
        )

        assert result.category == Category.BILLING
        assert result.intents[0].score >= result.intents[1].score
        labels = {i.label for i in result.intents}
        assert "billing" in labels and "general_inquiry" in labels

    def test_policy_context_prepended_when_enabled(self, mock_llm, mock_rag):
        from app.core.config import Settings
        from app.models.domain import RAGSnippet

        settings = Settings(
            app_secret_key="test-secret-key-1234",
            triage_policy_context_top_k=2,
        )
        mock_rag.retrieve.return_value = RAGContextResponse(
            snippets=[
                RAGSnippet(id="x", title="SLA", body="Critical tickets in 15 minutes.", score=0.8)
            ]
        )
        mock_llm.complete_json.return_value = {
            "priority": "high",
            "category": "technical_bug",
            "intents": [{"label": "technical_bug", "score": 0.9}],
            "sentiment_score": -0.3,
            "rationale": "Bug report.",
            "confidence": 0.88,
        }
        svc = TriageService(mock_llm, settings, mock_rag)
        svc.triage(TriageRequest(ticket_text="The dashboard export crashes every time I use it."))
        prompt = mock_llm.complete_json.call_args[0][0]
        assert "POLICY CONTEXT" in prompt
        assert "Critical tickets" in prompt

    def test_hybrid_classifier_hint_in_prompt(self, tmp_path, mock_llm, mock_rag):
        from app.core.config import Settings

        pytest.importorskip("sklearn")
        import joblib
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline

        model = Pipeline(
            [
                ("tfidf", TfidfVectorizer(max_features=100)),
                ("clf", LogisticRegression(max_iter=300, random_state=42)),
            ]
        )
        model.fit(
            ["billing duplicate charge help", "cannot log in password reset fails"],
            ["billing", "authentication"],
        )
        mpath = tmp_path / "baseline.joblib"
        joblib.dump(model, mpath)

        settings = Settings(
            app_secret_key="test-secret-key-1234",
            triage_hybrid_enabled=True,
            triage_baseline_model_path=mpath,
        )
        mock_llm.complete_json.return_value = {
            "priority": "high",
            "category": "billing",
            "intents": [{"label": "billing", "score": 0.9}],
            "sentiment_score": -0.4,
            "rationale": "Billing issue.",
            "confidence": 0.9,
        }
        svc = TriageService(mock_llm, settings, mock_rag)
        svc.triage(TriageRequest(ticket_text="billing duplicate charge help"))
        prompt = mock_llm.complete_json.call_args[0][0]
        assert "baseline classifier" in prompt.lower()
        assert "billing" in prompt.lower()

    def test_transformer_hint_in_prompt(self, tmp_path, mock_llm, mock_rag):
        from app.core.config import Settings

        fake = MagicMock()
        fake.predict_category.return_value = "authentication"

        settings = Settings(
            app_secret_key="test-secret-key-1234",
            triage_transformer_enabled=True,
            triage_transformer_model_dir=tmp_path,
        )
        mock_llm.complete_json.return_value = {
            "priority": "high",
            "category": "authentication",
            "intents": [{"label": "authentication", "score": 0.92}],
            "sentiment_score": -0.3,
            "rationale": "Login issue.",
            "confidence": 0.92,
        }
        with patch(
            "app.services.triage_transformer_predict.load_triage_transformer",
            return_value=fake,
        ):
            svc = TriageService(mock_llm, settings, mock_rag)
            svc.triage(TriageRequest(ticket_text="password reset not working"))
        prompt = mock_llm.complete_json.call_args[0][0]
        assert "fine-tuned encoder" in prompt.lower()
        assert "authentication" in prompt.lower()
        fake.predict_category.assert_called_once()

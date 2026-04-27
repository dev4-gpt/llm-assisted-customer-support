from __future__ import annotations

from app.core.config import Settings
from app.services.intent_fallback_service import IntentFallbackService


def _settings(**kwargs) -> Settings:
    base = {"app_secret_key": "test-secret-key-1234"}
    base.update(kwargs)
    return Settings(**base)


def test_synonym_maps_refund_to_billing():
    svc = IntentFallbackService(_settings(triage_embedding_fallback_enabled=False))
    mapped = svc.map_to_valid_category("refund")
    assert mapped == "billing"


def test_token_synonym_maps_password_phrase_to_authentication():
    svc = IntentFallbackService(_settings(triage_embedding_fallback_enabled=False))
    mapped = svc.map_to_valid_category("password reset blocked")
    assert mapped == "authentication"


def test_valid_category_passthrough():
    svc = IntentFallbackService(_settings(triage_embedding_fallback_enabled=False))
    mapped = svc.map_to_valid_category("technical_bug")
    assert mapped == "technical_bug"

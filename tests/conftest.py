"""
tests/conftest.py
──────────────────
Shared pytest fixtures and configuration.
Sets env vars before Settings() is instantiated so validation passes.
"""

from __future__ import annotations

import os

import pytest

# Default: hosted OpenAI-compatible path for tests (no real network call required).
os.environ.setdefault("LLM_PROVIDER", "openai_compatible")
os.environ.setdefault("LLM_MODEL", "google/gemini-2.0-flash-exp:free")
os.environ.setdefault("OPENAI_COMPATIBLE_BASE_URL", "https://openrouter.ai/api/v1")
os.environ.setdefault("OPENAI_COMPATIBLE_API_KEY", "test-api-key")
os.environ.setdefault("APP_SECRET_KEY", "test-secret-key-minimum-16chars")
os.environ.setdefault("QUALITY_POLICY_CONTEXT_TOP_K", "0")
os.environ.setdefault("TRIAGE_POLICY_CONTEXT_TOP_K", "0")


@pytest.fixture(autouse=True)
def reset_settings_cache():
    """Clear the lru_cache on get_settings between tests to avoid state leakage."""
    from app.core.config import get_settings
    from app.core.dependencies import (
        get_cache,
        get_llm_client,
        get_pipeline_service,
        get_quality_service,
        get_rag_service,
        get_summarization_service,
        get_triage_service,
    )

    for fn in (
        get_settings,
        get_llm_client,
        get_cache,
        get_rag_service,
        get_triage_service,
        get_quality_service,
        get_pipeline_service,
        get_summarization_service,
    ):
        fn.cache_clear()
    yield
    for fn in (
        get_settings,
        get_llm_client,
        get_cache,
        get_rag_service,
        get_triage_service,
        get_quality_service,
        get_pipeline_service,
        get_summarization_service,
    ):
        fn.cache_clear()

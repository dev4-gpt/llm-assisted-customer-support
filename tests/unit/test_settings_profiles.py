from __future__ import annotations

import pytest

from app.core.config import Settings


def test_openrouter_profile_applies_provider_and_key() -> None:
    s = Settings(
        app_secret_key="test-secret-key-1234",
        llm_profile="openrouter",
        openrouter_api_key="sk-or-test",
    )
    assert s.llm_provider == "openai_compatible"
    assert s.openai_compatible_base_url == "https://openrouter.ai/api/v1"
    assert s.openai_compatible_api_key == "sk-or-test"


def test_nvidia_profile_applies_provider_and_key() -> None:
    s = Settings(
        app_secret_key="test-secret-key-1234",
        llm_profile="nvidia",
        nvidia_api_key="nvapi-test",
    )
    assert s.llm_provider == "openai_compatible"
    assert s.openai_compatible_base_url == "https://integrate.api.nvidia.com/v1"
    assert s.openai_compatible_api_key == "nvapi-test"


def test_ollama_profile_uses_local_defaults() -> None:
    s = Settings(
        app_secret_key="test-secret-key-1234",
        llm_profile="ollama",
        ollama_model="qwen2.5:7b-instruct",
    )
    assert s.llm_provider == "ollama"
    assert s.openai_compatible_base_url == "http://127.0.0.1:11434/v1"
    assert s.openai_compatible_api_key == "ollama"
    assert s.llm_model == "qwen2.5:7b-instruct"


def test_manual_openai_provider_requires_real_api_key() -> None:
    with pytest.raises(ValueError, match="OPENAI_COMPATIBLE_API_KEY is not set"):
        Settings(
            app_secret_key="test-secret-key-1234",
            llm_profile="manual",
            llm_provider="openai_compatible",
            openai_compatible_api_key="replace-with-real-api-key",
        )

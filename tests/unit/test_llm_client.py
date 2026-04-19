"""
tests/unit/test_llm_client.py
──────────────────────────────
Unit tests for LLMClient JSON parsing and provider-specific HTTP/API behavior.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import anthropic
import httpx
import pytest

from app.core.config import Settings
from app.core.exceptions import LLMParseError, LLMRateLimitError, LLMTimeoutError
from app.services.llm_client import LLMClient


@pytest.fixture()
def settings_ollama() -> Settings:
    return Settings(
        llm_provider="ollama",
        app_secret_key="test-secret-key-1234",
    )


@pytest.fixture()
def settings_anthropic() -> Settings:
    return Settings(
        llm_provider="anthropic",
        anthropic_api_key="sk-ant-test",
        llm_model="claude-sonnet-4-20250514",
        app_secret_key="test-secret-key-1234",
    )


@pytest.fixture()
def client_ollama(settings_ollama: Settings) -> LLMClient:
    return LLMClient(settings_ollama)


@pytest.fixture()
def client_anthropic(settings_anthropic: Settings) -> LLMClient:
    return LLMClient(settings_anthropic)


class TestLLMClientJsonParsing:
    def test_clean_json_parses(self, client_ollama: LLMClient):
        raw = '{"priority": "high", "score": 0.8}'
        result = client_ollama._parse_json(raw)
        assert result["priority"] == "high"

    def test_json_with_markdown_fence_parses(self, client_ollama: LLMClient):
        raw = '```json\n{"priority": "high"}\n```'
        result = client_ollama._parse_json(raw)
        assert result["priority"] == "high"

    def test_json_with_plain_fence_parses(self, client_ollama: LLMClient):
        raw = '```\n{"priority": "critical"}\n```'
        result = client_ollama._parse_json(raw)
        assert result["priority"] == "critical"

    def test_invalid_json_raises_parse_error(self, client_ollama: LLMClient):
        with pytest.raises(LLMParseError):
            client_ollama._parse_json("This is not JSON at all.")

    def test_empty_string_raises_parse_error(self, client_ollama: LLMClient):
        with pytest.raises(LLMParseError):
            client_ollama._parse_json("")


class TestLLMClientOpenAICompatible:
    def test_complete_json_uses_http_and_parses(self, client_ollama: LLMClient):
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": '{"ok": true, "n": 1}'}}]
        }
        with patch.object(client_ollama._http_client, "post", return_value=mock_resp):
            out = client_ollama.complete_json("prompt", schema_hint="test")
        assert out == {"ok": True, "n": 1}


class TestLLMClientAnthropicErrors:
    def test_rate_limit_raises_rate_limit_error(self, client_anthropic: LLMClient):
        with patch.object(client_anthropic._anthropic_client.messages, "create") as mock_create:
            mock_create.side_effect = anthropic.RateLimitError(
                message="rate limited",
                response=MagicMock(status_code=429),
                body={},
            )
            with pytest.raises(LLMRateLimitError):
                client_anthropic._complete_anthropic_with_retry("test")

    def test_timeout_raises_timeout_error(self, client_anthropic: LLMClient):
        with patch.object(client_anthropic._anthropic_client.messages, "create") as mock_create:
            mock_create.side_effect = anthropic.APITimeoutError(request=MagicMock())
            with pytest.raises(LLMTimeoutError):
                client_anthropic._complete_anthropic_with_retry("test")


class TestLLMClientHttpErrors:
    def test_http_timeout_maps_to_timeout_error(self, settings_ollama: Settings):
        client = LLMClient(settings_ollama)
        with patch.object(client._http_client, "post") as mock_post:
            mock_post.side_effect = httpx.TimeoutException("timeout")
            with pytest.raises(LLMTimeoutError):
                client._complete_openai_compatible("x")

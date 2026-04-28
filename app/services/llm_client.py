"""
app/services/llm_client.py
──────────────────────────
LLM client for JSON-only prompts.

Default: OpenAI-compatible HTTP API (OpenRouter, NVIDIA NIM, vLLM, Ollama, etc.).
Optional: Anthropic Messages API when LLM_PROVIDER=anthropic.

Responsibilities:
  • Retry with exponential back-off (tenacity)
  • Structured JSON extraction with fallback parsing
  • Latency + error metrics
  • Centralised system prompt
"""

from __future__ import annotations

import json
import re
import time
from typing import Any

import httpx
from tenacity import (
    retry,
    retry_if_exception,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.core.config import Settings
from app.core.exceptions import LLMError, LLMParseError, LLMRateLimitError, LLMTimeoutError
from app.core.logging import get_logger
from app.utils.metrics import LLM_ERRORS, LLM_LATENCY_SECONDS, LLM_REQUESTS

logger = get_logger(__name__)

_SYSTEM_PROMPT = """\
You are a senior customer support AI specialised in ticket triage and quality monitoring.
Always respond with valid JSON that exactly matches the schema specified in the user prompt.
Do not include markdown fences, preamble, or any text outside the JSON object.
"""


class LLMEmptyResponseError(Exception):
    pass


def _retryable_http_exception(exc: BaseException) -> bool:
    if isinstance(exc, httpx.TimeoutException):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in (429, 503)
    if isinstance(exc, LLMEmptyResponseError):
        return True
    return False


class LLMClient:
    """
    LLM client: OpenAI-compatible chat completions (default) or Anthropic Messages API.

    Usage::

        client = LLMClient(settings)
        data = client.complete_json(prompt, schema_hint="TriageResult")
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._anthropic_client: Any = None
        self._http_client: httpx.Client | None = None

        if settings.llm_provider == "anthropic":
            try:
                import anthropic
            except ImportError as exc:
                raise LLMError(
                    "Anthropic provider selected but the 'anthropic' package is not installed. "
                    "Install with: pip install anthropic"
                ) from exc
            self._anthropic_client = anthropic.Anthropic(
                api_key=settings.anthropic_api_key or "",
                timeout=settings.llm_timeout_seconds,
                max_retries=0,
            )
        else:
            self._http_client = httpx.Client(timeout=settings.llm_timeout_seconds)

    # ── Public API ────────────────────────────────────────────────────────────

    def complete_json(self, user_prompt: str, *, schema_hint: str = "") -> dict[str, Any]:
        """
        Send a prompt to the LLM and parse the response as JSON.
        Raises LLMError subclasses on failure.
        """
        start = time.perf_counter()
        label = schema_hint or "generic"
        pv = self._settings.llm_prompt_version
        LLM_REQUESTS.labels(schema=label, prompt_version=pv).inc()

        try:
            raw = self._complete_with_retry(user_prompt)
            result = self._parse_json(raw)
            LLM_LATENCY_SECONDS.labels(schema=label, prompt_version=pv).observe(
                time.perf_counter() - start
            )
            return result

        except LLMError:
            LLM_ERRORS.labels(schema=label, error_type="llm_error", prompt_version=pv).inc()
            raise
        except Exception as exc:
            LLM_ERRORS.labels(schema=label, error_type="unexpected", prompt_version=pv).inc()
            logger.exception("Unexpected error in LLMClient.complete_json", exc_info=exc)
            raise LLMError(f"Unexpected LLM client error: {exc}") from exc

    # ── Internal ──────────────────────────────────────────────────────────────

    def _complete_with_retry(self, user_prompt: str) -> str:
        if self._settings.llm_provider == "anthropic":
            return self._complete_anthropic_with_retry(user_prompt)
        return self._complete_openai_compatible(user_prompt)

    def _complete_anthropic_with_retry(self, user_prompt: str) -> str:
        import anthropic

        @retry(
            retry=retry_if_exception_type((anthropic.RateLimitError, anthropic.APITimeoutError)),
            wait=wait_exponential(multiplier=1, min=2, max=30),
            stop=stop_after_attempt(3),
            reraise=True,
        )
        def _call() -> str:
            assert self._anthropic_client is not None
            message = self._anthropic_client.messages.create(
                model=self._settings.llm_model,
                max_tokens=self._settings.llm_max_tokens,
                system=_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_prompt}],
            )
            return message.content[0].text

        try:
            return _call()
        except anthropic.RateLimitError as exc:
            logger.warning("Anthropic rate limit after retries")
            raise LLMRateLimitError() from exc
        except anthropic.APITimeoutError as exc:
            logger.warning("Anthropic timeout after retries")
            raise LLMTimeoutError() from exc
        except anthropic.AuthenticationError as exc:
            raise LLMError("Invalid Anthropic API key.") from exc
        except anthropic.APIError as exc:
            raise LLMError(f"Anthropic API error: {exc}") from exc

    def _complete_openai_compatible(self, user_prompt: str) -> str:
        """POST /v1/chat/completions (OpenAI-compatible gateways)."""
        try:
            return self._post_chat_completion_json(user_prompt)
        except httpx.TimeoutException as exc:
            logger.warning("LLM HTTP timeout after retries")
            raise LLMTimeoutError() from exc
        except httpx.HTTPStatusError as exc:
            code = exc.response.status_code
            if code == 429:
                raise LLMRateLimitError() from exc
            raise LLMError(f"LLM HTTP error {code}: {exc.response.text[:500]}") from exc
        except httpx.RequestError as exc:
            raise LLMError(f"LLM connection error: {exc}") from exc

    @retry(
        retry=retry_if_exception(_retryable_http_exception),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        stop=stop_after_attempt(3),
        reraise=True,
    )
    def _post_chat_completion_json(self, user_prompt: str) -> str:
        """Inner call: raises raw httpx exceptions so tenacity can retry 429/503/timeouts."""
        assert self._http_client is not None
        base = self._settings.openai_compatible_base_url.rstrip("/")
        url = f"{base}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self._settings.openai_compatible_api_key}",
            "Content-Type": "application/json",
        }
        payload: dict[str, Any] = {
            "model": self._settings.llm_model,
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            "max_tokens": self._settings.llm_max_tokens,
            "temperature": self._settings.llm_temperature,
        }
        response = self._http_client.post(url, json=payload, headers=headers)
        response.raise_for_status()

        try:
            data = response.json()
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"Unexpected chat completion response: {response.text[:800]}") from exc

        if not isinstance(content, str):
            raise LLMError("LLM response content is not a string")
        if not content.strip():
            logger.warning("LLM returned an empty string, triggering retry.")
            raise LLMEmptyResponseError("Empty string from LLM")
        return content

    @staticmethod
    def _parse_json(raw: str) -> dict[str, Any]:
        """Parse JSON from the LLM response, stripping markdown fences if present."""
        text = raw.strip()

        # Strip ```json ... ``` fences defensively
        fence_match = re.search(r"```(?:json)?\s*([\s\S]+?)\s*```", text)
        if fence_match:
            text = fence_match.group(1)

        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise LLMParseError(raw) from exc

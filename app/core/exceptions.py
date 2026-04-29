"""
app/core/exceptions.py
──────────────────────
Domain exception hierarchy. Keep HTTP concerns out of here;
exception handlers in app/api/v1/errors.py map these to HTTP responses.
"""

from __future__ import annotations


class SupportTriageError(Exception):
    """Base exception for all application errors."""

    def __init__(self, message: str, *, code: str = "INTERNAL_ERROR") -> None:
        super().__init__(message)
        self.message = message
        self.code = code


# ── Validation ───────────────────────────────────────────────────────────────


class ValidationError(SupportTriageError):
    """Input payload failed business-rule validation."""

    def __init__(self, message: str) -> None:
        super().__init__(message, code="VALIDATION_ERROR")


# ── LLM / Upstream ───────────────────────────────────────────────────────────


class LLMError(SupportTriageError):
    """LLM Provider API call failed."""

    def __init__(self, message: str) -> None:
        super().__init__(message, code="LLM_ERROR")


class LLMRateLimitError(LLMError):
    """LLM Provider rate-limit hit."""

    def __init__(self) -> None:
        super().__init__("LLM Provider rate limit exceeded. Retry after back-off.")
        self.code = "LLM_RATE_LIMIT"


class LLMTimeoutError(LLMError):
    """LLM Provider API timed out."""

    def __init__(self) -> None:
        super().__init__("LLM Provider API request timed out.")
        self.code = "LLM_TIMEOUT"


class LLMParseError(LLMError):
    """Could not parse structured data from LLM response."""

    def __init__(self, raw: str) -> None:
        super().__init__(f"Failed to parse LLM JSON response. Raw: {raw[:200]}")
        self.code = "LLM_PARSE_ERROR"
        self.raw = raw


# ── Infrastructure ───────────────────────────────────────────────────────────


class CacheError(SupportTriageError):
    """Redis cache operation failed (non-fatal)."""

    def __init__(self, message: str) -> None:
        super().__init__(message, code="CACHE_ERROR")

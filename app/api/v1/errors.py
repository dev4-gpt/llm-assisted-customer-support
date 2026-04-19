"""
app/api/v1/errors.py
─────────────────────
Maps domain exceptions → structured HTTP responses.
All error payloads share a consistent envelope so API consumers
can rely on a single error schema.
"""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.responses import ORJSONResponse

from app.core.exceptions import (
    LLMError,
    LLMRateLimitError,
    LLMTimeoutError,
    SupportTriageError,
    ValidationError,
)
from app.core.logging import get_logger

logger = get_logger(__name__)


def _error_response(code: str, message: str, status: int) -> ORJSONResponse:
    return ORJSONResponse(
        status_code=status,
        content={"error": {"code": code, "message": message}},
    )


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(ValidationError)
    async def handle_validation(request: Request, exc: ValidationError) -> ORJSONResponse:
        logger.warning("Validation error", path=request.url.path, message=exc.message)
        return _error_response(exc.code, exc.message, 422)

    @app.exception_handler(LLMRateLimitError)
    async def handle_rate_limit(request: Request, exc: LLMRateLimitError) -> ORJSONResponse:
        logger.warning("LLM rate limit", path=request.url.path)
        return _error_response(exc.code, exc.message, 429)

    @app.exception_handler(LLMTimeoutError)
    async def handle_timeout(request: Request, exc: LLMTimeoutError) -> ORJSONResponse:
        logger.warning("LLM timeout", path=request.url.path)
        return _error_response(exc.code, exc.message, 504)

    @app.exception_handler(LLMError)
    async def handle_llm_error(request: Request, exc: LLMError) -> ORJSONResponse:
        logger.error("LLM error", path=request.url.path, message=exc.message)
        return _error_response(exc.code, exc.message, 502)

    @app.exception_handler(SupportTriageError)
    async def handle_domain(request: Request, exc: SupportTriageError) -> ORJSONResponse:
        logger.error("Domain error", path=request.url.path, code=exc.code)
        return _error_response(exc.code, exc.message, 500)

    @app.exception_handler(Exception)
    async def handle_unhandled(request: Request, exc: Exception) -> ORJSONResponse:
        logger.exception("Unhandled exception", path=request.url.path, exc_info=exc)
        return _error_response("INTERNAL_ERROR", "An unexpected error occurred.", 500)

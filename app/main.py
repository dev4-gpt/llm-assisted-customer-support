"""
app/main.py
───────────
FastAPI application factory.

The create_app() factory pattern allows the app to be instantiated with
different settings for testing vs production without side effects at import time.
"""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import AsyncGenerator, Awaitable, Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import ORJSONResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from app.api.v1.errors import register_exception_handlers
from app.api.v1.routers import router
from app.core.config import get_settings
from app.core.logging import configure_logging, get_logger

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Run startup tasks before yielding, teardown tasks after."""
    settings = get_settings()
    configure_logging(
        log_level=settings.app_log_level,
        json_logs=settings.is_production,
    )
    logger.info(
        "Application starting",
        env=settings.app_env,
        llm_provider=settings.llm_provider,
        model=settings.llm_model,
    )
    yield
    logger.info("Application shutting down")


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title="Support Triage & Quality Monitoring API",
        description=(
            "LLM-augmented customer support triage and agent response quality evaluation. "
            "Default: local models via Ollama (OpenAI-compatible API); optional Anthropic."
        ),
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        default_response_class=ORJSONResponse,
        lifespan=lifespan,
        debug=settings.app_debug,
    )

    # ── Middleware ────────────────────────────────────────────────────────────
    app.add_middleware(GZipMiddleware, minimum_size=1000)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"] if not settings.is_production else [],
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def api_key_and_audit_middleware(
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        """Optional X-API-Key gate for mutating API routes; append-only audit metadata."""
        s = get_settings()
        rid = request.headers.get("x-request-id") or str(uuid.uuid4())

        is_mutating_api = (
            s.api_key_list
            and request.method == "POST"
            and request.url.path.startswith("/api/v1/")
            and request.url.path not in ("/api/v1/health",)
        )
        if is_mutating_api:
            supplied = request.headers.get("x-api-key")
            if supplied not in s.api_key_list:
                return JSONResponse(
                    status_code=401,
                    content={"detail": "Invalid or missing API key"},
                    headers={"X-Request-ID": rid},
                )

        response = await call_next(request)

        audit_path = s.audit_log_path
        audit_this = (
            audit_path is not None
            and request.method == "POST"
            and request.url.path.startswith("/api/v1/")
        )
        if audit_this:
            assert audit_path is not None
            try:
                entry = {
                    "ts": time.time(),
                    "request_id": rid,
                    "method": request.method,
                    "path": request.url.path,
                    "status_code": response.status_code,
                }
                audit_path.parent.mkdir(parents=True, exist_ok=True)
                with audit_path.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(entry) + "\n")
            except OSError:
                logger.warning("audit_log_path write failed", path=str(audit_path))

        if hasattr(response, "headers"):
            response.headers["X-Request-ID"] = rid
        return response

    # ── Routes ────────────────────────────────────────────────────────────────
    app.include_router(router, prefix="/api/v1")

    # ── Prometheus scrape endpoint ────────────────────────────────────────────
    if settings.metrics_enabled:

        @app.get("/metrics", include_in_schema=False)
        def metrics() -> Response:
            return Response(
                content=generate_latest(),
                media_type=CONTENT_TYPE_LATEST,
            )

    # ── Exception handlers ────────────────────────────────────────────────────
    register_exception_handlers(app)

    return app


# Entrypoint for `uvicorn app.main:app`
app = create_app()

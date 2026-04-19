"""
app/core/dependencies.py
─────────────────────────
FastAPI dependency injection container.

Services are singletons (created once at startup, reused per request).
This avoids recreating HTTP connections and Redis clients on every request.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Annotated

from fastapi import Depends

from app.core.config import Settings, get_settings
from app.services.llm_client import LLMClient
from app.services.pipeline_service import PipelineService
from app.services.quality_service import QualityService
from app.services.rag_service import RAGService
from app.services.summarization_service import SummarizationService
from app.services.triage_service import TriageService
from app.utils.cache import ResponseCache


@lru_cache(maxsize=1)
def get_llm_client(settings: Settings = Depends(get_settings)) -> LLMClient:
    return LLMClient(settings)


@lru_cache(maxsize=1)
def get_cache(settings: Settings = Depends(get_settings)) -> ResponseCache:
    return ResponseCache(settings)


@lru_cache(maxsize=1)
def get_rag_service() -> RAGService:
    """Singleton RAG index (embedding model load once per process)."""
    return RAGService(get_settings())


@lru_cache(maxsize=1)
def get_triage_service(
    llm: LLMClient = Depends(get_llm_client),
    settings: Settings = Depends(get_settings),
    rag: RAGService = Depends(get_rag_service),
) -> TriageService:
    return TriageService(llm, settings, rag)


@lru_cache(maxsize=1)
def get_quality_service(
    llm: LLMClient = Depends(get_llm_client),
    settings: Settings = Depends(get_settings),
    rag: RAGService = Depends(get_rag_service),
) -> QualityService:
    return QualityService(llm, settings, rag)


@lru_cache(maxsize=1)
def get_pipeline_service(
    triage: TriageService = Depends(get_triage_service),
    quality: QualityService = Depends(get_quality_service),
    settings: Settings = Depends(get_settings),
) -> PipelineService:
    return PipelineService(triage, quality, settings)


@lru_cache(maxsize=1)
def get_summarization_service(
    llm: LLMClient = Depends(get_llm_client),
) -> SummarizationService:
    return SummarizationService(llm)


# Convenience type aliases for cleaner router signatures
SettingsDep = Annotated[Settings, Depends(get_settings)]
CacheDep = Annotated[ResponseCache, Depends(get_cache)]
TriageDep = Annotated[TriageService, Depends(get_triage_service)]
QualityDep = Annotated[QualityService, Depends(get_quality_service)]
PipelineDep = Annotated[PipelineService, Depends(get_pipeline_service)]
SummarizeDep = Annotated[SummarizationService, Depends(get_summarization_service)]
RAGDep = Annotated[RAGService, Depends(get_rag_service)]

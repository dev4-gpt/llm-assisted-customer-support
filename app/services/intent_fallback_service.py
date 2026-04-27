"""
Intent/category fallback mapping for off-schema LLM labels.

This keeps API contracts strict while making runtime behavior robust against
near-miss labels such as "refund", "password reset", or "bug".
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.core.config import Settings
from app.core.logging import get_logger
from app.models.domain import Category

logger = get_logger(__name__)

_SYNONYM_TO_CATEGORY: dict[str, str] = {
    # billing
    "refund": "billing",
    "chargeback": "billing",
    "invoice": "billing",
    "payment": "billing",
    "billing_issue": "billing",
    "duplicate_charge": "billing",
    # authentication
    "login": "authentication",
    "password": "authentication",
    "2fa": "authentication",
    "mfa": "authentication",
    "signin": "authentication",
    "sign_in": "authentication",
    "auth": "authentication",
    # technical bug
    "bug": "technical_bug",
    "error": "technical_bug",
    "crash": "technical_bug",
    "broken": "technical_bug",
    "issue": "technical_bug",
    # feature request
    "feature": "feature_request",
    "enhancement": "feature_request",
    "improvement": "feature_request",
    "request": "feature_request",
    # general inquiry
    "question": "general_inquiry",
    "inquiry": "general_inquiry",
    "help": "general_inquiry",
    "support": "general_inquiry",
}

_CATEGORY_PROTOTYPES: dict[str, str] = {
    "billing": "refund billing payment duplicate charge invoice subscription",
    "authentication": "login authentication password reset 2fa mfa account access",
    "technical_bug": "bug crash error broken fails issue technical problem app freezes",
    "feature_request": "feature request enhancement new capability improvement add support",
    "general_inquiry": "general inquiry question help documentation support hours information",
}


@dataclass
class _EmbeddingBackend:
    model: object
    prototypes: dict[str, object]


class IntentFallbackService:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._valid_categories = {c.value for c in Category}
        self._embedding_backend: _EmbeddingBackend | None = None
        self._embedding_enabled = settings.triage_embedding_fallback_enabled
        if self._embedding_enabled:
            self._embedding_backend = self._load_embedding_backend()

    @staticmethod
    def _normalize_text(text: str) -> str:
        text = text.strip().lower()
        text = re.sub(r"[^a-z0-9_ ]+", " ", text)
        return re.sub(r"\s+", " ", text).strip()

    def _load_embedding_backend(self) -> _EmbeddingBackend | None:
        try:
            from sentence_transformers import SentenceTransformer
        except Exception as exc:  # pragma: no cover - environment dependent
            logger.warning(
                "Embedding fallback disabled: sentence-transformers unavailable",
                error=str(exc),
            )
            return None

        try:
            model = SentenceTransformer(self._settings.triage_embedding_model)
            proto_embeds = {
                cat: model.encode(text, normalize_embeddings=True)
                for cat, text in _CATEGORY_PROTOTYPES.items()
            }
            logger.info(
                "Embedding fallback initialised",
                model=self._settings.triage_embedding_model,
            )
            return _EmbeddingBackend(model=model, prototypes=proto_embeds)
        except Exception as exc:  # pragma: no cover - environment dependent
            logger.warning(
                "Embedding fallback disabled: model load failed",
                model=self._settings.triage_embedding_model,
                error=str(exc),
            )
            return None

    def _map_with_synonyms(self, raw_label: str) -> str | None:
        normalized = self._normalize_text(raw_label)
        if not normalized:
            return None
        if normalized in self._valid_categories:
            return normalized
        if normalized in _SYNONYM_TO_CATEGORY:
            return _SYNONYM_TO_CATEGORY[normalized]

        tokens = normalized.split(" ")
        for token in tokens:
            mapped = _SYNONYM_TO_CATEGORY.get(token)
            if mapped:
                return mapped
        return None

    def _map_with_embeddings(self, raw_label: str, ticket_text: str | None = None) -> str | None:
        backend = self._embedding_backend
        if backend is None:
            return None

        query = self._normalize_text(raw_label)
        if ticket_text:
            query = f"{query} {self._normalize_text(ticket_text)[:240]}".strip()
        if not query:
            return None

        try:
            query_embedding = backend.model.encode(query, normalize_embeddings=True)
            best_label = None
            best_score = -1.0
            for category, proto_embedding in backend.prototypes.items():
                score = float((query_embedding * proto_embedding).sum())
                if score > best_score:
                    best_score = score
                    best_label = category

            if best_label is not None and best_score >= self._settings.triage_embedding_min_similarity:
                logger.info(
                    "Embedding fallback mapped label",
                    input_label=raw_label,
                    mapped_label=best_label,
                    score=round(best_score, 4),
                )
                return best_label
            return None
        except Exception as exc:  # pragma: no cover - environment dependent
            logger.warning("Embedding fallback mapping failed", error=str(exc))
            return None

    def map_to_valid_category(self, raw_label: str, ticket_text: str | None = None) -> str | None:
        mapped = self._map_with_synonyms(raw_label)
        if mapped:
            return mapped
        return self._map_with_embeddings(raw_label, ticket_text=ticket_text)

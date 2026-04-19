"""
Policy snippet retrieval: lexical (Jaccard) or embedding cosine similarity.

Lexical mode needs no extra dependencies. Embedding mode requires:
  pip install -e ".[embedding]"
"""

from __future__ import annotations

import json
import re
from typing import Any

from app.core.config import Settings
from app.core.logging import get_logger
from app.models.domain import RAGContextRequest, RAGContextResponse, RAGSnippet

logger = get_logger(__name__)

_TOKEN_RE = re.compile(r"[a-z0-9]+", re.I)


def _tokens(text: str) -> set[str]:
    return {m.group(0).lower() for m in _TOKEN_RE.finditer(text)}


def _lexical_score(query: str, body: str) -> float:
    q, b = _tokens(query), _tokens(body)
    blow = body.lower()
    sub_boost = 0.0
    for t in q:
        if len(t) >= 4 and t in blow:
            sub_boost += 0.12
    sub_boost = min(sub_boost, 0.5)

    if not q or not b:
        return sub_boost

    inter = len(q & b)
    union = len(q | b)
    jaccard = inter / union if union else 0.0
    return min(1.0, jaccard + sub_boost)


class RAGService:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._path = settings.policy_snippets_path
        self._snippets: list[dict[str, str]] = []
        self._load_snippets()

        self._embed_model: Any = None
        self._embed_matrix: Any = None  # np.ndarray (n, d) L2-normalized rows
        self._backend = settings.rag_backend

        if self._backend == "embedding" and self._snippets:
            self._init_embedding_index()

    def _load_snippets(self) -> None:
        path = self._path
        if not path.is_file():
            logger.warning("Policy snippets file not found", path=str(path.resolve()))
            return
        try:
            raw = path.read_text(encoding="utf-8")
            data = json.loads(raw)
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Failed to load policy snippets", path=str(path), error=str(exc))
            return
        if not isinstance(data, list):
            logger.warning("Policy snippets JSON must be a list", path=str(path))
            return
        for row in data:
            if isinstance(row, dict) and {"id", "title", "body"} <= row.keys():
                self._snippets.append(
                    {
                        "id": str(row["id"]),
                        "title": str(row["title"]),
                        "body": str(row["body"]),
                    }
                )

    def _init_embedding_index(self) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "RAG_BACKEND=embedding requires optional dependencies. "
                "Install with: pip install -e '.[embedding]'"
            ) from exc

        model_name = self._settings.rag_embedding_model
        logger.info("Loading embedding model for RAG", model=model_name)
        self._embed_model = SentenceTransformer(model_name)
        texts = [f"{s['title']} {s['body']}" for s in self._snippets]
        self._embed_matrix = self._embed_model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

    def retrieve(self, request: RAGContextRequest, *, top_k: int = 3) -> RAGContextResponse:
        if not self._snippets:
            return RAGContextResponse(snippets=[])

        if self._backend == "embedding" and self._embed_matrix is not None:
            return self._retrieve_embedding(request, top_k=top_k)
        return self._retrieve_lexical(request, top_k=top_k)

    def _retrieve_lexical(self, request: RAGContextRequest, *, top_k: int) -> RAGContextResponse:
        scored: list[tuple[float, dict[str, str]]] = []
        for sn in self._snippets:
            text = f"{sn['title']} {sn['body']}"
            scored.append((_lexical_score(request.query, text), sn))
        scored.sort(key=lambda t: t[0], reverse=True)
        top = scored[:top_k]
        out = [
            RAGSnippet(
                id=sn["id"],
                title=sn["title"],
                body=sn["body"],
                score=round(s, 4),
            )
            for s, sn in top
            if s > 0
        ]
        return RAGContextResponse(snippets=out)

    def _retrieve_embedding(self, request: RAGContextRequest, *, top_k: int) -> RAGContextResponse:
        assert self._embed_model is not None and self._embed_matrix is not None
        import numpy as np

        q_vec = self._embed_model.encode(
            [request.query],
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0]
        mat = np.asarray(self._embed_matrix)
        sims = mat @ q_vec
        order = np.argsort(-sims)[:top_k]
        out: list[RAGSnippet] = []
        for idx in order:
            score = float(sims[idx])
            if score <= 0:
                continue
            sn = self._snippets[int(idx)]
            out.append(
                RAGSnippet(
                    id=sn["id"],
                    title=sn["title"],
                    body=sn["body"],
                    score=round(min(1.0, max(0.0, score)), 4),
                )
            )
        return RAGContextResponse(snippets=out)

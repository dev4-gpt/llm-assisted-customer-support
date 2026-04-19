"""Unit tests for RAGService lexical retrieval."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.core.config import Settings
from app.models.domain import RAGContextRequest
from app.services.rag_service import RAGService


@pytest.fixture()
def snippets_path(tmp_path: Path) -> Path:
    p = tmp_path / "snippets.json"
    p.write_text(
        json.dumps(
            [
                {"id": "a", "title": "Refunds", "body": "Refunds take five business days."},
                {"id": "b", "title": "Passwords", "body": "Reset link expires in one hour."},
            ]
        ),
        encoding="utf-8",
    )
    return p


def test_retrieve_prefers_matching_topic(snippets_path: Path) -> None:
    settings = Settings(
        app_secret_key="test-secret-key-1234",
        policy_snippets_path=snippets_path,
    )
    svc = RAGService(settings)
    res = svc.retrieve(RAGContextRequest(query="how long until my refund arrives"))
    assert len(res.snippets) >= 1
    assert res.snippets[0].id == "a"
    assert res.snippets[0].score > 0

"""Unit tests for SummarizationService (LLM mocked)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from app.core.exceptions import ValidationError
from app.models.domain import DialogRole, DialogTurn, SummarizeRequest
from app.services.summarization_service import SummarizationService


@pytest.fixture()
def mock_llm() -> MagicMock:
    return MagicMock()


@pytest.fixture()
def service(mock_llm: MagicMock) -> SummarizationService:
    return SummarizationService(mock_llm)


def test_summarize_success(service: SummarizationService, mock_llm: MagicMock) -> None:
    mock_llm.complete_json.return_value = {
        "summary": "Customer reported a billing issue; agent opened a case.",
        "key_points": ["Double charge", "Case opened"],
        "confidence": 0.9,
    }
    req = SummarizeRequest(
        turns=[
            DialogTurn(role=DialogRole.CUSTOMER, content="I was charged twice."),
            DialogTurn(role=DialogRole.AGENT, content="I opened ticket #1 for billing."),
        ]
    )
    out = service.summarize(req)
    assert "billing" in out.summary.lower() or "charge" in out.summary.lower()
    assert len(out.key_points) >= 1
    assert out.confidence == pytest.approx(0.9)


def test_summarize_rejects_bad_confidence(
    service: SummarizationService, mock_llm: MagicMock
) -> None:
    mock_llm.complete_json.return_value = {
        "summary": "Too short",
        "key_points": ["a"],
        "confidence": 2.0,
    }
    req = SummarizeRequest(
        turns=[
            DialogTurn(role=DialogRole.CUSTOMER, content="Hello there I need help with my account"),
            DialogTurn(role=DialogRole.AGENT, content="Hi, I can help you with that today"),
        ]
    )
    with pytest.raises(ValidationError, match="confidence"):
        service.summarize(req)

"""Unit tests for Zendesk worker mapping helpers."""

from __future__ import annotations

import json

import pytest

from app.integrations.zendesk_worker import (
    suggested_zendesk_tags,
    ticket_to_triage_body,
)


def test_ticket_to_triage_body_combines_subject_and_description() -> None:
    body = ticket_to_triage_body(
        {
            "subject": "Wrong invoice",
            "description": "The total does not match my receipt from last week.",
        }
    )
    assert "ticket_text" in body
    assert "Wrong invoice" in body["ticket_text"]
    assert "receipt" in body["ticket_text"]


def test_ticket_to_triage_body_rejects_too_short() -> None:
    with pytest.raises(ValueError, match="at least 10"):
        ticket_to_triage_body({"subject": "Hi", "description": ""})


def test_suggested_zendesk_tags() -> None:
    tags = suggested_zendesk_tags(
        {"priority": "high", "category": "billing", "routed_team": "billing_specialists"}
    )
    assert "llm_priority_high" in tags
    assert "llm_category_billing" in tags


def test_fixture_file_shape() -> None:
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    fx = root / "data" / "fixtures" / "zendesk_ticket.json"
    data = json.loads(fx.read_text(encoding="utf-8"))
    body = ticket_to_triage_body(data["ticket"])
    assert len(body["ticket_text"]) >= 10


def test_post_triage_calls_api(monkeypatch) -> None:
    from unittest.mock import MagicMock

    from app.integrations import zendesk_worker as zw

    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json.return_value = {"priority": "high", "category": "billing", "routed_team": "x"}

    mock_client = MagicMock()
    mock_client.__enter__.return_value = mock_client
    mock_client.__exit__.return_value = None
    mock_client.post.return_value = mock_resp

    monkeypatch.setattr(zw.httpx, "Client", lambda **kwargs: mock_client)

    out = zw.post_triage(
        "http://localhost:9999", {"ticket_text": "hello world billing issue"}, api_key="k"
    )
    assert out["priority"] == "high"
    mock_client.post.assert_called_once()
    _args, kwargs = mock_client.post.call_args
    assert kwargs["headers"]["X-API-Key"] == "k"

"""
Map Zendesk ticket JSON → `POST /api/v1/triage` (and optional tag payload).

Fixture mode (CI / demos): load a saved JSON file with `subject` + `description`.
Live mode: `GET /api/v2/tickets/{id}.json` with HTTP Basic auth (email/token).

Environment (live only):
  ZENDESK_SUBDOMAIN  — e.g. "mycompany" for mycompany.zendesk.com
  ZENDESK_EMAIL      — agent email
  ZENDESK_API_TOKEN  — API token (Settings token, not password)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import httpx


def ticket_to_triage_body(ticket: dict[str, Any]) -> dict[str, str]:
    """Build JSON body for POST /api/v1/triage from a Zendesk-like ticket dict."""
    subject = str(ticket.get("subject") or "").strip()
    description = str(ticket.get("description") or ticket.get("body") or "").strip()
    parts = [p for p in (subject, description) if p]
    ticket_text = "\n\n".join(parts) if parts else description or subject
    ticket_text = ticket_text.strip()
    if len(ticket_text) < 10:
        raise ValueError(
            "Combined subject+description must be at least 10 characters for the triage API."
        )
    return {"ticket_text": ticket_text[:10_000]}


def suggested_zendesk_tags(triage_response: dict[str, Any]) -> list[str]:
    """Derive tag strings from triage JSON (for PUT ticket update)."""
    return [
        f"llm_priority_{triage_response.get('priority', 'unknown')}",
        f"llm_category_{triage_response.get('category', 'unknown')}",
        f"llm_team_{triage_response.get('routed_team', 'unknown')}",
    ]


def load_fixture(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if "ticket" in data:
        return dict(data["ticket"])
    return dict(data)


def fetch_zendesk_ticket(ticket_id: str) -> dict[str, Any]:
    sub = os.environ.get("ZENDESK_SUBDOMAIN", "").strip()
    email = os.environ.get("ZENDESK_EMAIL", "").strip()
    token = os.environ.get("ZENDESK_API_TOKEN", "").strip()
    if not (sub and email and token):
        raise RuntimeError(
            "Live mode requires ZENDESK_SUBDOMAIN, ZENDESK_EMAIL, "
            "ZENDESK_API_TOKEN in the environment."
        )
    url = f"https://{sub}.zendesk.com/api/v2/tickets/{ticket_id}.json"
    auth = (f"{email}/token", token)
    with httpx.Client(timeout=60.0) as client:
        r = client.get(url, auth=auth)
        r.raise_for_status()
        payload = r.json()
    ticket = payload.get("ticket")
    if not isinstance(ticket, dict):
        raise RuntimeError("Unexpected Zendesk response: missing ticket object")
    return ticket


def post_triage(api_base: str, body: dict[str, str], *, api_key: str | None) -> dict[str, Any]:
    base = api_base.rstrip("/")
    url = f"{base}/api/v1/triage"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["X-API-Key"] = api_key
    with httpx.Client(timeout=120.0) as client:
        r = client.post(url, json=body, headers=headers)
        r.raise_for_status()
        return dict(r.json())


def main() -> None:
    desc = "Zendesk → support triage API worker (stub)."
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument(
        "--api-base",
        default="http://127.0.0.1:8000",
        help="Root URL of running API",
    )
    parser.add_argument("--fixture", type=Path, help="Path to JSON ticket fixture")
    parser.add_argument("--ticket-id", help="Zendesk numeric ticket id (live fetch)")
    parser.add_argument(
        "--api-key",
        default=os.environ.get("TRIAGE_API_KEY"),
        help="X-API-Key if server requires it",
    )
    args = parser.parse_args()

    if args.fixture:
        ticket = load_fixture(args.fixture)
    elif args.ticket_id:
        ticket = fetch_zendesk_ticket(args.ticket_id)
    else:
        parser.error("Provide --fixture or --ticket-id")

    triage_body = ticket_to_triage_body(ticket)
    result = post_triage(args.api_base, triage_body, api_key=args.api_key)
    tags = suggested_zendesk_tags(result)

    out = {"triage": result, "suggested_zendesk_tags": tags}
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    try:
        main()
    except (httpx.HTTPError, OSError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)

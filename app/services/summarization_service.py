"""
Summarization of multi-turn support threads via structured LLM JSON.
"""

from __future__ import annotations

import json
from typing import Any

from app.core.exceptions import LLMParseError, ValidationError
from app.core.logging import get_logger
from app.models.domain import SummarizeRequest, SummarizeResult
from app.services.llm_client import LLMClient

logger = get_logger(__name__)

_SUMMARY_PROMPT = """\
You summarise customer support threads for agents. Return ONLY JSON (no markdown) with keys:
  "summary": string (2-4 sentences),
  "key_points": array of 2-6 short strings (issue, what was tried, current status),
  "confidence": float 0-1 (how well the thread supports the summary)

THREAD (role-prefixed lines):
{thread_block}
"""


class SummarizationService:
    def __init__(self, llm_client: LLMClient) -> None:
        self._llm = llm_client

    def summarize(self, request: SummarizeRequest) -> SummarizeResult:
        lines: list[str] = []
        for turn in request.turns:
            prefix = turn.role.value.upper()
            lines.append(f"[{prefix}] {turn.content}")
        thread_block = "\n".join(lines)

        prompt = _SUMMARY_PROMPT.format(thread_block=thread_block)
        raw = self._llm.complete_json(prompt, schema_hint="SummarizeResult")
        return self._to_result(raw)

    @staticmethod
    def _to_result(data: dict[str, Any]) -> SummarizeResult:
        required = {"summary", "key_points", "confidence"}
        missing = required - data.keys()
        if missing:
            raise LLMParseError(json.dumps(data) + f" | Missing fields: {missing}")

        summary = data["summary"]
        if not isinstance(summary, str) or not summary.strip():
            raise ValidationError("summary must be a non-empty string")

        key_points = data["key_points"]
        if not isinstance(key_points, list) or not all(isinstance(x, str) for x in key_points):
            raise ValidationError("key_points must be a list of strings")

        conf = data["confidence"]
        if not isinstance(conf, (int, float)):
            raise ValidationError("confidence must be a number")
        c = float(conf)
        if not 0.0 <= c <= 1.0:
            raise ValidationError("confidence must be between 0 and 1")

        return SummarizeResult(
            summary=summary.strip(),
            key_points=[kp.strip() for kp in key_points if kp.strip()],
            confidence=round(c, 4),
        )

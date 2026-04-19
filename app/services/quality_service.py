"""
app/services/quality_service.py
────────────────────────────────
LLM-powered agent response quality evaluation.

Each quality dimension maps to an explicit rubric in the prompt so the model
produces consistent, auditable scores rather than vibes-based judgments.
"""

from __future__ import annotations

import json
from typing import Any

from app.core.config import Settings
from app.core.exceptions import LLMParseError, ValidationError
from app.core.logging import get_logger
from app.models.domain import (
    QualityChecks,
    QualityRequest,
    QualityResult,
    RAGContextRequest,
)
from app.services.llm_client import LLMClient
from app.services.rag_service import RAGService

logger = get_logger(__name__)

_QUALITY_PROMPT = """\
You are an expert customer support quality analyst. Evaluate the AGENT RESPONSE against the
CUSTOMER TICKET using the rubric below. Return ONLY a JSON object with this exact schema:

{{
  "score": <float 0.0-1.0>,
  "checks": {{
    "empathetic_tone": <true|false>,
    "actionable_next_step": <true|false>,
    "policy_safety": <true|false>,
    "resolved_or_escalated": <true|false>
  }},
  "coaching_feedback": "<2-4 sentences of specific, actionable feedback>",
  "flagged_phrases": ["<phrase1>", "<phrase2>"]
}}

RUBRIC:
  empathetic_tone         → Response acknowledges the customer's frustration/situation
                            with genuine empathy (not boilerplate "I understand").
  actionable_next_step    → Response provides at least one clear, specific next step
                            the customer or agent will take (not vague "we'll look into it").
  policy_safety           → Response does NOT: promise refunds without authorisation,
                            share confidential data, make legal admissions, or violate SLA.
  resolved_or_escalated   → Response either confirms resolution OR explicitly states
                            it has been escalated with a timeline/ticket number.

SCORING:
  score = (sum of true checks) / 4
  Deduct 0.1 from score for each flagged unsafe/unprofessional phrase (floor 0.0).

coaching_feedback: Be specific. Reference actual phrases from the response.
flagged_phrases: List verbatim phrases that are problematic (empty list [] if none).

{policy_section}
CUSTOMER TICKET:
\"\"\"
{ticket_text}
\"\"\"

AGENT RESPONSE:
\"\"\"
{agent_response}
\"\"\"
"""


class QualityService:
    def __init__(
        self,
        llm_client: LLMClient,
        settings: Settings,
        rag_service: RAGService,
    ) -> None:
        self._llm = llm_client
        self._settings = settings
        self._rag = rag_service

    def _build_policy_section(self, request: QualityRequest) -> str:
        if not request.include_policy_context:
            return ""
        k = self._settings.quality_policy_context_top_k
        if k <= 0:
            return ""
        query = f"{request.ticket_text[:1500]} {request.agent_response[:1500]}"
        ctx = self._rag.retrieve(RAGContextRequest(query=query), top_k=k)
        if not ctx.snippets:
            return ""
        lines = [f"[{s.id}] {s.title}: {s.body}" for s in ctx.snippets]
        return (
            "RELEVANT POLICY SNIPPETS (the agent response must not contradict these when applicable):\n"
            + "\n".join(lines)
            + "\n\n"
        )

    def evaluate(self, request: QualityRequest) -> QualityResult:
        """Evaluate agent response quality against the original ticket."""
        logger.info(
            "Starting quality evaluation",
            ticket_length=len(request.ticket_text),
            response_length=len(request.agent_response),
        )

        policy_section = self._build_policy_section(request)
        prompt = _QUALITY_PROMPT.format(
            ticket_text=request.ticket_text,
            agent_response=request.agent_response,
            policy_section=policy_section,
        )
        raw = self._llm.complete_json(prompt, schema_hint="QualityResult")
        quality_data = self._validate_quality_response(raw)

        score: float = quality_data["score"]
        passed = score >= self._settings.quality_pass_threshold

        result = QualityResult(
            score=score,
            passed=passed,
            checks=QualityChecks(**quality_data["checks"]),
            coaching_feedback=quality_data["coaching_feedback"],
            flagged_phrases=quality_data.get("flagged_phrases", []),
        )

        logger.info(
            "Quality evaluation complete",
            score=result.score,
            passed=result.passed,
            flagged_count=len(result.flagged_phrases),
        )
        return result

    # ── Private helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _validate_quality_response(data: dict[str, Any]) -> dict[str, Any]:
        required = {"score", "checks", "coaching_feedback"}
        missing = required - data.keys()
        if missing:
            raise LLMParseError(json.dumps(data) + f" | Missing fields: {missing}")

        checks = data.get("checks", {})
        required_checks = {
            "empathetic_tone",
            "actionable_next_step",
            "policy_safety",
            "resolved_or_escalated",
        }
        missing_checks = required_checks - checks.keys()
        if missing_checks:
            raise ValidationError(f"Missing quality check fields: {missing_checks}")

        if not isinstance(data["score"], (int, float)):
            raise ValidationError("score must be a number")

        score = float(data["score"])
        if not 0.0 <= score <= 1.0:
            raise ValidationError(f"score {score} out of range [0.0, 1.0]")

        # Normalise booleans (LLMs sometimes return strings)
        for key in required_checks:
            val = checks[key]
            if isinstance(val, str):
                checks[key] = val.lower() in ("true", "yes", "1")

        data["score"] = score
        return data

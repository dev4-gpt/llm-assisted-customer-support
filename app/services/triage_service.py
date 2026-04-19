"""
app/services/triage_service.py
──────────────────────────────
Orchestrates ticket triage via the LLM.

Design decisions:
  • Prompt is self-contained and schema-driven so swapping models is trivial.
  • Routing logic lives here (not in the LLM) for determinism and auditability.
  • Each method has a single responsibility.
"""

from __future__ import annotations

import json
from typing import Any

from app.core.config import Settings
from app.core.exceptions import LLMParseError, ValidationError
from app.core.logging import get_logger
from app.models.domain import (
    Category,
    IntentScore,
    Priority,
    RAGContextRequest,
    RoutedTeam,
    TriageRequest,
    TriageResult,
)
from app.services.llm_client import LLMClient
from app.services.rag_service import RAGService

logger = get_logger(__name__)

# Routing matrix: (priority, category) → team
# Falls back to ESCALATIONS for negative sentiment override (applied in service).
_ROUTING_MATRIX: dict[tuple[Priority, Category], RoutedTeam] = {
    # Critical → always critical response desk
    **{(Priority.CRITICAL, cat): RoutedTeam.CRITICAL_RESPONSE for cat in Category},
    # High
    (Priority.HIGH, Category.BILLING): RoutedTeam.BILLING_SPECIALISTS,
    (Priority.HIGH, Category.AUTHENTICATION): RoutedTeam.AUTH_SECURITY,
    (Priority.HIGH, Category.TECHNICAL_BUG): RoutedTeam.TIER2_ENGINEERING,
    (Priority.HIGH, Category.FEATURE_REQUEST): RoutedTeam.PRODUCT_TEAM,
    (Priority.HIGH, Category.GENERAL_INQUIRY): RoutedTeam.GENERAL_SUPPORT,
    # Medium
    (Priority.MEDIUM, Category.BILLING): RoutedTeam.BILLING_SPECIALISTS,
    (Priority.MEDIUM, Category.AUTHENTICATION): RoutedTeam.AUTH_SECURITY,
    (Priority.MEDIUM, Category.TECHNICAL_BUG): RoutedTeam.TIER2_ENGINEERING,
    (Priority.MEDIUM, Category.FEATURE_REQUEST): RoutedTeam.PRODUCT_TEAM,
    (Priority.MEDIUM, Category.GENERAL_INQUIRY): RoutedTeam.GENERAL_SUPPORT,
    # Low
    (Priority.LOW, Category.BILLING): RoutedTeam.GENERAL_SUPPORT,
    (Priority.LOW, Category.AUTHENTICATION): RoutedTeam.GENERAL_SUPPORT,
    (Priority.LOW, Category.TECHNICAL_BUG): RoutedTeam.GENERAL_SUPPORT,
    (Priority.LOW, Category.FEATURE_REQUEST): RoutedTeam.PRODUCT_TEAM,
    (Priority.LOW, Category.GENERAL_INQUIRY): RoutedTeam.GENERAL_SUPPORT,
}

_TRIAGE_PROMPT = """\
{prefix}You are a support triage expert. Analyse the customer ticket below and return ONLY a JSON
object matching this exact schema (no extra keys):

{{
  "priority": "<critical|high|medium|low>",
  "category": "<billing|authentication|technical_bug|feature_request|general_inquiry>",
  "intents": [
    {{ "label": "<same enum as category values>", "score": <float 0.0-1.0> }}
  ],
  "sentiment_score": <float -1.0 to 1.0>,
  "rationale": "<1-3 sentence explanation>",
  "confidence": <float 0.0 to 1.0>
}}

Multi-intent rules:
  • Include EVERY distinct issue mentioned (e.g. refund + address change → two intent objects).
  • Labels must be exactly: billing | authentication | technical_bug | feature_request | general_inquiry
  • Scores need not sum to 1.0; higher means stronger evidence for that intent.
  • "category" must be the single primary routing label (highest-severity business category if multiple).

Priority rules:
  critical  = data loss, service outage, security breach, double-charge, legal threat
  high      = core feature broken, payment issue, locked out of account
  medium    = degraded functionality, billing question, slow response
  low       = feature requests, general questions, compliments

Sentiment rules:
  -1.0 = extremely frustrated / threatening churn
   0.0 = neutral
  +1.0 = very happy / complimentary

TICKET:
\"\"\"
{ticket_text}
\"\"\"
"""


class TriageService:
    def __init__(
        self,
        llm_client: LLMClient,
        settings: Settings,
        rag_service: RAGService,
    ) -> None:
        self._llm = llm_client
        self._settings = settings
        self._rag = rag_service
        self._baseline_pipe: Any = None
        self._transformer_predictor: Any = None

    def _policy_prefix(self, request: TriageRequest) -> str:
        if not request.include_policy_context:
            return ""
        k = self._settings.triage_policy_context_top_k
        if k <= 0:
            return ""
        q = request.ticket_text[:2000]
        ctx = self._rag.retrieve(RAGContextRequest(query=q), top_k=k)
        if not ctx.snippets:
            return ""
        lines = [f"- [{s.id}] {s.title}: {s.body[:600]}" for s in ctx.snippets]
        header = "POLICY CONTEXT (use when disambiguating; ticket text is authoritative):\n"
        return header + "\n".join(lines) + "\n\n"

    def _classifier_hint(self, ticket_text: str) -> str:
        if not self._settings.triage_hybrid_enabled:
            return ""
        path = self._settings.triage_baseline_model_path
        if path is None or not path.is_file():
            return ""
        if self._baseline_pipe is None:
            import joblib

            self._baseline_pipe = joblib.load(path)
        suggested = str(self._baseline_pipe.predict([ticket_text])[0])
        msg1 = f'A fast baseline classifier suggests primary category "{suggested}". '
        msg2 = "Return the full JSON schema; override if the ticket clearly differs.\n\n"
        return msg1 + msg2

    def _transformer_hint(self, ticket_text: str) -> str:
        if not self._settings.triage_transformer_enabled:
            return ""
        model_dir = self._settings.triage_transformer_model_dir
        if model_dir is None or not model_dir.is_dir():
            return ""
        if self._transformer_predictor is None:
            from app.services.triage_transformer_predict import load_triage_transformer

            self._transformer_predictor = load_triage_transformer(model_dir)
            if self._transformer_predictor is None:
                return ""
        try:
            suggested = self._transformer_predictor.predict_category(ticket_text)
        except Exception:
            logger.exception("Transformer category prediction failed")
            return ""
        msg1 = (
            f'A fine-tuned encoder model (BERT/RoBERTa-style) suggests primary category '
            f'"{suggested}". '
        )
        msg2 = "Return the full JSON schema; override if the ticket clearly differs.\n\n"
        return msg1 + msg2

    def triage(self, request: TriageRequest) -> TriageResult:
        """Run full triage pipeline for a single ticket."""
        logger.info("Starting triage", ticket_length=len(request.ticket_text))

        prefix = (
            self._policy_prefix(request)
            + self._transformer_hint(request.ticket_text)
            + self._classifier_hint(request.ticket_text)
        )
        prompt = _TRIAGE_PROMPT.format(prefix=prefix, ticket_text=request.ticket_text)
        raw = self._llm.complete_json(prompt, schema_hint="TriageResult")

        triage_data = self._validate_triage_response(raw)
        priority = Priority(triage_data["priority"])
        category = Category(triage_data["category"])
        sentiment_score: float = triage_data["sentiment_score"]
        intents = self._normalize_intents(
            triage_data["intents"],
            primary=category,
            confidence=float(triage_data["confidence"]),
        )

        routed_team = self._resolve_routing(priority, category, sentiment_score)

        result = TriageResult(
            priority=priority,
            category=category,
            intents=intents,
            sentiment_score=sentiment_score,
            routed_team=routed_team,
            rationale=triage_data["rationale"],
            confidence=triage_data["confidence"],
        )

        logger.info(
            "Triage complete",
            priority=result.priority,
            category=result.category,
            sentiment=result.sentiment_score,
            team=result.routed_team,
        )
        return result

    # ── Private helpers ───────────────────────────────────────────────────────

    def _resolve_routing(
        self, priority: Priority, category: Category, sentiment: float
    ) -> RoutedTeam:
        """Apply routing matrix with sentiment-based escalation override."""
        if sentiment < self._settings.sentiment_escalation_cutoff and priority in (
            Priority.HIGH,
            Priority.CRITICAL,
        ):
            return RoutedTeam.ESCALATIONS

        return _ROUTING_MATRIX.get((priority, category), RoutedTeam.GENERAL_SUPPORT)

    @staticmethod
    def _normalize_intents(
        raw_intents: list[dict[str, Any]],
        *,
        primary: Category,
        confidence: float,
    ) -> list[IntentScore]:
        valid_categories = {c.value for c in Category}
        scored: list[IntentScore] = []
        for item in raw_intents:
            label = item.get("label")
            score = item.get("score")
            if not isinstance(label, str) or label not in valid_categories:
                raise ValidationError(
                    f"Invalid intent label '{label}'. Expected one of: {valid_categories}"
                )
            if not isinstance(score, (int, float)):
                raise ValidationError("Each intent score must be a number")
            s = float(score)
            if not 0.0 <= s <= 1.0:
                raise ValidationError("Intent scores must be between 0 and 1")
            scored.append(IntentScore(label=label, score=round(s, 4)))

        labels = {i.label for i in scored}
        if primary.value not in labels:
            scored.append(IntentScore(label=primary.value, score=round(confidence, 4)))

        scored.sort(key=lambda x: x.score, reverse=True)
        return scored

    @staticmethod
    def _validate_triage_response(data: dict[str, Any]) -> dict[str, Any]:
        required = {
            "priority",
            "category",
            "sentiment_score",
            "rationale",
            "confidence",
        }
        missing = required - data.keys()
        if missing:
            raise LLMParseError(json.dumps(data) + f" | Missing fields: {missing}")

        valid_priorities = {p.value for p in Priority}
        if data["priority"] not in valid_priorities:
            raise ValidationError(
                f"LLM returned invalid priority '{data['priority']}'. "
                f"Expected one of: {valid_priorities}"
            )

        valid_categories = {c.value for c in Category}
        if data["category"] not in valid_categories:
            raise ValidationError(
                f"LLM returned invalid category '{data['category']}'. "
                f"Expected one of: {valid_categories}"
            )

        if not isinstance(data["sentiment_score"], (int, float)):
            raise ValidationError("sentiment_score must be a number")

        if not isinstance(data["confidence"], (int, float)):
            raise ValidationError("confidence must be a number")

        raw_intents = data.get("intents")
        if raw_intents is None:
            raw_intents = [
                {
                    "label": data["category"],
                    "score": float(data["confidence"]),
                }
            ]
        elif not isinstance(raw_intents, list) or not raw_intents:
            raise ValidationError("intents must be a non-empty list when provided")

        for item in raw_intents:
            if not isinstance(item, dict):
                raise ValidationError("Each intent must be an object with label and score")

        data["intents"] = raw_intents
        return data

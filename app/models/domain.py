"""
app/models/domain.py
────────────────────
Pydantic v2 domain models: enums, value objects, request/response contracts.
These are the single source of truth for data shapes across the entire system.
"""

from __future__ import annotations

from enum import Enum
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# ── Enumerations ─────────────────────────────────────────────────────────────


class Priority(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class Category(str, Enum):
    BILLING = "billing"
    AUTHENTICATION = "authentication"
    TECHNICAL_BUG = "technical_bug"
    FEATURE_REQUEST = "feature_request"
    GENERAL_INQUIRY = "general_inquiry"


class RoutedTeam(str, Enum):
    CRITICAL_RESPONSE = "critical_response"
    BILLING_SPECIALISTS = "billing_specialists"
    AUTH_SECURITY = "auth_security"
    TIER2_ENGINEERING = "tier2_engineering"
    PRODUCT_TEAM = "product_team"
    GENERAL_SUPPORT = "general_support"
    ESCALATIONS = "escalations"


class DialogRole(str, Enum):
    CUSTOMER = "customer"
    AGENT = "agent"
    BRAND = "brand"
    SYSTEM = "system"


# ── Shared constraints ────────────────────────────────────────────────────────

TicketText = Annotated[str, Field(min_length=10, max_length=10_000)]
AgentResponse = Annotated[str, Field(min_length=10, max_length=10_000)]
DialogContent = Annotated[str, Field(min_length=1, max_length=8_000)]


# ── Request models ────────────────────────────────────────────────────────────


class TriageRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, frozen=True)

    ticket_text: TicketText = Field(
        ...,
        description="Raw customer ticket text to triage.",
        examples=["My payment failed and I've been charged twice. This is urgent!"],
    )
    include_policy_context: bool = Field(
        default=True,
        description="When enabled and triage_policy_context_top_k > 0, prepend retrieved policy snippets.",
    )


class QualityRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, frozen=True)

    ticket_text: TicketText = Field(..., description="Original customer ticket.")
    agent_response: AgentResponse = Field(..., description="Agent draft to evaluate.")
    include_policy_context: bool = Field(
        default=True,
        description="When enabled and quality_policy_context_top_k > 0, inject policy snippets into rubric.",
    )


class PipelineRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, frozen=True)

    ticket_text: TicketText
    agent_response: AgentResponse


class DialogTurn(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, frozen=True)

    role: DialogRole
    content: DialogContent


class SummarizeRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, frozen=True)

    turns: list[DialogTurn] = Field(
        ...,
        min_length=2,
        max_length=50,
        description="Ordered conversation turns (customer / agent / brand).",
    )


class RAGContextRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, frozen=True)

    query: Annotated[str, Field(min_length=3, max_length=2_000)]


# ── Response models ───────────────────────────────────────────────────────────


class IntentScore(BaseModel):
    """Multi-intent support: label aligns with Category enum strings for routing consistency."""

    model_config = ConfigDict(frozen=True)

    label: str = Field(
        ...,
        description="Intent label (billing, authentication, technical_bug, feature_request, general_inquiry).",
    )
    score: float = Field(
        ge=0.0, le=1.0, description="Relative strength of this intent for the ticket."
    )

    @field_validator("score")
    @classmethod
    def round_score(cls, v: float) -> float:
        return round(v, 4)


class TriageResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    priority: Priority
    category: Category
    intents: list[IntentScore] = Field(
        ...,
        min_length=1,
        description="All detected intents with scores; primary routing uses `category`.",
    )
    sentiment_score: float = Field(ge=-1.0, le=1.0)
    routed_team: RoutedTeam
    rationale: str = Field(description="LLM-generated triage rationale (1-3 sentences).")
    confidence: float = Field(ge=0.0, le=1.0, description="Model confidence score.")

    @field_validator("sentiment_score", "confidence")
    @classmethod
    def round_floats(cls, v: float) -> float:
        return round(v, 4)

    @model_validator(mode="after")
    def primary_category_in_intents(self) -> TriageResult:
        labels = {i.label for i in self.intents}
        if self.category.value not in labels:
            raise ValueError(
                f"Primary category '{self.category.value}' must appear in intents list"
            )
        return self


class QualityChecks(BaseModel):
    model_config = ConfigDict(frozen=True)

    empathetic_tone: bool
    actionable_next_step: bool
    policy_safety: bool
    resolved_or_escalated: bool


class QualityResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    score: float = Field(ge=0.0, le=1.0)
    passed: bool
    checks: QualityChecks
    coaching_feedback: str
    flagged_phrases: list[str] = Field(
        default_factory=list,
        description="Specific phrases the LLM flagged as problematic.",
    )

    @field_validator("score")
    @classmethod
    def round_score(cls, v: float) -> float:
        return round(v, 4)


class PipelineResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    triage: TriageResult
    quality: QualityResult
    recommended_sla_minutes: int
    workflow_passed: bool = Field(
        description="True iff quality passed and priority is not critical-unresolved."
    )


class SummarizeResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    summary: str = Field(description="Concise overview of the thread for agents.")
    key_points: list[str] = Field(
        default_factory=list,
        description="Bullet-style facts: issue, actions taken, open status.",
    )
    confidence: float = Field(ge=0.0, le=1.0)

    @field_validator("confidence")
    @classmethod
    def round_conf(cls, v: float) -> float:
        return round(v, 4)


class RAGSnippet(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    title: str
    body: str
    score: float = Field(ge=0.0, le=1.0, description="Lexical relevance score (higher is better).")


class RAGContextResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    snippets: list[RAGSnippet]


# ── Health ────────────────────────────────────────────────────────────────────


class HealthStatus(str, Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


class HealthResponse(BaseModel):
    status: HealthStatus
    version: str
    checks: dict[str, str]

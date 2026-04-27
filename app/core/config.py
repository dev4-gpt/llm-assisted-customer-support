"""
app/core/config.py
──────────────────
Centralised, validated configuration via pydantic-settings.
All env-var access goes through Settings(); never os.environ directly.
"""

from __future__ import annotations

from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Environment(str, Enum):
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── LLM provider ─────────────────────────────────────────────────────────
    # Default: hosted OpenAI-compatible API (OpenRouter/NVIDIA/etc at /v1/chat/completions).
    # Set LLM_PROVIDER=anthropic only if using Anthropic cloud (requires pip install anthropic).
    llm_profile: Literal["manual", "ollama", "openrouter", "nvidia"] = Field(
        default="manual",
        description=(
            "Convenience profile switch. manual=use LLM_PROVIDER directly; "
            "ollama/openrouter/nvidia auto-fill provider/base/model/key fields."
        ),
    )
    llm_provider: Literal["ollama", "openai_compatible", "anthropic"] = Field(
        default="openai_compatible",
        description="ollama | openai_compatible (same HTTP API) | anthropic",
    )
    llm_model: str = Field(
        default="google/gemini-2.0-flash-exp:free",
        validation_alias=AliasChoices("LLM_MODEL", "ANTHROPIC_MODEL"),
        description="Model name on the provider (e.g. OpenRouter/NVIDIA model id)",
    )
    llm_max_tokens: int = Field(2048, ge=256, le=8192)
    llm_timeout_seconds: float = Field(30.0, gt=0)
    llm_temperature: float = Field(0.2, ge=0.0, le=2.0)

    # OpenAI-compatible servers: OpenRouter, NVIDIA NIM, Ollama, vLLM, LM Studio, etc.
    openai_compatible_base_url: str = Field(
        default="https://openrouter.ai/api/v1",
        description="Base URL with /v1 suffix for OpenAI-compatible chat completions",
    )
    openai_compatible_api_key: str = Field(
        default="replace-with-real-api-key",
        description="Bearer token for the OpenAI-compatible provider",
    )

    # Optional profile-specific key/model slots (for llm_profile switching)
    ollama_model: str = Field(
        default="qwen2.5:7b-instruct",
        description="Model tag to use when LLM_PROFILE=ollama",
    )
    openrouter_api_key: str | None = Field(
        default=None,
        description="OpenRouter key used when LLM_PROFILE=openrouter",
    )
    openrouter_model: str = Field(
        default="google/gemini-2.0-flash-exp:free",
        description="Model id used when LLM_PROFILE=openrouter",
    )
    nvidia_api_key: str | None = Field(
        default=None,
        description="NVIDIA API key used when LLM_PROFILE=nvidia",
    )
    nvidia_model: str = Field(
        default="meta/llama-3.1-8b-instruct",
        description="Model id used when LLM_PROFILE=nvidia",
    )

    # Anthropic cloud (optional)
    anthropic_api_key: str | None = Field(
        default=None,
        description="Required only when LLM_PROVIDER=anthropic",
    )

    # ── App ──────────────────────────────────────────────────────────────────
    app_env: Environment = Environment.DEVELOPMENT
    app_debug: bool = False
    app_log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    app_secret_key: str = Field("change-me-in-production", min_length=16)

    # ── API ──────────────────────────────────────────────────────────────────
    api_host: str = "0.0.0.0"
    api_port: int = Field(8000, ge=1, le=65535)
    api_workers: int = Field(4, ge=1, le=64)
    api_rate_limit_per_minute: int = Field(60, ge=1)

    # ── Redis ────────────────────────────────────────────────────────────────
    redis_url: str = "redis://localhost:6379/0"
    redis_ttl_seconds: int = Field(300, ge=10)

    # ── Quality & Sentiment ──────────────────────────────────────────────────
    quality_pass_threshold: float = Field(0.70, ge=0.0, le=1.0)
    sentiment_escalation_cutoff: float = Field(-0.6, ge=-1.0, le=0.0)

    # ── SLA (minutes) ────────────────────────────────────────────────────────
    sla_critical_minutes: int = Field(15, ge=1)
    sla_high_minutes: int = Field(60, ge=1)
    sla_medium_minutes: int = Field(240, ge=1)
    sla_low_minutes: int = Field(1440, ge=1)

    # ── RAG (local policy snippets) ─────────────────────────────────────────
    policy_snippets_path: Path = Field(
        default=Path("data/policy_snippets.json"),
        description="JSON list of {id, title, body} for lexical RAG retrieval.",
    )
    rag_backend: Literal["lexical", "embedding"] = Field(
        default="lexical",
        description=(
            "lexical: token Jaccard; embedding: sentence-transformers cosine "
            "(requires [embedding] extra)."
        ),
    )
    rag_embedding_model: str = Field(
        default="all-MiniLM-L6-v2",
        description="sentence-transformers model name when RAG_BACKEND=embedding",
    )

    # ── Policy grounding (quality / triage prompts) ─────────────────────────
    quality_policy_context_top_k: int = Field(
        0,
        ge=0,
        le=10,
        description="Inject top-k policy snippets into quality rubric (0 = disabled).",
    )
    triage_policy_context_top_k: int = Field(
        0,
        ge=0,
        le=10,
        description="Prepend top-k policy snippets to triage prompt (0 = disabled).",
    )

    # ── Hybrid triage (baseline classifier hint) ─────────────────────────────
    triage_hybrid_enabled: bool = Field(
        False,
        description=(
            "If true and triage_baseline_model_path is set, inject TF-IDF+LR category hint."
        ),
    )
    triage_baseline_model_path: Path | None = Field(
        default=None,
        description="joblib pipeline from scripts/train_encoder_classifier.py",
    )

    # ── Optional fine-tuned encoder hint (BERT / RoBERTa family) ───────────────
    triage_transformer_enabled: bool = Field(
        False,
        description=(
            "If true and triage_transformer_model_dir is set, inject HF encoder "
            "category hint (requires pip install -e \".[transformer]\")."
        ),
    )
    triage_transformer_model_dir: Path | None = Field(
        default=None,
        description="Directory from scripts/train_triage_transformer.py (config.json + weights).",
    )

    # ── Intent/category fallback hardening ───────────────────────────────────
    triage_embedding_fallback_enabled: bool = Field(
        True,
        description=(
            "If true, invalid LLM intent/category labels are mapped to allowed taxonomy "
            "using synonym rules and optional embedding similarity."
        ),
    )
    triage_embedding_model: str = Field(
        default="all-MiniLM-L6-v2",
        description="Embedding model for triage label fallback (requires [embedding] extra).",
    )
    triage_embedding_min_similarity: float = Field(
        0.22,
        ge=0.0,
        le=1.0,
        description="Minimum cosine similarity for embedding-based label fallback mapping.",
    )

    # ── LLM observability ────────────────────────────────────────────────────
    llm_prompt_version: str = Field(
        default="v1",
        min_length=1,
        max_length=64,
        description="Label for Prometheus LLM metrics (prompt / rubric version).",
    )

    # ── API hardening ────────────────────────────────────────────────────────
    api_keys: str | None = Field(
        default=None,
        description=(
            "Comma-separated keys; when set, POST /api/v1/* require X-API-Key (except /health)."
        ),
    )
    audit_log_path: Path | None = Field(
        default=None,
        description="Append-only JSONL path for API audit metadata (no request bodies).",
    )

    # ── Observability ────────────────────────────────────────────────────────
    otel_exporter_otlp_endpoint: str = "http://localhost:4317"
    otel_service_name: str = "support-triage"
    metrics_enabled: bool = True

    @model_validator(mode="after")
    def validate_llm_provider(self) -> Settings:
        if self.llm_profile == "ollama":
            self.llm_provider = "ollama"
            self.openai_compatible_base_url = "http://127.0.0.1:11434/v1"
            self.openai_compatible_api_key = "ollama"
            self.llm_model = self.ollama_model
        elif self.llm_profile == "openrouter":
            self.llm_provider = "openai_compatible"
            self.openai_compatible_base_url = "https://openrouter.ai/api/v1"
            self.llm_model = self.openrouter_model
            if self.openrouter_api_key:
                self.openai_compatible_api_key = self.openrouter_api_key
        elif self.llm_profile == "nvidia":
            self.llm_provider = "openai_compatible"
            self.openai_compatible_base_url = "https://integrate.api.nvidia.com/v1"
            self.llm_model = self.nvidia_model
            if self.nvidia_api_key:
                self.openai_compatible_api_key = self.nvidia_api_key

        if self.llm_provider == "anthropic" and not self.anthropic_api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY is required when LLM_PROVIDER=anthropic "
                "(install optional dependency: pip install "
                "'support-triage[anthropic]' or anthropic)"
            )
        if (
            self.llm_provider == "openai_compatible"
            and self.openai_compatible_api_key == "replace-with-real-api-key"
        ):
            raise ValueError(
                "OPENAI_COMPATIBLE_API_KEY is not set. Provide a real key "
                "or set LLM_PROFILE=ollama for local usage."
            )
        return self

    @field_validator("app_secret_key")
    @classmethod
    def warn_insecure_secret(cls, v: str) -> str:
        if v == "change-me-in-production":
            import warnings

            warnings.warn(
                "APP_SECRET_KEY is set to the default insecure value. "
                "Please set a strong secret in production.",
                stacklevel=2,
            )
        return v

    @property
    def is_production(self) -> bool:
        return self.app_env == Environment.PRODUCTION

    @property
    def sla_map(self) -> dict[str, int]:
        return {
            "critical": self.sla_critical_minutes,
            "high": self.sla_high_minutes,
            "medium": self.sla_medium_minutes,
            "low": self.sla_low_minutes,
        }

    @property
    def api_key_list(self) -> list[str]:
        if not self.api_keys:
            return []
        return [k.strip() for k in self.api_keys.split(",") if k.strip()]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached Settings singleton. Safe for use as a FastAPI dependency."""
    return Settings()  # type: ignore[call-arg]

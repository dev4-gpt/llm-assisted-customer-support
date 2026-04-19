# LLM-Augmented Customer Support Triage & Quality Monitoring

Production-grade FastAPI service that uses a **configurable LLM** (default: **Ollama** / OpenAI-compatible API; optional **Anthropic**) to triage support tickets, evaluate agent response quality, and summarize threads. Includes **lexical or embedding RAG**, **policy-grounded quality**, optional **hybrid triage** (TF–IDF+LR hint), **offline golden-set evaluation**, and a **Zendesk worker stub**. Typed, tested, observable, containerised, and CI-ready.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        FastAPI  /api/v1                      │
│  ┌──────────┐    ┌──────────┐    ┌──────────────────────┐   │
│  │ /triage  │    │ /quality │    │     /pipeline        │   │
│  └────┬─────┘    └────┬─────┘    └──────────┬───────────┘   │
│       │               │                      │ (concurrent)  │
│  ┌────▼─────────────────────────────────────▼───────────┐   │
│  │              Service Layer                            │   │
│  │   TriageService  │  QualityService  │ PipelineService │   │
│  └────────────────────┬──────────────────────────────────┘   │
│                       │                                       │
│  ┌────────────────────▼──────────────────────────────────┐   │
│  │         LLMClient (Ollama / OpenAI-compat / Anthropic)│   │
│  │   • Retry (tenacity)  • JSON parsing  • Error mapping │   │
│  └───────────────────────────────────────────────────────┘   │
│                                                               │
│  ┌─────────────┐   ┌──────────────┐   ┌───────────────────┐ │
│  │ Redis Cache │   │  Prometheus  │   │  Structlog (JSON) │ │
│  └─────────────┘   └──────────────┘   └───────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

### Key design decisions

| Decision | Rationale |
|---|---|
| **Pydantic v2 models as single source of truth** | Enforces contract between layers; auto-generates OpenAPI docs |
| **LLM client decoupled via interface** | Swap Anthropic for OpenAI or local model without touching service logic |
| **Routing matrix in code, not LLM** | Deterministic, auditable, no hallucination risk on business rules |
| **Concurrent triage + quality in pipeline** | Halves wall-clock latency for the full workflow |
| **Redis cache on all endpoints** | Avoids redundant LLM API costs for identical payloads |
| **Prometheus metrics on every endpoint** | Enables SLA dashboards and alerting in production |
| **RAG + grounding** | Quality/triage can consume retrieved policy snippets; `RAG_BACKEND=embedding` optional |
| **Hybrid triage flag** | Optional joblib baseline suggests category; LLM still emits full JSON |
| **`LLM_PROMPT_VERSION` on metrics** | Track prompt/rubric changes in dashboards |

---

## Project Structure

```
support_triage/
├── app/
│   ├── api/v1/
│   │   ├── errors.py          # Exception → HTTP response mapping
│   │   └── routers.py         # All endpoint handlers
│   ├── core/
│   │   ├── config.py          # Pydantic-settings (single env-var access point)
│   │   ├── dependencies.py    # FastAPI DI container (singleton services)
│   │   ├── exceptions.py      # Domain exception hierarchy
│   │   └── logging.py         # Structlog JSON logging
│   ├── models/
│   │   └── domain.py          # Request/response Pydantic models + enums
│   ├── services/
│   │   ├── llm_client.py      # LLM JSON client (retry, parse, metrics)
│   │   ├── triage_service.py  # Ticket triage orchestration
│   │   ├── quality_service.py # Agent response quality evaluation
│   │   ├── pipeline_service.py# End-to-end pipeline (concurrent)
│   │   ├── rag_service.py     # Policy retrieval (lexical or embedding)
│   │   └── summarization_service.py
│   ├── integrations/
│   │   └── zendesk_worker.py  # Zendesk → POST /triage (fixture or live)
│   ├── utils/
│   │   ├── cache.py           # Redis response cache (graceful degradation)
│   │   └── metrics.py         # Prometheus counters/histograms
│   └── main.py                # App factory (API key + audit middleware, routes)
├── tests/
│   ├── unit/
│   │   ├── test_triage_service.py
│   │   ├── test_quality_service.py
│   │   └── test_llm_client.py
│   ├── integration/
│   │   └── test_api.py
│   └── conftest.py
├── data/
│   ├── golden/                # eval_set.jsonl + README
│   └── fixtures/              # zendesk_ticket.json (worker demo)
├── scripts/
│   ├── run_offline_eval.py    # Mock or live benchmark → artifacts/eval/
│   ├── run_eda.py             # EDA plots → artifacts/eda/ (pip install -e ".[eda]")
│   ├── train_encoder_classifier.py
│   └── train_triage_transformer.py  # optional BERT/RoBERTa fine-tune (.[transformer])
├── docker/
│   └── prometheus.yml
├── .github/workflows/ci.yml
├── docker-compose.yml
├── Dockerfile
├── pyproject.toml
├── .env.example
└── sample_payloads.json
```

---

## Quick Start

### 1. Clone and configure

```bash
git clone <repo>
cd llm-assist
cp .env.example .env
# Edit .env — default: Ollama on 11434; or set LLM_PROVIDER=anthropic + ANTHROPIC_API_KEY
```

### 2. Run locally (Python)

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
uvicorn app.main:app --reload
```

### 3. Run with Docker Compose

```bash
docker compose up --build
```

With full observability stack (Prometheus + Grafana):

```bash
docker compose --profile observability up --build
```

### 4. Explore the API

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc
- Health: http://localhost:8000/api/v1/health
- Metrics: http://localhost:8000/metrics

---

## Docs

- Project proposal (PDF): `LLM‑Augmented_Customer SupportTriage_QualityMonitoringSystem .pdf` (repo root)
- `docs/DATASETS.md`: dataset sources, features, splits, preprocessing
- `docs/IMPLEMENTATION_REPORT.md`: implementation report; **section 6** separates **implemented inference stack** from **literature / future transformer fine-tuning** (use for accurate course reporting)
- `docs/METHODOLOGY_EDA_AND_DL.md`: **EDA + deep-learning choices**, train/test strategy, evaluation metrics (for reports and slides)
- `docs/IMPLEMENTATION_CHANGELOG.xlsx`: changelog / tracking sheet

---

## API Reference

### `POST /api/v1/triage`

Triage a support ticket → priority, category, sentiment, routing.

```bash
curl -X POST http://localhost:8000/api/v1/triage \
  -H "Content-Type: application/json" \
  -d '{
    "ticket_text": "I was charged twice for my subscription! This is urgent!"
  }'
```

**Response:**
```json
{
  "priority": "critical",
  "category": "billing",
  "intents": [
    { "label": "billing", "score": 0.94 },
    { "label": "general_inquiry", "score": 0.2 }
  ],
  "sentiment_score": -0.85,
  "routed_team": "critical_response",
  "rationale": "Customer reports a duplicate charge — a high-urgency financial issue requiring immediate billing team intervention.",
  "confidence": 0.94
}
```

---

### `POST /api/v1/quality`

Evaluate an agent draft response against the original ticket.

```bash
curl -X POST http://localhost:8000/api/v1/quality \
  -H "Content-Type: application/json" \
  -d '{
    "ticket_text": "I was charged twice for my subscription!",
    "agent_response": "Sorry for the inconvenience. We will look into this."
  }'
```

**Response:**
```json
{
  "score": 0.25,
  "passed": false,
  "checks": {
    "empathetic_tone": false,
    "actionable_next_step": false,
    "policy_safety": true,
    "resolved_or_escalated": false
  },
  "coaching_feedback": "The response lacks genuine empathy and offers no concrete next step. Replace 'we will look into this' with a specific action and timeline. Add a ticket number to set expectations.",
  "flagged_phrases": []
}
```

---

### `POST /api/v1/pipeline`

Full workflow: triage + quality + SLA recommendation.

```bash
curl -X POST http://localhost:8000/api/v1/pipeline \
  -H "Content-Type: application/json" \
  -d @sample_payloads.json  # see sample_payloads.json for full examples
```

**Response schema:** `{ triage: TriageResult, quality: QualityResult, recommended_sla_minutes: int, workflow_passed: bool }`

---

### `POST /api/v1/summarize`

Summarize a multi-turn thread (ordered `customer` / `agent` / `brand` turns).

```bash
curl -X POST http://localhost:8000/api/v1/summarize \
  -H "Content-Type: application/json" \
  -d '{
    "turns": [
      {"role": "customer", "content": "I was charged twice for my subscription."},
      {"role": "agent", "content": "I have opened billing case #991; we will reply within 2 hours."}
    ]
  }'
```

**Response:** `summary`, `key_points`, `confidence`.

---

### `POST /api/v1/rag/context`

Retrieve top matching **local** policy snippets (`data/policy_snippets.json`). Default: **lexical** overlap; set `RAG_BACKEND=embedding` and `pip install -e ".[embedding]"` for sentence-transformers cosine similarity.

```bash
curl -X POST http://localhost:8000/api/v1/rag/context \
  -H "Content-Type: application/json" \
  -d '{"query": "refund SLA duplicate charge escalation"}'
```

---

## Configuration

All configuration is via environment variables (see `.env.example`).

| Variable | Default | Description |
|---|---|---|
| `LLM_PROVIDER` | `ollama` | `ollama` \| `openai_compatible` \| `anthropic` |
| `LLM_MODEL` | `llama3.2` | Model id on the provider |
| `OPENAI_COMPATIBLE_BASE_URL` | `http://127.0.0.1:11434/v1` | Chat completions base (Ollama) |
| `ANTHROPIC_API_KEY` | — | Required when `LLM_PROVIDER=anthropic` |
| `LLM_PROMPT_VERSION` | `v1` | Label on Prometheus LLM metrics |
| `QUALITY_PASS_THRESHOLD` | `0.70` | Minimum score to pass quality check |
| `QUALITY_POLICY_CONTEXT_TOP_K` | `0` | Inject top-k policy snippets into quality prompt (`>0` enables) |
| `TRIAGE_POLICY_CONTEXT_TOP_K` | `0` | Prepend top-k snippets to triage prompt (`>0` enables) |
| `RAG_BACKEND` | `lexical` | `lexical` or `embedding` (needs `.[embedding]`) |
| `RAG_EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | When `RAG_BACKEND=embedding` |
| `TRIAGE_HYBRID_ENABLED` | `false` | Use TF–IDF+LR joblib hint when model path set |
| `TRIAGE_BASELINE_MODEL_PATH` | — | Path to `train_encoder_classifier.py` output |
| `TRIAGE_TRANSFORMER_ENABLED` | `false` | Use fine-tuned BERT/RoBERTa hint (`pip install -e ".[transformer]"`) |
| `TRIAGE_TRANSFORMER_MODEL_DIR` | — | Directory from `train_triage_transformer.py` (HF checkpoint) |
| `SENTIMENT_ESCALATION_CUTOFF` | `-0.6` | Sentiment below this escalates (high/critical) |
| `SLA_*_MINUTES` | see `.env.example` | SLA map by priority |
| `REDIS_URL` | `redis://localhost:6379/0` | Redis (cache degrades if down) |
| `POLICY_SNIPPETS_PATH` | `data/policy_snippets.json` | Policy JSON for RAG |
| `API_KEYS` | — | Comma-separated; if set, POST `/api/v1/*` needs `X-API-Key` (not `/health`) |
| `AUDIT_LOG_PATH` | — | Append-only JSONL for POST metadata (no bodies) |
| `APP_ENV` | `development` | `development` / `staging` / `production` |
| `APP_LOG_LEVEL` | `INFO` | Log level |

**Reverse proxy:** In production, terminate TLS at the proxy and optionally inject `X-API-Key` there instead of exposing keys to browsers.

---

## Offline evaluation (golden set)

```bash
pip install -e ".[dev]"
# Mock LLM (CI-default): perfect triage match on synthetic gold labels
python scripts/run_offline_eval.py --data data/golden/eval_set.jsonl
# Live LLM (requires working provider in .env)
EVAL_LLM=1 python scripts/run_offline_eval.py --data data/golden/eval_set.jsonl
# Optional TF–IDF baseline category accuracy (after training a joblib pipeline)
python scripts/run_offline_eval.py --mock --baseline-model artifacts/triage_baseline.joblib
```

Outputs: `artifacts/eval/metrics.json`, `artifacts/eval/summary.md` (path is gitignored under `artifacts/`).

## Exploratory data analysis (EDA)

Generate PNG figures from the **in-repo golden set** and, optionally, a **labeled CSV** (`text`, `category`) after you download a public corpus (see `docs/DATASETS.md`).

```bash
pip install -e ".[eda]"
python scripts/run_eda.py
# optional: python scripts/run_eda.py --csv data/raw/your_labeled_export.csv
```

Figures go to `artifacts/eda/` (gitignored). Methodology write-up: [`docs/METHODOLOGY_EDA_AND_DL.md`](docs/METHODOLOGY_EDA_AND_DL.md).

## Optional BERT/RoBERTa category hint (course / demo)

For a visible **transformer fine-tuning** step (HF `Trainer`) that still leaves the **LLM as the source of truth** for full triage JSON:

```bash
pip install -e ".[transformer]"
python scripts/train_triage_transformer.py \
  --data data/raw/tickets_labeled.csv \
  --out artifacts/triage_roberta \
  --model roberta-base
# or: --model bert-base-uncased
```

Then set `TRIAGE_TRANSFORMER_ENABLED=true` and `TRIAGE_TRANSFORMER_MODEL_DIR=artifacts/triage_roberta` (see `.env.example`). You can enable this **together with** the TF–IDF hybrid hint; both appear as suggestions in the prompt.

## Zendesk worker (stub)

Fixture mode (no Zendesk credentials):

```bash
pip install -e .
python -m app.integrations.zendesk_worker --fixture data/fixtures/zendesk_ticket.json --api-base http://127.0.0.1:8000
```

Live fetch: set `ZENDESK_SUBDOMAIN`, `ZENDESK_EMAIL`, `ZENDESK_API_TOKEN`, then `--ticket-id <id>`. Pass `--api-key` if the API enforces `API_KEYS`.

## Running Tests

Default `pytest` flags (in `pyproject.toml`) include `-p no:asyncio` so collection stays stable if a global `pytest-asyncio` install is incompatible with your pytest version.

```bash
# All tests with coverage
pytest

# Unit tests only (fast, no Redis needed)
pytest tests/unit/ -v

# Integration tests
pytest tests/integration/ -v

# With coverage report
pytest --cov=app --cov-report=html
open htmlcov/index.html
```

---

## Observability

### Prometheus Metrics

| Metric | Type | Labels |
|---|---|---|
| `support_triage_llm_requests_total` | Counter | `schema`, `prompt_version` |
| `support_triage_llm_errors_total` | Counter | `schema`, `error_type`, `prompt_version` |
| `support_triage_llm_latency_seconds` | Histogram | `schema`, `prompt_version` |
| `support_triage_http_requests_total` | Counter | `endpoint`, `method`, `status` |
| `support_triage_priority_total` | Counter | `priority` |
| `support_triage_quality_score` | Histogram | — |
| `support_triage_quality_pass_total` | Counter | `result` |
| `support_triage_pipeline_workflow_total` | Counter | `result` |

### Structured Logs

Every request produces structured JSON logs (in production) or colourised console output (in development):

```json
{
  "timestamp": "2026-03-25T12:00:00Z",
  "level": "info",
  "event": "Triage complete",
  "service": "support-triage",
  "priority": "critical",
  "category": "billing",
  "sentiment": -0.85,
  "team": "critical_response"
}
```

---

## Extending the System

### Replace keyword routing with embeddings

The `TriageService.triage()` method calls `self._llm.complete_json()`. To swap in an embedding-based classifier:

1. Implement a new classifier class with a `.predict(text) -> TriageResult` method.
2. Inject it into `TriageService.__init__()` alongside or instead of `LLMClient`.
3. The API contract and routing logic remain unchanged.

### Add a new quality dimension

1. Add the new boolean field to `QualityChecks` in `app/models/domain.py`.
2. Add the rubric description to `_QUALITY_PROMPT` in `app/services/quality_service.py`.
3. Update `_validate_quality_response()` to include the new field.
4. Update tests.

### Connect to Zendesk / Freshdesk

Use [`app/integrations/zendesk_worker.py`](app/integrations/zendesk_worker.py) as a starting point: map ticket JSON → `POST /api/v1/triage`, then push `suggested_zendesk_tags` back via the Zendesk API. Extend with polling or webhooks as needed.

---

## Roadmap

- [x] Embedding RAG option (`RAG_BACKEND=embedding`, `pip install -e ".[embedding]"`)
- [x] Hybrid triage hint (`TRIAGE_HYBRID_ENABLED` + joblib from `train_encoder_classifier.py`)
- [x] Optional encoder fine-tune hint (`TRIAGE_TRANSFORMER_*` + `train_triage_transformer.py`, `.[transformer]`)
- [ ] Named Entity Recognition for order IDs, product names
- [x] Multi-turn conversation summarisation (LLM `/api/v1/summarize`)
- [x] Multi-intent signals on triage (`intents` array)
- [x] Lexical + embedding RAG over policy snippets (`/api/v1/rag/context`)
- [x] Policy grounding in quality (and optional triage) prompts
- [ ] Historical analytics API (queue load, quality trend, escalation rate)
- [ ] Human-in-the-loop override with feedback learning
- [x] Zendesk worker stub (`python -m app.integrations.zendesk_worker`)
- [x] Offline evaluation golden set + `scripts/run_offline_eval.py`
- [x] Prompt version label on LLM metrics (`LLM_PROMPT_VERSION`)
- [ ] Full Freshdesk / Intercom connectors
- [ ] Richer A/B dashboard beyond Prometheus `prompt_version`

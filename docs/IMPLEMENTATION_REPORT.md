# End-to-End Implementation Change Report

This report maps the `llm-assist` codebase to the course proposal (LLM-augmented customer support triage and quality monitoring) and lists what is implemented versus planned.

## 1) Executive overview

The repository implements a **production-style FastAPI** service under the `app/` package:

- **Triage** via LLM structured JSON: priority, primary category, **multi-intent scores**, sentiment, deterministic team routing; optional **policy context** and **hybrid TF–IDF+LR category hint**.
- **Quality** evaluation of agent drafts with rubric checks and coaching feedback; optional **policy snippet grounding** from RAG.
- **Pipeline** running triage and quality concurrently with SLA recommendation.
- **Summarization** of multi-turn threads (customer / agent turns) via LLM JSON.
- **RAG** over local policy snippets: **lexical** (default, CI-friendly) or **embedding** cosine (`RAG_BACKEND=embedding`, requires `pip install -e ".[embedding]"`).
- **Offline evaluation**: golden JSONL + [`scripts/run_offline_eval.py`](../scripts/run_offline_eval.py) (mock LLM by default; set `EVAL_LLM=1` for live model metrics); outputs `artifacts/eval/metrics.json` and `summary.md`.
- **Zendesk bridge** ([`app/integrations/zendesk_worker.py`](../app/integrations/zendesk_worker.py)): fixture JSON or live ticket fetch → `POST /api/v1/triage` → suggested tags.
- **API hardening**: optional comma-separated **`API_KEYS`** (require `X-API-Key` on POST `/api/v1/*` except `/health`); optional **`AUDIT_LOG_PATH`** JSONL (metadata only, no bodies).
- **Prometheus**: LLM metrics include **`prompt_version`** label (`LLM_PROMPT_VERSION`, default `v1`).

Dependencies and configuration are defined in [`pyproject.toml`](../pyproject.toml). Default LLM backend is **OpenAI-compatible** (e.g. **Ollama**); **Anthropic** is optional (`LLM_PROVIDER=anthropic`).

## 2) Package layout (authoritative)

| Path | Role |
|------|------|
| [`app/main.py`](../app/main.py) | FastAPI factory, middleware (API key + audit), `/api/v1` router, `/metrics` |
| [`app/core/config.py`](../app/core/config.py) | `pydantic-settings` (`Settings`): LLM, RAG backend, grounding top-k, hybrid triage, API keys, audit path, `llm_prompt_version` |
| [`app/core/dependencies.py`](../app/core/dependencies.py) | DI: `LLMClient`, cache, triage, quality, pipeline, summarization, **singleton `RAGService`** |
| [`app/core/exceptions.py`](../app/core/exceptions.py), [`logging.py`](../app/core/logging.py) | Errors and structlog |
| [`app/models/domain.py`](../app/models/domain.py) | Pydantic v2 request/response models; `include_policy_context` on triage/quality requests |
| [`app/services/llm_client.py`](../app/services/llm_client.py) | JSON completions (Ollama/OpenAI-compatible or Anthropic); metrics tagged by `prompt_version` |
| [`app/services/triage_service.py`](../app/services/triage_service.py) | Triage prompt, validation, routing matrix, optional RAG prefix + hybrid baseline hint |
| [`app/services/quality_service.py`](../app/services/quality_service.py) | Quality rubric, LLM evaluation, optional policy injection |
| [`app/services/pipeline_service.py`](../app/services/pipeline_service.py) | Concurrent triage + quality |
| [`app/services/summarization_service.py`](../app/services/summarization_service.py) | Thread summarization |
| [`app/services/rag_service.py`](../app/services/rag_service.py) | Lexical or embedding retrieval |
| [`app/integrations/zendesk_worker.py`](../app/integrations/zendesk_worker.py) | Zendesk → triage API stub |
| [`app/api/v1/routers.py`](../app/api/v1/routers.py) | HTTP handlers |
| [`app/utils/cache.py`](../app/utils/cache.py), [`metrics.py`](../app/utils/metrics.py) | Redis cache, Prometheus |
| [`evaluation/`](../evaluation/) | Offline split + metric utilities |
| [`data/golden/`](../data/golden/) | Synthetic eval JSONL + README |
| [`data/fixtures/zendesk_ticket.json`](../data/fixtures/zendesk_ticket.json) | Zendesk worker fixture |
| [`docs/DATASETS.md`](DATASETS.md) | Dataset sources, features, splits, preprocessing |
| [`scripts/run_offline_eval.py`](../scripts/run_offline_eval.py) | Reproducible benchmark runner |
| [`scripts/train_encoder_classifier.py`](../scripts/train_encoder_classifier.py) | Baseline TF–IDF + logistic regression (hybrid triage) |

Legacy root-level `app.py`, `pipeline.py`, and `config.py` are **not** used; all logic lives under `app/`.

## 3) API surface (`/api/v1`)

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Liveness; Redis check when cache enabled |
| POST | `/triage` | Ticket triage + multi-intent list |
| POST | `/quality` | Agent response quality |
| POST | `/pipeline` | Full triage + quality + SLA |
| POST | `/summarize` | Multi-turn conversation summary |
| POST | `/rag/context` | Top-k policy snippets |

OpenAPI: `/docs`, `/redoc`.

## 4) Workflow coverage matrix

| Stage | Implemented | Location |
|-------|-------------|----------|
| Ticket / thread ingestion | Yes | `TriageRequest`, `SummarizeRequest` |
| Priority classification | Yes | `TriageService`, LLM JSON |
| Primary category (routing) | Yes | `Category` enum + routing matrix |
| Multi-intent (multi-label signals) | Yes | `intents` on `TriageResult` |
| Sentiment score | Yes | LLM + escalation override |
| Team routing | Yes | `_ROUTING_MATRIX` + sentiment rule |
| Agent response quality | Yes | `QualityService` |
| Policy grounding (quality / triage) | Yes | `quality_policy_context_top_k`, `triage_policy_context_top_k`, `RAGService` |
| Hybrid triage hint | Yes | TF–IDF+LR joblib (`triage_hybrid_*`) and/or optional HF encoder (`triage_transformer_*`) |
| End-to-end orchestration | Yes | `PipelineService` |
| Multi-turn summarization | Yes | `SummarizationService` |
| RAG over policies | Yes | Lexical + optional embedding |
| Offline eval / golden set | Yes | `data/golden/eval_set.jsonl`, `scripts/run_offline_eval.py` |
| Ticketing integration (stub) | Yes | `app/integrations/zendesk_worker.py` |
| Append-only audit metadata | Yes | `AUDIT_LOG_PATH` |
| Optional API key gate | Yes | `API_KEYS` + `X-API-Key` |
| Persistent audit store (DB) | No | Planned |
| Full OAuth / RBAC | No | Planned |

## 5) Offline evaluation methodology

1. **Data:** [`data/golden/eval_set.jsonl`](../data/golden/eval_set.jsonl) — synthetic rows (`task`: `triage` | `quality` | `summarize`). See [`data/golden/README.md`](../data/golden/README.md).
2. **Mock run (CI):** `python scripts/run_offline_eval.py` — uses a queue LLM that returns gold-aligned JSON; expect perfect triage accuracy on the golden labels.
3. **Live run:** `EVAL_LLM=1` with a working LLM env — reports real accuracy / mean quality / ROUGE-L (single reference; interpret cautiously).
4. **Optional baseline:** `--baseline-model artifacts/triage_baseline.joblib` after `train_encoder_classifier.py` — category accuracy vs gold.
5. **Limitations:** LLM-as-judge drift; single-reference summarization; small public gold set — expand for production reporting.
6. **EDA (optional):** [`scripts/run_eda.py`](../scripts/run_eda.py) with `pip install -e ".[eda]"` — plots task/category/priority/length distributions; see [`METHODOLOGY_EDA_AND_DL.md`](METHODOLOGY_EDA_AND_DL.md).

## 6) Deep learning narrative: implemented stack vs literature / future training

**Use this section in course reports** to avoid claiming training that does not exist in the repository.

### 6.1 Implemented inference stack (what the code does today)

| Layer | Mechanism | Notes |
|-------|-----------|--------|
| **Primary “deep” models** | Pre-trained **LLMs** via API ([`LLMClient`](../app/services/llm_client.py)) | **Inference only** — no gradient updates in this repo. Triage, quality, summarization use **structured JSON** prompts + validation. |
| **Dense retrieval (optional)** | **sentence-transformers** when `RAG_BACKEND=embedding` ([`RAGService`](../app/services/rag_service.py)) | **Inference only** on a fixed encoder for policy similarity. |
| **Classical baseline / hybrid** | **TF–IDF + logistic regression** ([`scripts/train_encoder_classifier.py`](../scripts/train_encoder_classifier.py)) + optional hint in [`TriageService`](../app/services/triage_service.py) | **Not** a deep neural network; serves baseline metrics and a fast category hint. |
| **Optional encoder fine-tune (course demo)** | **BERT / RoBERTa** sequence classification via HF [`Trainer`](https://huggingface.co/docs/transformers/main_classes/trainer) ([`scripts/train_triage_transformer.py`](../scripts/train_triage_transformer.py)) + optional hint ([`triage_transformer_predict.py`](../app/services/triage_transformer_predict.py)) | **Off by default**; requires `pip install -e ".[transformer]"` and env `TRIAGE_TRANSFORMER_*`. Same role as the TF–IDF hint: a **category suggestion** prepended to the LLM prompt—not a replacement for the LLM stack. |
| **Lexical RAG (default)** | Token overlap | No neural model. |

**Not implemented in-repo:** training **T5 / BART** for summarization inside this repo; **BiLSTM / CNN** baselines; **multi-task encoder + multiple heads**; **BERT token-classification NER** for order IDs. **BERT/RoBERTa** appear only as the **optional** category head above, not as the primary triage engine.

### 6.2 Literature and datasets (related work, not this repo’s training pipeline)

[`literature_review.md`](../literature_review.md) and [`docs/DATASETS.md`](DATASETS.md) discuss papers and public corpora (e.g. multi-task intent, TWEETSUMM, emotion datasets). Those inform **design motivation** and **possible evaluation data**, not an assertion that this codebase **trains** the architectures cited. Keep that distinction explicit in write-ups.

### 6.3 Why the project is not built around full DL fine-tuning

1. **Labels and maintenance** — Per-task transformer fine-tuning needs sustained **labeled data** and taxonomy upkeep; prompting one LLM covers several tasks without separate heads.
2. **MLOps** — Training-centric stacks need **experiment tracking, registries, rollbacks**; this repo prioritizes **deployed API**, caching, metrics, and offline eval.
3. **Time-to-behavior** — End-to-end triage + quality + summarization + RAG is faster to realize via **one LLM interface** than training and serving many specialists.
4. **Honest positioning** — A **deployment-first, LLM-inference-first** system is valid for “AI for support”; it is a different artifact than a **research training codebase**.

### 6.4 When full transformer fine-tuning would be justified

Consider adding real training (e.g. Hugging Face **Trainer**) if you need: **lower cost/latency at volume**, **no external LLM** (privacy/air-gap), a **stable narrow taxonomy** with **enough labels** to beat prompts, or a **course requirement** to show **training curves and baseline vs transformer** comparisons.

### 6.5 Optional single-task fine-tuned encoder (implemented, off by default)

For coursework or demos that require a visible **training loop + BERT/RoBERTa checkpoint**, the repository includes:

- **Task:** primary **category** only, CSV `text,category` aligned with [`Category`](../app/models/domain.py) (same contract as [`train_encoder_classifier.py`](../scripts/train_encoder_classifier.py)).
- **Training:** [`scripts/train_triage_transformer.py`](../scripts/train_triage_transformer.py) — **Hugging Face `Trainer`**, defaults to `roberta-base` (override with `--model bert-base-uncased`, etc.). Install: `pip install -e ".[transformer]"`.
- **Artifact:** HF save directory under `artifacts/` (e.g. `artifacts/triage_roberta`) — `config.json`, tokenizer, weights (gitignored).
- **Serving:** `TRIAGE_TRANSFORMER_ENABLED=true` and `TRIAGE_TRANSFORMER_MODEL_DIR=...` — [`TriageService`](../app/services/triage_service.py) prepends an encoder **hint** before the LLM (can be combined with the TF–IDF hybrid hint). The LLM remains authoritative for the full JSON triage output.

This path is **optional** so default deploys stay lightweight (no PyTorch on the critical path unless you enable it).

## 7) Verification

```bash
pip install -e ".[dev]"
ruff check app/ tests/
mypy app/
pytest
python scripts/run_offline_eval.py --data data/golden/eval_set.jsonl
```

Coverage gate: `--cov-fail-under=80` in `pyproject.toml`.

## 8) Related documents

- [`README.md`](../README.md): runbook, architecture, configuration.
- [`docs/DATASETS.md`](DATASETS.md): dataset alignment and preprocessing.
- [`literature_review.md`](../literature_review.md): papers and lessons learned (not implementation claims).
- [`docs/IMPLEMENTATION_CHANGELOG.xlsx`](IMPLEMENTATION_CHANGELOG.xlsx): spreadsheet tracker (if present).

## 9) Open gaps (production hardening)

- Named-entity recognition and PII redaction hooks.
- Vector DB / incremental index updates for large policy corpora.
- Human evaluation workflow and CSAT correlation studies using `evaluation/correlation.py`.
- Prompt registry UI and automated A/B analysis beyond `LLM_PROMPT_VERSION` metrics.

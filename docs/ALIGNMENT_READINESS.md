# Project Alignment and Readiness

This document maps project claims to what is implemented and demo-ready, and defines a single reproducible run path for submission/demo.

## Golden run path (reproducible, no external LLM dependency)

Use this path to prove the end-to-end workflow in a deterministic way:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest tests/integration/test_pipeline_e2e.py -v
```

What this validates in one run:
- API wiring and dependency injection
- End-to-end `/api/v1/pipeline` flow (triage + quality + SLA decision)
- Supporting `/api/v1/summarize` endpoint
- Supporting `/api/v1/rag/context` endpoint

Output evidence:
- `1 passed` from `tests/integration/test_pipeline_e2e.py`

## Claims vs evidence matrix

| Feature | Claimed in report/proposal | Implemented in code | Demo status | Reproducible command | Notes |
|---|---|---|---|---|---|
| Ticket triage (priority/category/intents/routing) | Yes | Yes | Demo-ready | `pytest tests/integration/test_pipeline_e2e.py -v` | Fully wired via `/api/v1/triage` and `/api/v1/pipeline` |
| Agent quality scoring + coaching | Yes | Yes | Demo-ready | `pytest tests/integration/test_pipeline_e2e.py -v` | Rubric-based output from `/api/v1/quality` and `/api/v1/pipeline` |
| End-to-end workflow orchestration | Yes | Yes | Demo-ready | `pytest tests/integration/test_pipeline_e2e.py -v` | Concurrent triage + quality in `PipelineService` |
| Conversation summarization | Yes | Yes | Demo-ready | `pytest tests/integration/test_pipeline_e2e.py -v` | `/api/v1/summarize` included in golden run |
| Policy RAG context retrieval | Yes | Yes | Demo-ready | `pytest tests/integration/test_pipeline_e2e.py -v` | Lexical backend is default and deterministic |
| Invalid-label recovery for triage taxonomy | Implicit robustness need | Yes | Demo-ready | `pytest tests/integration/test_pipeline_label_recovery.py -v` | Uses synonym mapping + optional embedding similarity fallback |
| Offline evaluation artifacts | Yes | Yes | Demo-ready | `python scripts/run_offline_eval.py --data data/golden/eval_set.jsonl` | Creates `artifacts/eval/*` |
| Data augmentation via LLM | Yes | Partial (design-level, no dedicated module) | Planned/optional | N/A | Mention as extension, not core demo |
| NER extraction module | Yes | Not implemented as standalone module | Planned | N/A | Keep in roadmap/future work |
| Full external helpdesk write-back loop | Implied | Partial (Zendesk worker stub) | Partial demo | `python -m app.integrations.zendesk_worker --fixture data/fixtures/zendesk_ticket.json --api-base http://127.0.0.1:8000` | No full bi-directional production sync |

## Final demo scope (recommended)

Present as **Core (guaranteed)** vs **Extensions (optional)**:

- Core (guaranteed): triage, quality, pipeline orchestration, summarization, lexical RAG, offline eval.
- Extensions (optional): embedding RAG, baseline/transformer category hints, live provider comparisons, Zendesk live mode.
- Reliability NLP technique (class-aligned): embedding-similarity fallback that maps off-schema labels (e.g. `refund`) to allowed taxonomy labels.
- Future work: NER, richer analytics dashboard, full production connector write-back.

## Environment parity notes

- CI and Docker smoke use the same explicit OpenAI-compatible provider profile variables.
- Integration/e2e tests use deterministic LLM mocking to avoid network/runtime variance.
- Live provider demos remain supported through `.env`, but are no longer required to prove end-to-end functionality.
- For hosted models, keep `TRIAGE_EMBEDDING_FALLBACK_ENABLED=true` to reduce pipeline failures from taxonomy drift.

# Claims vs Evidence Matrix

Source claim document: `LlmCustomerSupport_project_report (2).pdf`

Status labels:
- `implemented`: exists in runtime code path
- `demoed`: validated in tests/notebook/demo flow
- `planned`: discussed but not fully implemented in current repo
- `partial`: scaffold/stub exists, but not full production behavior

## Matrix

| Claim | Evidence paths | Status | Presentation wording |
|---|---|---|---|
| End-to-end support NLP pipeline (triage + quality + summarize) | `app/api/v1/routers.py`, `app/services/pipeline_service.py`, `tests/integration/test_pipeline_e2e.py` | implemented, demoed | “Fully implemented API pipeline; demoed end-to-end.” |
| Ticket triage (priority/category/intents/sentiment/routing) | `app/services/triage_service.py`, `app/models/domain.py` | implemented | “Structured LLM inference with deterministic routing matrix.” |
| Quality scoring/coaching rubric | `app/services/quality_service.py` | implemented | “Rubric-constrained scoring + actionable feedback.” |
| Summarization endpoint | `app/services/summarization_service.py`, `notebooks/llm_assist_showcase.ipynb` | implemented, demoed | “Multi-turn summarization available via `/api/v1/summarize`.” |
| RAG grounding support | `app/services/rag_service.py`, `data/policy_snippets.json` | implemented | “Lexical RAG default; embedding RAG optional.” |
| Robust taxonomy handling for off-schema labels | `app/services/intent_fallback_service.py`, `tests/integration/test_pipeline_label_recovery.py` | implemented, demoed | “Invalid labels recovered via deterministic synonym + embedding fallback.” |
| Optional transformer fine-tuning path | `scripts/train_triage_transformer.py`, `app/services/triage_service.py` | implemented (optional) | “Fine-tuned encoder used as optional category hint.” |
| Optional classical baseline path | `scripts/train_encoder_classifier.py`, `app/services/triage_service.py` | implemented (optional) | “TF-IDF/LR baseline available as optional hybrid hint.” |
| Offline evaluation framework | `scripts/run_offline_eval.py`, `data/golden/eval_set.jsonl`, `evaluation/metrics.py` | implemented, demoed | “Deterministic offline evaluation included.” |
| Large-scale real corpus training execution | `docs/DATASETS.md`, `data/download_kaggle.py` | planned/partial | “Datasets are documented and fetchable; full-scale runs are optional/ongoing.” |
| NER module for IDs/products/dates | no standalone NER service/module | planned | “Planned future work.” |
| Full production helpdesk write-back loop | `app/integrations/zendesk_worker.py` | partial | “Zendesk worker stub/demo exists; full bi-directional sync is future work.” |

## Wording risks to avoid in final presentation

- Do not claim full multi-corpus training results unless showing artifacts and metrics from those runs.
- Do not present optional transformer/baseline hint paths as the mandatory runtime backbone.
- Do not claim full production connector automation; present current integration as a validated stub path.

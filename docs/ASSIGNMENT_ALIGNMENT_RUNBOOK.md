# Assignment Alignment Runbook (Single Source)

This document is the unified assignment-facing reference for `llm-assist`.  
It consolidates objective alignment, datasets/splits/imbalance policy, required metrics, reproducible execution, and presentation readiness.

## 1) Objectives and Assignment Alignment

- End-to-end NLP pipeline is implemented through FastAPI endpoints (`/triage`, `/quality`, `/pipeline`, `/summarize`, `/rag/context`).
- Deep-learning path is implemented with optional transformer fine-tuning in `scripts/train_triage_transformer.py`.
- Classical baseline is implemented in `scripts/train_encoder_classifier.py` and can be compared in offline eval.
- Lecture-aligned robustness is implemented with semantic fallback mapping in `app/services/intent_fallback_service.py`.
- Reproducibility is provided through notebook-first execution in `notebooks/llm_assist_showcase.ipynb` plus automated tests.

## 2) Dataset and Split Evidence (Task-wise)

| Task | Labeled dataset used | Split strategy | Conversation-level separation | Imbalance handling |
|---|---|---|---|---|
| Triage category/priority | `data/golden/eval_set.jsonl` for eval; optional labeled CSV for training (`data/raw/tickets_labeled.csv`) | Eval set fixed; training scripts use explicit split (`evaluation/splits.py` for stratified index split) | Triage eval rows are single-ticket records; no thread leakage in golden set | Class distribution checked in EDA (`scripts/run_eda.py`); minority metrics surfaced in eval output |
| Sentiment (from triage output) | Produced by LLM triage JSON in runtime | No separate supervised split currently | N/A (inference attribute inside triage output) | Not separately rebalanced; documented as inference-time signal |
| Summarization | `data/golden/eval_set.jsonl` summarize rows with `gold_summary` | Fixed eval rows | Each summarize sample is a self-contained thread, no cross-thread mixing | Small eval-set limitation documented; ROUGE-L reported as regression metric |
| Quality scoring | `data/golden/eval_set.jsonl` quality rows | Fixed eval rows | Ticket/response pair is self-contained | Rubric score evaluated by mean score tracking; class imbalance N/A for numeric score |

Notes:
- Conversation-level separation is enforced by using independent examples in the golden set; no shared thread IDs across tasks in the current eval file.
- For larger future experiments, use grouped splits by conversation ID before randomization.

## 3) Required Metrics Coverage

For triage outputs, the evaluator now publishes:

- Accuracy
- Micro F1
- Macro F1
- Per-label precision/recall/F1/support
- Confusion matrix
- Minority-class performance slice

Implementation paths:
- Metric computation: `evaluation/metrics.py`
- Eval driver: `scripts/run_offline_eval.py`
- Notebook rendering: `notebooks/llm_assist_showcase.ipynb` section 7

## 4) Live-Only Evaluation Policy

Final presentation/report artifacts must come from live evaluation only:

```bash
EVAL_LLM=1 python scripts/run_offline_eval.py --data data/golden/eval_set.jsonl
```

Canonical artifacts:
- `artifacts/eval/metrics_live.json`
- `artifacts/eval/summary_live.md`

Presentation/report generation (`scripts/generate_presentation_assets.py`) consumes `metrics_live.json` and `summary_live.md` first.

## 5) Notebook-First Reproducible Run

Use `notebooks/llm_assist_showcase.ipynb` as the single execution surface:

1. Environment/setup checks
2. Optional training demo
3. Uvicorn startup in notebook subprocess
4. Endpoint smoke and pipeline calls
5. Test suite run
6. Live evaluation + required triage analytics
7. EDA figures
8. Regenerated PPTX/PDF assets

## 6) Script and Folder Explainability

### `scripts/`
- `run_offline_eval.py`: live eval driver and markdown summary writer
- `run_eda.py`: class/task distribution and length plots
- `train_encoder_classifier.py`: TF-IDF + LR baseline for triage category
- `train_triage_transformer.py`: transformer fine-tune path (BERT/RoBERTa)
- `generate_presentation_assets.py`: regenerates 50-slide deck and updated report

### `evaluation/`
- `metrics.py`: classification and ROUGE metrics
- `splits.py`: deterministic stratified split helper
- `correlation.py`: judge-proxy correlation utilities

### `app/`
- API contracts, services, provider client, fallback mapping, pipeline orchestration

### `tests/`
- Unit tests for service behavior and metrics helpers
- Integration tests for end-to-end API flow and label recovery path

### `data/`
- Golden eval data and policy snippets

### `artifacts/`
- Generated outputs: eval JSON/markdown, EDA plots, trained model checkpoints

### `notebooks/`
- Final reproducible runbook notebook for presentation

## 7) Testing Matrix (What validates what)

- `tests/integration/test_pipeline_e2e.py`: full pipeline and supporting endpoints
- `tests/integration/test_pipeline_label_recovery.py`: invalid-label recovery
- `tests/unit/test_intent_fallback_service.py`: synonym/embedding fallback behavior
- `tests/unit/test_triage_service.py`: triage normalization and validation path
- `tests/unit/test_evaluation.py`: metric helper correctness

## 8) Repo Cleanup Policy

- Keep source, tests, docs, and required artifacts for reproducibility/presentation.
- Remove only generated clutter/cache files.
- `support_triage.egg-info` should remain untracked/generated only (safe to delete if present locally).

## 9) Final Presentation-Day Checklist

- `.env` points to working live provider/profile.
- Notebook kernel is project `.venv`.
- Notebook sections execute in order without terminal switching.
- `artifacts/eval/metrics_live.json` refreshed in the same session.
- Triage metrics table includes micro/macro F1 and confusion matrix.
- `LLM_Assist_Final_Presentation_50_Slides.pptx` regenerated.
- `LlmCustomerSupport_project_report_updated.pdf` regenerated.
- Core integration tests pass.

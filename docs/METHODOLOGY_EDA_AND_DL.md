# Methodology: EDA, deep-learning choices, train/test strategy, and metrics

This document supports coursework and presentations: **what we plot**, **which neural / non-neural components we use and why**, **how we split data**, and **how we measure quality**.

---

## 1) Exploratory data analysis (EDA) and visualizations

### 1.1 Data you can analyze today

| Source | Path | Role |
|--------|------|------|
| **Golden eval set** (in-repo) | [`data/golden/eval_set.jsonl`](../data/golden/eval_set.jsonl) | Small, **synthetic**, public-safe JSONL for triage / quality / summarize tasks. Good for **pipeline demos** and **CI**. |
| **Labeled ticket CSV** (optional) | e.g. Kaggle-style export; columns `text`, `category` | Use after you download a public dataset (see [`DATASETS.md`](DATASETS.md)) for **class balance** and **length** EDA aligned with [`train_encoder_classifier.py`](../scripts/train_encoder_classifier.py) / [`train_triage_transformer.py`](../scripts/train_triage_transformer.py). |

### 1.2 How to generate figures

```bash
pip install -e ".[eda]"
python scripts/run_eda.py
# Optional second source (imbalance, real scale):
python scripts/run_eda.py --csv path/to/tickets_labeled.csv
```

Figures are written to **`artifacts/eda/`** (gitignored). Typical outputs:

- **`golden_task_counts.png`** — how many rows per `task` (triage / quality / summarize).
- **`golden_triage_category.png`**, **`golden_triage_priority.png`** — label distribution on triage gold rows (the bundled set is intentionally small and balanced for testing, not representative of production skew).
- **`golden_text_length_by_task.png`** — histogram of `ticket_text` length by task (proxy for cost / truncation risk).
- With **`--csv`**: **`csv_category_counts.png`**, **`csv_text_length.png`** — category imbalance and length for supervised training data.

**Interpretation tip:** For reports, combine **golden-set** plots (reproducible) with **one public CSV** (from [`DATASETS.md`](DATASETS.md)) to discuss **real-world imbalance** and **long-tail categories**.

---

## 2) Deep learning process: model choices and justification

### 2.1 What “deep learning” means in this project

The system is **deployment-first**: the main “deep” component is a **large pre-trained language model (LLM)** accessed via [`LLMClient`](../app/services/llm_client.py) (**inference only** in this repository). Additional neural components are **optional** and narrower.

| Component | Model family | Role | Why this choice |
|-----------|--------------|------|------------------|
| **Triage / quality / summarize** | General-purpose **LLM** (e.g. via Ollama or Anthropic) | Structured JSON outputs + rubrics | One interface covers **many tasks** without training separate heads; fastest path to end-to-end behavior. |
| **Policy RAG (optional)** | **Sentence-transformer** encoders | Dense similarity over policy snippets | Strong semantic retrieval vs pure lexical overlap when `RAG_BACKEND=embedding`. Still **inference-only** on fixed weights unless you add fine-tuning elsewhere. |
| **Hybrid hint (optional)** | **TF–IDF + logistic regression** | Fast **category** suggestion to prepend to the triage prompt | **Not** a deep net: cheap baseline, interpretable, no GPU; useful for **metrics** and **latency** comparisons in reports. |
| **Encoder fine-tune (optional)** | **BERT / RoBERTa** sequence classification ([`train_triage_transformer.py`](../scripts/train_triage_transformer.py)) | Same role as TF–IDF hint: **single-task category** suggestion | Satisfies coursework that expects a **Hugging Face `Trainer`** loop and **encoder fine-tuning**; **off by default** (`TRIAGE_TRANSFORMER_*`). |

**Not primary in this repo:** training **T5/BART** for summarization here, **BiLSTM/CNN** ticket classifiers, **multi-task** one-model-many-heads, or **NER** token classifiers. Those remain **literature / extension** unless you add new code paths.

### 2.2 Why not “everything” (BERT + RoBERTa + T5 + multi-task + NER at once)?

Each addition needs **labels**, **serving path**, and **versioning**. Stacking many models increases **MLOps** cost without a clear gain until you have **volume**, **privacy constraints**, or **stable taxonomies**. The implemented pattern is: **LLM for behavior**, **optional small specialists** for hints or retrieval.

---

## 3) Training / testing strategy

### 3.1 Supervised baselines and optional encoder (category)

| Script | Split | Notes |
|--------|-------|--------|
| [`scripts/train_encoder_classifier.py`](../scripts/train_encoder_classifier.py) | **Stratified** train/test (`sklearn.model_selection.train_test_split`, default test **15%**) | Input: CSV `text`, `category`. Outputs **joblib** pipeline + `*.metrics.json` (sklearn `classification_report`). |
| [`scripts/train_triage_transformer.py`](../scripts/train_triage_transformer.py) | Same idea: stratified when possible; **fallback** to unstratified split if a class is too small | HF **`Trainer`** with **epoch** eval; **best checkpoint** by validation **accuracy**. Categories restricted to [`Category`](../app/models/domain.py) enum values. |

### 3.2 End-to-end system evaluation (triage / quality / summarize)

| Mechanism | Split | Notes |
|-----------|-------|--------|
| [`scripts/run_offline_eval.py`](../scripts/run_offline_eval.py) + [`data/golden/eval_set.jsonl`](../data/golden/eval_set.jsonl) | Fixed **held-out** JSONL lines (small gold set) | **Mock LLM** (CI): returns gold-aligned JSON → checks plumbing. **`EVAL_LLM=1`**: real LLM → reports **triage accuracy**, **quality score**, **ROUGE-L** on summarize (single reference; interpret cautiously). |
| Optional `--baseline-model` | Uses same golden triage rows | Reports **category accuracy** of the TF–IDF+LR joblib vs `gold_category`. |

### 3.3 What to say in a report

- **Training** (when you run it): describe **stratification**, **seed**, **class filter** (enum alignment), and **where checkpoints live** (`artifacts/`, gitignored).
- **Testing**: distinguish **unit/integration tests** (pytest), **offline golden eval**, and any **manual** API checks (`/docs`).

---

## 4) Evaluation metrics

| Layer | Metric | Where / how |
|-------|--------|-------------|
| **Triage** | **Category accuracy**, **priority accuracy** (exact match to gold) | `run_offline_eval.py` on golden `triage` rows |
| **Triage (optional baseline)** | **Accuracy** vs `gold_category` | `--baseline-model` + joblib pipeline |
| **Quality** | Mean **score**, pass rate vs threshold | Offline eval (no human gold in golden file for quality) |
| **Summarize** | **ROUGE-L F1** vs `gold_summary` | `evaluation/metrics.py` |
| **Supervised category models** | Precision / recall / F1 per class (`classification_report`) | `train_encoder_classifier` metrics JSON; `train_triage_transformer` writes `train_metrics.json` under the output dir |
| **Production-style** | Latency histograms, counters | Prometheus (`/metrics`); LLM metrics labeled by `LLM_PROMPT_VERSION` |

---

## 5) Cross-links

- Implementation truth table: [`IMPLEMENTATION_REPORT.md`](IMPLEMENTATION_REPORT.md) (especially **section 6** — inference vs literature).
- Dataset sourcing: [`DATASETS.md`](DATASETS.md).
- Runbook: [`README.md`](../README.md).

# Datasets: sources, features, and preprocessing

Aligning public corpora with the tasks in the LLM-augmented support system (**triage**, **multi-intent**, **sentiment**, **emotion**, **summarization**, **quality proxies**). **Confirm exact row counts, column names, and licenses on each Kaggle page or repository before final citations in your report.**

For **EDA plots and methodology** (train/test, metrics, model choices), see [`METHODOLOGY_EDA_AND_DL.md`](METHODOLOGY_EDA_AND_DL.md) and run [`scripts/run_eda.py`](../scripts/run_eda.py) after `pip install -e ".[eda]"`.

---

## 1) Customer support on Twitter (dialogue, noise, threads)

| Field | Detail |
|-------|--------|
| **Source** | [Customer Support on Twitter (ThoughtVector)](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter) |
| **Size** | Full corpus ~3M tweets; project subset **80k–100k** messages (stratified sample) |
| **Key features** | Tweet text, timestamps, author IDs, `response_tweet_id` / conversation linkage |
| **Tasks** | Intent/topic signals, sentiment, **multi-turn context**, dialogue-style summarization, robustness to informal language |
| **Notes** | Channel differs from private tickets; use for **generalization**, not as sole source of routing labels |

---

## 2) TWEETSUMM (dialogue summarization benchmark)

| Field | Detail |
|-------|--------|
| **Source** | [TWEETSUMM (arXiv PDF)](https://arxiv.org/pdf/2111.11894.pdf); obtain data via authors / repositories linked in the paper |
| **Key features** | Customer-service **dialogues** and **reference summaries** |
| **Tasks** | Primary **summarization** evaluation for support-style threads (vs generic news) |
| **Metrics** | ROUGE + small **human rubric** samples (automatic metrics are weak for dialogue) |

---

## 3) News summary (auxiliary abstractive pairs)

| Field | Detail |
|-------|--------|
| **Source** | [News Summary](https://www.kaggle.com/datasets/sunnysai12345/news-summary) |
| **Key features** | Article–headline (or article–summary) pairs |
| **Tasks** | **Optional** auxiliary pretraining / transfer for abstractive compression only |
| **Limitation** | **Not** a substitute for dialog summarization metrics; do not claim CS-thread performance from this alone |

---

## 4) Amazon reviews (satisfaction proxy)

| Field | Detail |
|-------|--------|
| **Source** | [Amazon Reviews for NLP](https://www.kaggle.com/datasets/bittlingmayer/amazonreviews) |
| **Key features** | Review text, star rating |
| **Tasks** | Weak proxy for **text → satisfaction**; sanity-check lexical patterns vs numeric ratings |
| **Notes** | Not agent–customer dialogue; use with **LLM-as-judge** / ticket CSAT only as **approximate** validation |

---

## 5) Customer Support Data (single-table ticket dump, proposal alignment)

| Field | Detail |
|-------|--------|
| **Source** | [Customer Support Data](https://www.kaggle.com/datasets/akashbommidi/customer-support-data) |
| **Size** | **>85,000** entries (confirm on dataset card) |
| **Key features** | Issue metadata, categorization, agent information, customer feedback, transactional fields (use actual column names from the CSV after download) |
| **Tasks** | Supervised **triage**, **routing**, class-imbalance analysis, predictive satisfaction |
| **Split** | **70% / 15% / 15%** train / validation / test with **stratification** on major categories when label counts allow |

---

## 6) MAIA-DQE (customer-support emotion + dialogue quality)

| Field | Detail |
|-------|--------|
| **Source** | Paper: [Dialogue Quality and Emotion Annotations for Customer Support Conversations](https://aclanthology.org/2023.gem-1.2/) (GEM 2023); data & code: [johndmendonca/MAIA-DQE](https://github.com/johndmendonca/MAIA-DQE) |
| **Size** | **612** dialogues, **~25k** sentences (per paper/repo; verify in `data/`) |
| **Key features** | Multi-turn **agent vs customer** turns (`inbound` / `outbound`); sentence-level **Emotion**, **Engagement**, **Correctness**, **Templated**; turn-level **Understanding**, **Sensibleness**, **Politeness**, **IQ**; dialogue-level **task success** / dropped-conversation flags |
| **Tasks** | Fine-grained **emotion** models aligned with support; **dialogue quality estimation** aligned with this project’s LLM **quality-monitoring** component |
| **Notes** | Stronger domain fit than generic social text (e.g. Reddit): same domain as triage/summarization |

---

## Feature categories (cross-corpus)

Features are combined across Twitter, TWEETSUMM, news summary, Amazon, ticket CSV, and MAIA-DQE JSON.

| Category | Description | Typical sources |
|----------|-------------|-----------------|
| **Textual** | Customer message, agent/brand reply, full thread or ticket body, dialogue turns (`text_src` / `text_mt` in MAIA), article body (news), review text (Amazon) | Twitter, TWEETSUMM, tickets, MAIA, news, Amazon |
| **Labels** | Issue type / category, **multi-intent**, **emotion** (MAIA sentence-level + triage sentiment axis), **priority** (when mapped from tickets), **resolution** / outcome, summarization targets (TWEETSUMM references) | Tickets, MAIA, TWEETSUMM, LLM-derived labels for Twitter subsets |
| **Numeric** | Scalar **sentiment** (triage), **satisfaction** / CSAT-style scores (tickets), **star ratings** (Amazon as proxy), quality rubric scores (MAIA; comparable spirit to `/quality` checks) | Tickets, Amazon, MAIA, API outputs |
| **Metadata** | **Timestamps**, **IDs** (tweet, ticket, dialogue), **channel** (public Twitter vs ticket system), **floor** direction (MAIA inbound/outbound), language (MAIA bilingual) | Twitter, tickets, MAIA |

---

## Preprocessing pipeline (shared)

Steps apply per corpus with corpus-specific tweaks noted below.

1. **Cleaning**  
   - Unicode normalization, consistent whitespace.  
   - **Twitter:** strip/normalize URLs, handles, and noisy tokens; optional language filter if you restrict to English.  
   - **Kaggle CSVs (tickets, Amazon, news):** parse encodings safely; strip HTML or markup if present in text columns.  
   - **MAIA-DQE:** load JSON subsets from repo `data/`; align sentence lists with parallel annotation lists (same length per turn).

2. **Truncation**  
   - Enforce max token/character length for transformer encoders and for LLM context (triage, summarize, quality).  
   - Long threads: chunk with overlap or summarize intermediate segments before full-thread modeling.

3. **Missing values**  
   - Drop rows with empty primary text fields for supervised tasks; or impute labels only when a documented rule applies (e.g. “unknown” category).  
   - Document every rule per task (triage vs emotion vs summarization).

4. **Anonymization**  
   - Hash or remove user IDs, ticket IDs in published artifacts; redact order/account patterns where required for demos or submission.  
   - Keep internal stable IDs for joining labels inside your train/val/test pipeline only.

5. **Imbalance**  
   - Class weights, stratified sampling, or resampling for rare ticket categories and rare emotions in MAIA.  
   - Optional **synthetic augmentation** for minority classes (document ethics and data lineage if LLM-generated).

6. **Splits & leakage (recommended)**  
   - **Tickets / Amazon / news:** stratified **70% / 15% / 15%** where labels permit.  
   - **Twitter / TWEETSUMM / MAIA:** split by **conversation or dialogue ID** so turns from the same thread do not appear in both train and test.

---

## Local artifacts in this repo

- [`data/policy_snippets.json`](../data/policy_snippets.json) — hand-authored policy snippets for the **RAG** demo (not one of the datasets above).  
- [`data/download_kaggle.py`](../data/download_kaggle.py) — optional Kaggle download helper (`pip install -e ".[data]"`).  
- [`evaluation/`](../evaluation/) — utilities for stratified splits and offline metrics once labels are materialized from these corpora.

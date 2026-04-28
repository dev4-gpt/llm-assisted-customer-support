# Notebooks

## `llm_assist_showcase.ipynb`

**Single combined runbook** for coursework and demos (literature excerpts, dataset matrix, EDA, eval, stratified splits, TF–IDF baseline, optional transformer train, live API checks, tests, live metrics, regenerated PPTX/PDF):

- Assignment literature + dataset docs (`literature_review.md`, `docs/ASSIGNMENT_ALIGNMENT_RUNBOOK.md`)
- Optional Kaggle helper (`data/download_kaggle.py`) — documented inline; requires credentials
- EDA figures (`scripts/run_eda.py`)
- Offline / live evaluation (`scripts/run_offline_eval.py`, `evaluation/metrics.py`)
- Stratified split demo (`evaluation/splits.py`)
- Classical baseline (`scripts/train_encoder_classifier.py`) and optional **BERT/RoBERTa** (`scripts/train_triage_transformer.py`)
- Start the API with **uvicorn subprocess** and call endpoints
- Presentation assets (`scripts/generate_presentation_assets.py`)

### Setup

From the repo root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,eda,transformer,eval]"
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

### Run

Launch Jupyter however you prefer (classic, Lab, VS Code, Cursor), then open:

- `notebooks/llm_assist_showcase.ipynb`

### Notes / troubleshooting

- **Outputs**: all generated artifacts go under `artifacts/` (gitignored).
- **Ports**: the notebook starts uvicorn on `127.0.0.1:8000`. If that port is busy, stop the other process or change the port in the notebook cell.
- **Transformer training speed**: the demo uses **1 epoch** on a tiny generated CSV by default; replace `artifacts/demo_tickets.csv` with a real labeled dataset for meaningful accuracy.
- **LLM provider**: sections **5–8** (API smoke, live eval, deck/report) need a working LLM profile in `.env`. Earlier sections (EDA through **§4b** on `demo_tickets.csv`) can run without starting the server.


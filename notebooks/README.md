# Notebooks

## `llm_assist_showcase.ipynb`

End-to-end demo notebook:

- EDA figures (`scripts/run_eda.py`)
- Offline evaluation (`scripts/run_offline_eval.py`)
- Optional **BERT/RoBERTa** fine-tuning demo (`scripts/train_triage_transformer.py`)
- Start the API with **uvicorn subprocess** and call endpoints

### Setup

From the repo root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,eda,transformer]"
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

### Run

Launch Jupyter however you prefer (classic, Lab, VS Code, Cursor), then open:

- `notebooks/llm_assist_showcase.ipynb`

### Notes / troubleshooting

- **Outputs**: all generated artifacts go under `artifacts/` (gitignored).
- **Ports**: the notebook starts uvicorn on `127.0.0.1:8000`. If that port is busy, stop the other process or change the port in the notebook cell.
- **Transformer training speed**: the demo uses **1 epoch** on a tiny generated CSV by default; replace `artifacts/demo_tickets.csv` with a real labeled dataset for meaningful accuracy.
- **LLM provider**: the API calls require a working LLM backend (default is OpenRouter-compatible in `.env.example`). If you only want the offline/mock parts, you can run sections 2–4 without starting the server.


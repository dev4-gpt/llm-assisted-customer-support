# Data directory

- **`policy_snippets.json`** — Policy / FAQ snippets used by `RAGService` for retrieval-augmented context (demo scale).
- **`download_kaggle.py`** — Optional script to download Kaggle datasets with `kagglehub` after you configure Kaggle API credentials.

```bash
pip install -e ".[data]"
python data/download_kaggle.py --dataset thoughtvector/customer-support-on-twitter
```

See [`docs/DATASETS.md`](../docs/DATASETS.md) for full dataset documentation.

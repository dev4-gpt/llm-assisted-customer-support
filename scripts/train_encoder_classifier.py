#!/usr/bin/env python3
"""
Baseline supervised classifier for ticket text → category (or priority).

Uses TF–IDF character/word n-grams + logistic regression. Intended as a starting point
before transformer fine-tuning; keeps dependencies optional via `pip install -e ".[eval]"`.

Expected CSV columns (header row):
  text,category

Usage:
  python scripts/train_encoder_classifier.py --data data/raw/tickets_labeled.csv --out artifacts/triage_baseline.joblib
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train TF-IDF + logistic regression triage baseline.")
    parser.add_argument("--data", type=Path, required=True, help="CSV with text,category columns")
    parser.add_argument("--out", type=Path, default=Path("artifacts/triage_baseline.joblib"))
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    try:
        import joblib
        import pandas as pd
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import classification_report
        from sklearn.model_selection import train_test_split
        from sklearn.pipeline import Pipeline
    except ImportError as exc:
        raise SystemExit('Install eval extras: pip install -e ".[eval]"') from exc

    df = pd.read_csv(args.data)
    if "text" not in df.columns or "category" not in df.columns:
        raise SystemExit("CSV must contain columns: text, category")

    x_train, x_test, y_train, y_test = train_test_split(
        df["text"].astype(str),
        df["category"].astype(str),
        test_size=args.test_size,
        random_state=args.seed,
        stratify=df["category"],
    )

    pipe = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=50_000,
                    ngram_range=(1, 2),
                    min_df=2,
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    max_iter=200,
                    class_weight="balanced",
                    random_state=args.seed,
                ),
            ),
        ]
    )
    pipe.fit(x_train, y_train)
    preds = pipe.predict(x_test)
    report = classification_report(y_test, preds, output_dict=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, args.out)
    metrics_path = args.out.with_suffix(".metrics.json")
    metrics_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved model to {args.out.resolve()}")
    print(f"Saved metrics to {metrics_path.resolve()}")


if __name__ == "__main__":
    main()

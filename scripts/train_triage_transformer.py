#!/usr/bin/env python3
"""
Fine-tune BERT / RoBERTa (encoder) for ticket text → primary category.

Same CSV contract as ``train_encoder_classifier.py``:
  text,category

Saves a Hugging Face checkpoint directory (config, tokenizer, weights) for use with
``TRIAGE_TRANSFORMER_ENABLED`` + ``TRIAGE_TRANSFORMER_MODEL_DIR``.

Usage:
  pip install -e ".[transformer]"
  python scripts/train_triage_transformer.py \\
    --data data/raw/tickets_labeled.csv \\
    --out artifacts/triage_roberta \\
    --model roberta-base

  # Smaller / faster (course demo):
  python scripts/train_triage_transformer.py --data ... --model bert-base-uncased --epochs 2
"""
from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path

# Valid routing labels (must match app.models.domain.Category)
_CATEGORY_VALUES = frozenset(
    {
        "billing",
        "authentication",
        "technical_bug",
        "feature_request",
        "general_inquiry",
    }
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fine-tune BERT/RoBERTa for ticket category (HF Trainer).",
    )
    parser.add_argument("--data", type=Path, required=True, help="CSV with text,category")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("artifacts/triage_transformer"),
        help="Output directory (HF save_pretrained)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="roberta-base",
        help="HF model id, e.g. roberta-base, bert-base-uncased",
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warn-unknown-labels", action="store_true")
    args = parser.parse_args()

    try:
        import numpy as np
        import pandas as pd
        import torch
        from sklearn.metrics import accuracy_score
        from sklearn.model_selection import train_test_split
        from torch.utils.data import Dataset
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            Trainer,
            TrainingArguments,
        )
    except ImportError as exc:
        raise SystemExit('Install transformer extras: pip install -e ".[transformer]"') from exc

    df = pd.read_csv(args.data)
    if "text" not in df.columns or "category" not in df.columns:
        raise SystemExit("CSV must contain columns: text, category")

    texts = df["text"].astype(str).tolist()
    labels_raw = df["category"].astype(str).tolist()
    unknown = sorted({lab for lab in labels_raw if lab not in _CATEGORY_VALUES})
    if unknown:
        msg = f"Unknown category labels (expected {_CATEGORY_VALUES}): {unknown}"
        if args.warn_unknown_labels:
            print(f"WARNING: {msg}")
        else:
            raise SystemExit(msg + "\nUse --warn-unknown-labels to skip invalid rows.")

    pairs = [
        (t, lab)
        for t, lab in zip(texts, labels_raw, strict=True)
        if lab in _CATEGORY_VALUES
    ]
    if not pairs:
        raise SystemExit("No rows left after filtering to known categories.")
    texts_f = [p[0] for p in pairs]
    labels_f = [p[1] for p in pairs]

    sorted_labels = sorted(_CATEGORY_VALUES)
    label2id = {lab: i for i, lab in enumerate(sorted_labels)}
    id2label = {i: lab for lab, i in label2id.items()}
    y = np.array([label2id[lab] for lab in labels_f], dtype=np.int64)

    try:
        x_train, x_val, y_train, y_val = train_test_split(
            texts_f,
            y,
            test_size=args.test_size,
            random_state=args.seed,
            stratify=y,
        )
    except ValueError:
        # Too few samples per class for stratification
        x_train, x_val, y_train, y_val = train_test_split(
            texts_f,
            y,
            test_size=args.test_size,
            random_state=args.seed,
        )

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id,
    )

    class _EncDataset(Dataset):
        def __init__(self, xs: list[str], ys: np.ndarray) -> None:
            self._xs = xs
            self._ys = ys

        def __len__(self) -> int:
            return len(self._xs)

        def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
            enc = tokenizer(
                self._xs[idx],
                truncation=True,
                max_length=args.max_length,
                padding="max_length",
                return_tensors="pt",
            )
            return {
                "input_ids": enc["input_ids"].squeeze(0),
                "attention_mask": enc["attention_mask"].squeeze(0),
                "labels": torch.tensor(int(self._ys[idx]), dtype=torch.long),
            }

    train_ds = _EncDataset(x_train, y_train)
    val_ds = _EncDataset(x_val, y_val)

    def compute_metrics(eval_pred: tuple) -> dict[str, float]:
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {"accuracy": float(accuracy_score(labels, preds))}

    args.out.mkdir(parents=True, exist_ok=True)
    targs = TrainingArguments(
        output_dir=str(args.out),
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        save_total_limit=1,
        seed=args.seed,
        logging_steps=10,
        report_to=[],
    )

    trainer_kwargs = {
        "model": model,
        "args": targs,
        "train_dataset": train_ds,
        "eval_dataset": val_ds,
        "compute_metrics": compute_metrics,
    }
    # transformers API changed across versions:
    # older Trainer accepts `tokenizer`, newer releases removed it in favor of
    # `processing_class`. Detect at runtime for compatibility.
    trainer_params = set(inspect.signature(Trainer.__init__).parameters.keys())
    if "tokenizer" in trainer_params:
        trainer_kwargs["tokenizer"] = tokenizer
    elif "processing_class" in trainer_params:
        trainer_kwargs["processing_class"] = tokenizer

    trainer = Trainer(**trainer_kwargs)
    trainer.train()

    trainer.save_model(str(args.out))
    tokenizer.save_pretrained(str(args.out))
    metrics = trainer.evaluate()
    (args.out / "train_metrics.json").write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )
    print(f"Saved checkpoint to {args.out.resolve()}")
    print(f"Metrics: {metrics}")


if __name__ == "__main__":
    main()

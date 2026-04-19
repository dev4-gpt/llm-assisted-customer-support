"""
Optional fine-tuned BERT / RoBERTa category hint for triage.

Loads a Hugging Face sequence-classification checkpoint produced by
``scripts/train_triage_transformer.py``. Requires ``pip install -e ".[transformer]"``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from app.core.logging import get_logger

logger = get_logger(__name__)


class TriageTransformerPredictor:
    """Runs a single forward pass; CPU / CUDA / MPS when available."""

    def __init__(self, model_dir: Path, *, max_length: int = 256) -> None:
        self._model_dir = model_dir
        self._max_length = max_length
        self._tokenizer: Any = None
        self._model: Any = None
        self._device: Any = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(self._model_dir)
        self._model = AutoModelForSequenceClassification.from_pretrained(self._model_dir)
        if torch.cuda.is_available():
            self._device = torch.device("cuda")
        elif (
            getattr(torch.backends, "mps", None) is not None
            and torch.backends.mps.is_available()
        ):
            self._device = torch.device("mps")
        else:
            self._device = torch.device("cpu")
        self._model.to(self._device)
        self._model.eval()

    def predict_category(self, text: str) -> str:
        import torch

        self._ensure_loaded()
        assert self._tokenizer is not None and self._model is not None

        enc = self._tokenizer(
            text[:8000],
            return_tensors="pt",
            truncation=True,
            max_length=self._max_length,
            padding=True,
        )
        enc = {k: v.to(self._device) for k, v in enc.items()}
        with torch.no_grad():
            logits = self._model(**enc).logits
        pred_id = int(logits.argmax(dim=-1)[0].item())
        id2label = dict(self._model.config.id2label)
        if pred_id in id2label:
            return str(id2label[pred_id])
        return str(id2label[str(pred_id)])


def load_triage_transformer(model_dir: Path) -> TriageTransformerPredictor | None:
    """
    Return a predictor, or None if optional dependencies or weights are missing.
    """
    if not model_dir.is_dir():
        logger.warning("triage_transformer_model_dir is not a directory", path=str(model_dir))
        return None
    cfg = model_dir / "config.json"
    if not cfg.is_file():
        logger.warning("No config.json in transformer model dir", path=str(model_dir))
        return None
    try:
        return TriageTransformerPredictor(model_dir)
    except Exception:
        logger.exception("Failed to load triage transformer", path=str(model_dir))
        return None

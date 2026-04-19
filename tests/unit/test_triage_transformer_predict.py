"""
tests/unit/test_triage_transformer_predict.py
─────────────────────────────────────────────
Optional HF predictor — torch/transformers mocked (no .[transformer] install in CI).
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from app.services.triage_transformer_predict import (
    TriageTransformerPredictor,
    load_triage_transformer,
)


def test_load_returns_none_if_path_not_dir(tmp_path: Path) -> None:
    assert load_triage_transformer(tmp_path / "nope") is None


def test_load_returns_none_if_no_config(tmp_path: Path) -> None:
    assert load_triage_transformer(tmp_path) is None


def test_load_returns_predictor_when_config_exists(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    p = load_triage_transformer(tmp_path)
    assert p is not None
    assert isinstance(p, TriageTransformerPredictor)


def test_load_returns_none_on_unexpected_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")

    class _Boom:
        def __init__(self, *_a: object, **_k: object) -> None:
            raise RuntimeError("fail")

    monkeypatch.setattr(
        "app.services.triage_transformer_predict.TriageTransformerPredictor",
        _Boom,
    )
    assert load_triage_transformer(tmp_path) is None


def test_predict_category_with_stub_torch_transformers(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")

    tensor_val = MagicMock()
    tensor_val.item.return_value = 0
    argmax_row = MagicMock()
    argmax_row.__getitem__.return_value = tensor_val
    logits = MagicMock()
    logits.argmax.return_value = argmax_row

    forward_out = MagicMock()
    forward_out.logits = logits

    model = MagicMock(return_value=forward_out)
    model.config.id2label = {0: "billing"}
    model.to.return_value = model
    model.eval = MagicMock()

    tok_tensor = MagicMock()
    tok_tensor.to.return_value = tok_tensor
    tokenizer = MagicMock(
        return_value={"input_ids": tok_tensor, "attention_mask": tok_tensor}
    )

    fake_torch = MagicMock()
    fake_torch.cuda.is_available.return_value = False
    fake_torch.backends.mps.is_available.return_value = False
    fake_torch.device.return_value = "cpu"
    fake_torch.no_grad.return_value.__enter__ = MagicMock(return_value=None)
    fake_torch.no_grad.return_value.__exit__ = MagicMock(return_value=None)

    fake_transformers = MagicMock()
    fake_transformers.AutoTokenizer.from_pretrained.return_value = tokenizer
    fake_transformers.AutoModelForSequenceClassification.from_pretrained.return_value = model

    prev_torch = sys.modules.get("torch")
    prev_tf = sys.modules.get("transformers")
    try:
        sys.modules["torch"] = fake_torch
        sys.modules["transformers"] = fake_transformers

        p = TriageTransformerPredictor(tmp_path)
        assert p.predict_category("refund my card") == "billing"

        model.config.id2label = {"0": "authentication"}
        tensor_val.item.return_value = 0
        assert p.predict_category("login") == "authentication"
    finally:
        if prev_torch is None:
            sys.modules.pop("torch", None)
        else:
            sys.modules["torch"] = prev_torch
        if prev_tf is None:
            sys.modules.pop("transformers", None)
        else:
            sys.modules["transformers"] = prev_tf

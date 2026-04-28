#!/usr/bin/env python3
"""
Offline evaluation: triage accuracy (vs gold labels), quality mean score, summarization ROUGE-L.

  Mock mode (default, CI-safe): QueueLLM returns gold-aligned JSON → expect perfect triage match.
  Live mode: EVAL_LLM=1 and working LLM credentials → real model metrics.

Usage:
  python scripts/run_offline_eval.py --mock --data data/golden/eval_set.jsonl
  EVAL_LLM=1 python scripts/run_offline_eval.py --data data/golden/eval_set.jsonl
  python scripts/run_offline_eval.py --mock --baseline-model artifacts/triage_baseline.joblib
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Repo root on path
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


@dataclass
class EvalRow:
    task: str
    raw: dict[str, Any]


def load_jsonl(path: Path) -> list[EvalRow]:
    rows: list[EvalRow] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"{path}:{line_no}: invalid JSON: {exc}") from exc
        task = obj.get("task")
        if task not in ("triage", "quality", "summarize"):
            raise SystemExit(f"{path}:{line_no}: unknown task {task!r}")
        rows.append(EvalRow(task=task, raw=obj))
    return rows


class QueueLLM:
    """Minimal LLM stand-in for mock eval (same interface as LLMClient.complete_json)."""

    def __init__(self) -> None:
        self._q: deque[dict[str, Any]] = deque()

    def enqueue(self, payload: dict[str, Any]) -> None:
        self._q.append(payload)

    def complete_json(self, user_prompt: str, *, schema_hint: str = "") -> dict[str, Any]:
        if not self._q:
            raise RuntimeError("QueueLLM: no queued response (schema_hint=%r)" % schema_hint)
        return self._q.popleft()


def mock_payload_for_row(row: EvalRow) -> dict[str, Any]:
    r = row.raw
    if row.task == "triage":
        cat = r["gold_category"]
        pri = r["gold_priority"]
        return {
            "priority": pri,
            "category": cat,
            "intents": [{"label": cat, "score": 0.95}],
            "sentiment_score": -0.2,
            "rationale": "Mock gold-aligned triage for offline evaluation.",
            "confidence": 0.93,
        }
    if row.task == "quality":
        return {
            "score": 0.82,
            "checks": {
                "empathetic_tone": True,
                "actionable_next_step": True,
                "policy_safety": True,
                "resolved_or_escalated": True,
            },
            "coaching_feedback": "Mock quality evaluation for offline benchmark.",
            "flagged_phrases": [],
        }
    if row.task == "summarize":
        return {
            "summary": r["gold_summary"],
            "key_points": ["Issue reported", "Agent provided steps"],
            "confidence": 0.9,
        }
    raise ValueError(row.task)


def run_mock_eval(
    rows: list[EvalRow],
    *,
    baseline_model: Path | None,
) -> dict[str, Any]:
    from app.core.config import Settings
    from app.models.domain import DialogTurn, DialogRole, SummarizeRequest, TriageRequest
    from app.models.domain import QualityRequest
    from app.services.quality_service import QualityService
    from app.services.rag_service import RAGService
    from app.services.summarization_service import SummarizationService
    from app.services.triage_service import TriageService
    from evaluation.metrics import classification_report_dict, rouge_l_f1

    settings = Settings()
    llm = QueueLLM()
    rag = RAGService(settings)
    triage_svc = TriageService(llm, settings, rag)
    quality_svc = QualityService(llm, settings, rag)
    summ_svc = SummarizationService(llm)

    y_true_pri: list[str] = []
    y_pred_pri: list[str] = []
    y_true_cat: list[str] = []
    y_pred_cat: list[str] = []
    quality_scores: list[float] = []
    rouge_scores: list[float] = []

    for row in rows:
        payload = mock_payload_for_row(row)
        llm.enqueue(payload)

        if row.task == "triage":
            req = TriageRequest(ticket_text=row.raw["ticket_text"])
            out = triage_svc.triage(req)
            y_true_pri.append(row.raw["gold_priority"])
            y_pred_pri.append(out.priority.value)
            y_true_cat.append(row.raw["gold_category"])
            y_pred_cat.append(out.category.value)
        elif row.task == "quality":
            req = QualityRequest(
                ticket_text=row.raw["ticket_text"],
                agent_response=row.raw["agent_response"],
            )
            out = quality_svc.evaluate(req)
            quality_scores.append(out.score)
        elif row.task == "summarize":
            turns = [
                DialogTurn(role=DialogRole(t["role"]), content=t["content"])
                for t in row.raw["turns"]
            ]
            req = SummarizeRequest(turns=turns)
            out = summ_svc.summarize(req)
            rouge_scores.append(rouge_l_f1(row.raw["gold_summary"], out.summary))

    result: dict[str, Any] = {
        "mode": "mock",
        "counts": {
            "triage": sum(1 for r in rows if r.task == "triage"),
            "quality": sum(1 for r in rows if r.task == "quality"),
            "summarize": sum(1 for r in rows if r.task == "summarize"),
        },
    }

    if y_true_pri:
        result["triage_priority"] = classification_report_dict(y_true_pri, y_pred_pri)
        result["triage_category"] = classification_report_dict(y_true_cat, y_pred_cat)
    if quality_scores:
        result["quality"] = {"mean_score": sum(quality_scores) / len(quality_scores), "n": len(quality_scores)}
    if rouge_scores:
        result["summarize"] = {"mean_rouge_l_f1": sum(rouge_scores) / len(rouge_scores), "n": len(rouge_scores)}

    if baseline_model and y_true_cat:
        try:
            import joblib
        except ImportError as exc:
            raise SystemExit("Install eval extras for baseline: pip install -e '.[eval]'") from exc
        pipe = joblib.load(baseline_model)
        texts = [r.raw["ticket_text"] for r in rows if r.task == "triage"]
        gold = [r.raw["gold_category"] for r in rows if r.task == "triage"]
        preds = [str(p) for p in pipe.predict(texts)]
        result["baseline_triage_category"] = classification_report_dict(gold, preds)
        result["baseline_model"] = str(baseline_model.resolve())

    return result


def run_live_eval(rows: list[EvalRow]) -> dict[str, Any]:
    from app.core.config import Settings
    from app.models.domain import DialogTurn, DialogRole, QualityRequest, SummarizeRequest, TriageRequest
    from app.services.llm_client import LLMClient
    from app.services.quality_service import QualityService
    from app.services.rag_service import RAGService
    from app.services.summarization_service import SummarizationService
    from app.services.triage_service import TriageService
    from evaluation.metrics import classification_report_dict, rouge_l_f1

    settings = Settings()
    llm = LLMClient(settings)
    rag = RAGService(settings)
    triage_svc = TriageService(llm, settings, rag)
    quality_svc = QualityService(llm, settings, rag)
    summ_svc = SummarizationService(llm)

    y_true_pri: list[str] = []
    y_pred_pri: list[str] = []
    y_true_cat: list[str] = []
    y_pred_cat: list[str] = []
    quality_scores: list[float] = []
    rouge_scores: list[float] = []
    errors: list[str] = []

    for row in rows:
        try:
            if row.task == "triage":
                req = TriageRequest(ticket_text=row.raw["ticket_text"])
                out = triage_svc.triage(req)
                y_true_pri.append(row.raw["gold_priority"])
                y_pred_pri.append(out.priority.value)
                y_true_cat.append(row.raw["gold_category"])
                y_pred_cat.append(out.category.value)
            elif row.task == "quality":
                req = QualityRequest(
                    ticket_text=row.raw["ticket_text"],
                    agent_response=row.raw["agent_response"],
                )
                out = quality_svc.evaluate(req)
                quality_scores.append(out.score)
            elif row.task == "summarize":
                turns = [
                    DialogTurn(role=DialogRole(t["role"]), content=t["content"])
                    for t in row.raw["turns"]
                ]
                req = SummarizeRequest(turns=turns)
                out = summ_svc.summarize(req)
                rouge_scores.append(rouge_l_f1(row.raw["gold_summary"], out.summary))
        except Exception as exc:  # noqa: BLE001 — aggregate eval failures
            errors.append(f"{row.raw.get('id', '?')}: {exc}")

    result: dict[str, Any] = {
        "mode": "live",
        "counts": {
            "triage": sum(1 for r in rows if r.task == "triage"),
            "quality": sum(1 for r in rows if r.task == "quality"),
            "summarize": sum(1 for r in rows if r.task == "summarize"),
        },
        "errors": errors,
    }
    if y_true_pri:
        result["triage_priority"] = classification_report_dict(y_true_pri, y_pred_pri)
        result["triage_category"] = classification_report_dict(y_true_cat, y_pred_cat)
    if quality_scores:
        result["quality"] = {"mean_score": sum(quality_scores) / len(quality_scores), "n": len(quality_scores)}
    if rouge_scores:
        result["summarize"] = {"mean_rouge_l_f1": sum(rouge_scores) / len(rouge_scores), "n": len(rouge_scores)}
    return result


def metrics_to_markdown(data: dict[str, Any]) -> str:
    lines = [
        "# Offline evaluation summary",
        "",
        f"- **Mode:** {data.get('mode', '?')}",
        "",
    ]
    if data.get("errors"):
        lines.append("## Errors")
        for e in data["errors"]:
            lines.append(f"- {e}")
        lines.append("")
    if "triage_category" in data:
        tc = data["triage_category"]
        lines.append("## Triage category")
        lines.append(f"- accuracy: **{tc['accuracy']:.4f}**")
        lines.append(f"- micro F1: **{tc.get('micro_f1', 0.0):.4f}**")
        lines.append(f"- macro F1: **{tc.get('macro_f1', 0.0):.4f}**")
        lines.append("- minority-class performance:")
        for lab, stats in tc.get("minority_class_performance", {}).items():
            lines.append(
                f"  - {lab}: f1={stats['f1']:.4f}, precision={stats['precision']:.4f}, "
                f"recall={stats['recall']:.4f}, support={int(stats['support'])}"
            )
        lines.append("- confusion matrix (true -> predicted counts):")
        for true_lab, row in tc.get("confusion_matrix", {}).items():
            pairs = ", ".join(f"{pred_lab}:{count}" for pred_lab, count in row.items())
            lines.append(f"  - {true_lab}: {pairs}")
        lines.append("")
    if "triage_priority" in data:
        tp = data["triage_priority"]
        lines.append("## Triage priority")
        lines.append(f"- accuracy: **{tp['accuracy']:.4f}**")
        lines.append(f"- micro F1: **{tp.get('micro_f1', 0.0):.4f}**")
        lines.append(f"- macro F1: **{tp.get('macro_f1', 0.0):.4f}**")
        lines.append("")
    if "quality" in data:
        q = data["quality"]
        lines.append("## Quality")
        lines.append(f"- mean_score: **{q['mean_score']:.4f}** (n={q['n']})")
        lines.append("")
    if "summarize" in data:
        s = data["summarize"]
        lines.append("## Summarization")
        lines.append(f"- mean ROUGE-L F1: **{s['mean_rouge_l_f1']:.4f}** (n={s['n']})")
        lines.append("")
    if "baseline_triage_category" in data:
        b = data["baseline_triage_category"]
        lines.append("## Baseline (TF–IDF + LR) category")
        lines.append(f"- accuracy: **{b['accuracy']:.4f}**")
        lines.append("")
    lines.append(
        "_LLM-as-judge and single-reference ROUGE are approximate; "
        "use for regression tracking, not sole ground truth._"
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run offline eval on golden JSONL.")
    parser.add_argument("--data", type=Path, default=_ROOT / "data/golden/eval_set.jsonl")
    parser.add_argument(
        "--mock",
        action="store_true",
        default=None,
        help="Force QueueLLM. Default is mock unless EVAL_LLM=1.",
    )
    parser.add_argument("--output-dir", type=Path, default=_ROOT / "artifacts/eval")
    parser.add_argument(
        "--baseline-model",
        type=Path,
        default=None,
        help="Optional joblib pipeline from train_encoder_classifier.py (triage category only)",
    )
    args = parser.parse_args()

    live = os.environ.get("EVAL_LLM", "").strip().lower() in ("1", "true", "yes")
    use_mock = True if args.mock is True else (not live)

    if not args.data.is_file():
        raise SystemExit(f"Data file not found: {args.data}")

    rows = load_jsonl(args.data)
    if not rows:
        raise SystemExit("No rows in eval set")

    if use_mock:
        result = run_mock_eval(rows, baseline_model=args.baseline_model)
    else:
        result = run_live_eval(rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "metrics.json"
    md_path = args.output_dir / "summary.md"
    json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    md_path.write_text(metrics_to_markdown(result), encoding="utf-8")
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()

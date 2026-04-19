# Golden evaluation set

**Lineage:** Synthetic examples authored for this repository (no third-party text). Safe for public CI and reproducible benchmarks.

**Format:** `eval_set.jsonl` — one JSON object per line. See `task` field:

| `task` | Required fields | Gold labels |
|--------|-----------------|-------------|
| `triage` | `id`, `ticket_text`, `gold_priority`, `gold_category` | Compared to `TriageService` output |
| `quality` | `id`, `ticket_text`, `agent_response` | Reported: mean score (no human gold) |
| `summarize` | `id`, `turns`, `gold_summary` | ROUGE-L vs model summary |

Used by [`scripts/run_offline_eval.py`](../../scripts/run_offline_eval.py).

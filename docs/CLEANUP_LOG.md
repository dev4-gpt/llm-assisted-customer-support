# Cleanup Log

This cleanup removed only generated/local clutter and preserved all source, tests, docs, and demo artifacts.

## Removed

- `.DS_Store` (macOS metadata)
- `.coverage` (coverage data file)
- `.pytest_cache/` (pytest cache)
- `.mypy_cache/` (mypy cache)
- `.ruff_cache/` (ruff cache)
- `support_triage.egg-info/` (setuptools build/install metadata)

## Kept intentionally

- `.env.example` (required setup template)
- `.env` (local runtime secrets/config; not for commit)
- `.gitignore`, `pyproject.toml`, `.github/workflows/ci.yml` (project reproducibility/quality)
- `artifacts/` (demo outputs and evidence)
- `notebooks/llm_assist_showcase.ipynb` (presentation flow)
- all `app/`, `tests/`, `scripts/`, `evaluation/`, `data/`, and `docs/` source-of-truth content

## Rationale

The goal was a presentation-clean repository without removing anything needed for:
- reproducible setup,
- API + notebook demo execution,
- test validation,
- report defense evidence.

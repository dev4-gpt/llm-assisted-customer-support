#!/usr/bin/env python3
"""
Exploratory data analysis (EDA) plots for the golden eval set and/or a labeled ticket CSV.

Default input: ``data/golden/eval_set.jsonl`` (always present in-repo).

Outputs PNGs under ``artifacts/eda/`` (gitignored). 
Install: ``pip install -e ".[eda]"``.

Examples:
  python scripts/run_eda.py
  python scripts/run_eda.py --csv data/raw/tickets_labeled.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _savefig(path: Path) -> None:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_golden_eda(df: pd.DataFrame, out_dir: Path) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="whitegrid", context="talk")

    # Task distribution
    if "task" in df.columns:
        plt.figure(figsize=(8, 5))
        vc = df["task"].value_counts()
        sns.barplot(x=vc.values, y=vc.index.astype(str), color="steelblue")
        plt.xlabel("Count")
        plt.ylabel("task")
        plt.title("Golden eval set: rows by task")
        _savefig(out_dir / "golden_task_counts.png")

    tri = df[df["task"] == "triage"].copy() if "task" in df.columns else pd.DataFrame()
    if not tri.empty and "gold_category" in tri.columns:
        plt.figure(figsize=(9, 5))
        cvc = tri["gold_category"].value_counts()
        sns.barplot(x=cvc.values, y=cvc.index.astype(str), color="seagreen")
        plt.xlabel("Count")
        plt.title("Triage rows: gold category (balanced demo set)")
        _savefig(out_dir / "golden_triage_category.png")

    if not tri.empty and "gold_priority" in tri.columns:
        p_order = ["critical", "high", "medium", "low"]
        tri["gold_priority"] = pd.Categorical(tri["gold_priority"], categories=p_order, ordered=True)
        plt.figure(figsize=(8, 4))
        pvc = tri["gold_priority"].value_counts().reindex(p_order).fillna(0)
        sns.barplot(x=pvc.index.astype(str), y=pvc.values, color="coral")
        plt.title("Triage rows: gold priority")
        plt.xlabel("gold_priority")
        plt.ylabel("Count")
        plt.xticks(rotation=20)
        _savefig(out_dir / "golden_triage_priority.png")

    if "ticket_text" in df.columns:
        tdf = df[df["ticket_text"].notna()].copy()
        if not tdf.empty:
            tdf["char_len"] = tdf["ticket_text"].astype(str).str.len()
            plt.figure(figsize=(9, 5))
            tasks = [str(t) for t in tdf["task"].dropna().unique().tolist()]
            series = [tdf.loc[tdf["task"] == t, "char_len"] for t in tasks]
            if series:
                plt.hist(series, bins=12, stacked=True, label=tasks, alpha=0.85)
                plt.legend(title="task")
            plt.title("Ticket text length (characters) by task (rows with ticket_text)")
            _savefig(out_dir / "golden_text_length_by_task.png")


def plot_labeled_csv_eda(df: pd.DataFrame, out_dir: Path) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="whitegrid", context="talk")

    plt.figure(figsize=(10, max(4, 0.35 * df["category"].nunique())))
    cvc = df["category"].value_counts()
    sns.barplot(x=cvc.values, y=cvc.index.astype(str), color="teal")
    plt.xlabel("Count")
    plt.title("Labeled tickets: category frequency (check imbalance)")
    _savefig(out_dir / "csv_category_counts.png")

    plt.figure(figsize=(8, 5))
    sns.histplot(data=df, x="char_len", bins=30, kde=True, color="steelblue")
    plt.title("Labeled tickets: text length (characters)")
    _savefig(out_dir / "csv_text_length.png")


def main() -> None:
    parser = argparse.ArgumentParser(description="EDA plots for golden JSONL and/or labeled CSV.")
    parser.add_argument(
        "--golden",
        type=Path,
        default=Path("data/golden/eval_set.jsonl"),
        help="Path to eval_set.jsonl",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Optional labeled CSV (columns: text, category)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("artifacts/eda"),
        help="Output directory for PNG figures",
    )
    args = parser.parse_args()

    try:
        import matplotlib  # noqa: F401
        import seaborn  # noqa: F401
    except ImportError as exc:
        raise SystemExit('Install EDA extras: pip install -e ".[eda]"') from exc

    from evaluation.eda_loaders import load_golden_eval_jsonl, load_labeled_tickets_csv

    if not args.golden.is_file():
        raise SystemExit(f"Golden file not found: {args.golden}")

    gdf = load_golden_eval_jsonl(args.golden)
    plot_golden_eda(gdf, args.out)
    print(f"Wrote golden-set figures under {args.out.resolve()}")

    if args.csv is not None:
        if not args.csv.is_file():
            raise SystemExit(f"CSV not found: {args.csv}")
        cdf = load_labeled_tickets_csv(args.csv)
        plot_labeled_csv_eda(cdf, args.out)
        print(f"Wrote labeled-CSV figures under {args.out.resolve()}")


if __name__ == "__main__":
    main()

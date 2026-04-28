#!/usr/bin/env python3
"""
Multi-corpus dataset integration for the LLM-Assisted Customer Support system.

Processes all 6 datasets and writes normalized CSVs under data/processed/:

  1. twitter_support_sample.csv  — 80-100k stratified tweet-support samples (Dataset 1)
  2. news_summary_sample.csv     — article/headline pairs for summarization transfer (Dataset 3)
  3. amazon_csat.csv             — review text + binary sentiment label (Dataset 4)
  4. tickets_labeled.csv         — already built by prepare_real_dataset.py (Dataset 5)
  5. maia_emotion.csv            — per-turn emotion + quality labels (Dataset 6)
  Note: TWEETSUMM (Dataset 2) requires author contact per the paper; excluded here.

Output columns per file are documented below.

Usage:
  python scripts/prepare_all_datasets.py
  python scripts/prepare_all_datasets.py --datasets twitter news amazon maia
"""
from __future__ import annotations

import argparse
import bz2
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"

AVAILABLE = ["twitter", "news", "amazon", "maia"]


# ─────────────────────────────────────────────────────────────────────────────
# Dataset 1: Customer Support on Twitter (ThoughtVector)
# Cols: text, inbound, author_id, response_tweet_id, in_response_to_tweet_id
# ─────────────────────────────────────────────────────────────────────────────
def prepare_twitter(n_sample: int = 90_000) -> Path:
    import pandas as pd

    src = RAW / "twitter_support" / "twcs.csv"
    if not src.is_file():
        raise FileNotFoundError(f"Missing: {src}\nRun: python data/download_kaggle.py --dataset thoughtvector/customer-support-on-twitter")

    print(f"\n[Twitter] Loading {src} …")
    df = pd.read_csv(src, dtype=str)
    print(f"  Total rows: {len(df):,}")

    # Keep inbound (customer) messages only with real text
    customer = df[df["inbound"] == "True"].copy()
    customer = customer[customer["text"].notna() & (customer["text"].str.len() >= 10)].copy()

    # Stratified sample by company (author of the *reply*) to get diverse topics
    sample_size = min(n_sample, len(customer))
    sampled = customer.sample(sample_size, random_state=42)

    # Derive a simple sentiment label: presence of negative keywords
    NEG = ["not working", "broken", "fail", "issue", "problem", "error", "worst",
           "terrible", "awful", "never", "refuse", "useless", "can't", "cannot"]
    sampled["sentiment_label"] = sampled["text"].str.lower().apply(
        lambda t: "negative" if any(kw in t for kw in NEG) else "neutral"
    )

    out = pd.DataFrame({
        "text": sampled["text"].str.strip(),
        "tweet_id": sampled["tweet_id"],
        "in_response_to": sampled["in_response_to_tweet_id"],
        "sentiment_label": sampled["sentiment_label"],
        "source": "twitter_support",
    })

    dest = PROCESSED / "twitter_support_sample.csv"
    out.to_csv(dest, index=False)
    print(f"  Saved {len(out):,} rows → {dest}")
    print(f"  Sentiment distribution:\n{out['sentiment_label'].value_counts()}")
    return dest


# ─────────────────────────────────────────────────────────────────────────────
# Dataset 3: News Summary (Kaggle) — auxiliary abstractive pairs
# Cols: text (article), headlines (short summary), source
# ─────────────────────────────────────────────────────────────────────────────
def prepare_news(n_sample: int = 20_000) -> Path:
    import pandas as pd

    candidates = list((RAW / "news_summary").glob("news_summary*.csv"))
    if not candidates:
        raise FileNotFoundError(f"Missing news_summary CSVs in {RAW / 'news_summary'}")

    dfs = []
    for c in candidates:
        try:
            dfs.append(pd.read_csv(c, encoding="latin-1"))
        except Exception as e:
            print(f"  Warning: could not read {c.name}: {e}")

    df = pd.concat(dfs, ignore_index=True)
    print(f"\n[News Summary] Total rows: {len(df):,}, cols: {list(df.columns)}")

    # Column names vary — map text/headline columns
    text_col = next((c for c in df.columns if "text" in c.lower()), None)
    head_col = next((c for c in df.columns if "head" in c.lower() or "short" in c.lower()), None)
    if text_col is None or head_col is None:
        print(f"  Columns: {list(df.columns)}")
        raise ValueError("Cannot identify text/headline columns")

    df = df[[text_col, head_col]].dropna()
    df = df[df[text_col].str.len() >= 50].copy()

    sample_size = min(n_sample, len(df))
    sampled = df.sample(sample_size, random_state=42)

    out = pd.DataFrame({
        "text": sampled[text_col].str.strip(),
        "summary": sampled[head_col].str.strip(),
        "source": "news_summary",
    })

    dest = PROCESSED / "news_summary_sample.csv"
    out.to_csv(dest, index=False)
    print(f"  Saved {len(out):,} rows → {dest}")
    return dest


# ─────────────────────────────────────────────────────────────────────────────
# Dataset 4: Amazon Reviews — satisfaction proxy
# Format: __label__1 or __label__2 followed by review text (FastText .bz2)
# ─────────────────────────────────────────────────────────────────────────────
def prepare_amazon(n_sample: int = 30_000) -> Path:
    import pandas as pd

    src = RAW / "amazon_reviews" / "test.ft.txt.bz2"   # smaller file
    if not src.is_file():
        raise FileNotFoundError(f"Missing: {src}")

    print(f"\n[Amazon Reviews] Streaming {src} …")
    rows = []
    with bz2.open(src, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            label_token = line.split(" ")[0]
            text = line[len(label_token):].strip()
            sentiment = "positive" if label_token == "__label__2" else "negative"
            if len(text) >= 20:
                rows.append({"text": text, "sentiment_label": sentiment, "source": "amazon_reviews"})
            if len(rows) >= n_sample:
                break

    df = pd.DataFrame(rows)
    # Balance classes
    n_per = min(df["sentiment_label"].value_counts().min(), n_sample // 2)
    frames = [g.sample(n_per, random_state=42) for _, g in df.groupby("sentiment_label")]
    out = pd.concat(frames, ignore_index=True).sample(frac=1, random_state=42)

    dest = PROCESSED / "amazon_csat.csv"
    out.to_csv(dest, index=False)
    print(f"  Saved {len(out):,} rows → {dest}")
    print(f"  Sentiment distribution:\n{out['sentiment_label'].value_counts()}")
    return dest


# ─────────────────────────────────────────────────────────────────────────────
# Dataset 6: MAIA-DQE — emotion + dialogue quality
# Flattens turns into per-sentence rows with emotion, engagement, correctness
# ─────────────────────────────────────────────────────────────────────────────
def prepare_maia() -> Path:
    import pandas as pd

    emotion_dir = RAW / "maia_dqe" / "repo" / "Emotion" / "data" / "maia"
    if not emotion_dir.is_dir():
        raise FileNotFoundError(f"Missing MAIA-DQE repo at {RAW / 'maia_dqe'}\nRun: git clone https://github.com/johndmendonca/MAIA-DQE data/raw/maia_dqe/repo")

    print(f"\n[MAIA-DQE] Loading emotion annotation files from {emotion_dir} …")
    rows = []
    # _splits_a.json files are index files (just dialogue IDs). Only load *_client_* files.
    for json_file in sorted(emotion_dir.glob("*_client_*_a.json")):
        try:
            dialogues = json.loads(json_file.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"  Warning: {json_file.name}: {e}")
            continue

        for dlg in dialogues:
            dlg_id = dlg.get("id", "")
            for turn in dlg.get("turns", []):
                sentences = turn.get("text_mt", [])
                emotions = turn.get("Emotion", [])
                engagement = turn.get("Engagement", [])
                correctness = turn.get("Correctness", [])
                floor = turn.get("floor", "")

                for idx, sent in enumerate(sentences):
                    sent = str(sent).strip()
                    if len(sent) < 5 or sent.startswith("#") and len(sent) < 10:
                        continue
                    rows.append({
                        "dialogue_id": dlg_id,
                        "text": sent,
                        "floor": floor,  # "inbound" (customer) or "outbound" (agent)
                        "emotion": int(emotions[idx]) if idx < len(emotions) else None,
                        "engagement": int(engagement[idx]) if idx < len(engagement) else None,
                        "correctness": int(correctness[idx]) if idx < len(correctness) else None,
                        "source": "maia_dqe",
                    })

    df = pd.DataFrame(rows)
    print(f"  Loaded {len(df):,} sentence-level rows from {len(df['dialogue_id'].unique()):,} dialogues")
    print(f"  Emotion distribution:\n{df['emotion'].value_counts().head(10)}")

    dest = PROCESSED / "maia_emotion.csv"
    df.to_csv(dest, index=False)
    print(f"  Saved → {dest}")
    return dest


# ─────────────────────────────────────────────────────────────────────────────
# EDA summary across all prepared datasets
# ─────────────────────────────────────────────────────────────────────────────
def run_cross_corpus_eda(out_dir: Path) -> None:
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns

    sns.set_theme(style="whitegrid", context="talk")
    out_dir.mkdir(parents=True, exist_ok=True)

    files = {
        "twitter_support": PROCESSED / "twitter_support_sample.csv",
        "news_summary": PROCESSED / "news_summary_sample.csv",
        "amazon_csat": PROCESSED / "amazon_csat.csv",
        "maia_dqe": PROCESSED / "maia_emotion.csv",
        "tickets_labeled": RAW / "tickets_labeled.csv",
    }

    counts = {}
    for name, path in files.items():
        if path.is_file():
            df = pd.read_csv(path, usecols=["text"])
            counts[name] = len(df)
            print(f"  {name}: {len(df):,} rows")
        else:
            print(f"  {name}: NOT FOUND")

    if counts:
        fig, ax = plt.subplots(figsize=(9, 4))
        labels, vals = zip(*sorted(counts.items(), key=lambda x: -x[1]))
        sns.barplot(x=list(vals), y=list(labels), palette="mako", ax=ax)
        ax.set_title("Multi-corpus row counts (after processing)")
        ax.set_xlabel("Rows")
        plt.tight_layout()
        dest = out_dir / "corpus_sizes.png"
        plt.savefig(dest, dpi=150)
        plt.close()
        print(f"\nSaved cross-corpus size chart → {dest}")


# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare all research datasets.")
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=AVAILABLE + ["all"],
        default=["all"],
        help="Which datasets to prepare (default: all)",
    )
    parser.add_argument("--eda", action="store_true", default=True, help="Run cross-corpus EDA after preparing")
    args = parser.parse_args()

    datasets = AVAILABLE if "all" in args.datasets else args.datasets

    try:
        import pandas  # noqa: F401
    except ImportError as exc:
        raise SystemExit("Install dependencies: pip install -e '.[eda]'") from exc

    PROCESSED.mkdir(parents=True, exist_ok=True)

    errors = []
    for ds in datasets:
        try:
            if ds == "twitter":
                prepare_twitter()
            elif ds == "news":
                prepare_news()
            elif ds == "amazon":
                prepare_amazon()
            elif ds == "maia":
                prepare_maia()
        except FileNotFoundError as e:
            print(f"\n[WARNING] Skipping {ds}: {e}")
            errors.append(ds)
        except Exception as e:
            print(f"\n[ERROR] {ds}: {e}")
            errors.append(ds)

    if args.eda:
        print("\n=== Cross-corpus EDA ===")
        run_cross_corpus_eda(ROOT / "artifacts" / "eda")

    if errors:
        print(f"\n[INCOMPLETE] Could not prepare: {errors}")
    else:
        print("\n✅ All datasets prepared successfully.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Build the single combined training dataset from all 5 sources.

Output: data/processed/combined_dataset.csv

Unified schema:
  text          - customer/review/support text
  category      - our 5-class triage label (billing / authentication /
                  technical_bug / feature_request / general_inquiry)
  sentiment     - positive / negative / neutral
  source        - which corpus the row came from
  has_summary   - 1 if a reference summary is available (news rows)
  summary       - reference summary text (news rows only, else "")

Usage:
  python scripts/build_combined_dataset.py
  python scripts/build_combined_dataset.py --max-per-source 8000
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"


# ── Keyword-based category guesser (for Twitter + Amazon which lack labels) ─
_BILLING = re.compile(
    r"charge|refund|bill|payment|invoice|price|fee|overpay|money|cost|subscription|credit", re.I
)
_AUTH    = re.compile(r"login|log.?in|password|account|sign.?in|auth|access|hack|secur", re.I)
_BUG     = re.compile(r"crash|error|bug|broken|not.?work|fail|glitch|freeze|slow|down", re.I)
_FEAT    = re.compile(r"add|feature|wish|want|improve|suggest|request|dark.?mode|option|should", re.I)


def guess_category(text: str) -> str:
    t = str(text)
    if _BILLING.search(t): return "billing"
    if _AUTH.search(t):    return "authentication"
    if _BUG.search(t):     return "technical_bug"
    if _FEAT.search(t):    return "feature_request"
    return "general_inquiry"


# ── Sentiment from CSAT (1-2=neg, 3=neutral, 4-5=pos) ──────────────────────
def csat_to_sentiment(score) -> str:
    try:
        s = int(float(score))
        if s <= 2: return "negative"
        if s == 3: return "neutral"
        return "positive"
    except Exception:
        return "neutral"


# ── Emotion int → coarse sentiment (MAIA emotion label scheme) ───────────────
# MAIA emotion codes: 0=anger,1=disgust,2=neutral,3=fear,4=joy,5=sad,6=surprise,7=other
_MAIA_POSITIVE = {4}          # joy
_MAIA_NEGATIVE = {0, 1, 3, 5} # anger, disgust, fear, sadness


def emotion_to_sentiment(code) -> str:
    try:
        c = int(code)
        if c in _MAIA_POSITIVE: return "positive"
        if c in _MAIA_NEGATIVE: return "negative"
        return "neutral"
    except Exception:
        return "neutral"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build unified combined_dataset.csv")
    parser.add_argument("--max-per-source", type=int, default=10_000,
                        help="Max rows to take from each source (default 10 000)")
    parser.add_argument("--out", type=Path,
                        default=PROCESSED / "combined_dataset.csv")
    args = parser.parse_args()

    try:
        import pandas as pd
    except ImportError as exc:
        raise SystemExit("pip install pandas") from exc

    frames = []

    # ── 1. Tickets (Dataset 5) — has gold category labels ──────────────────
    p = RAW / "tickets_labeled.csv"
    if p.is_file():
        df = pd.read_csv(p)
        df = df.sample(min(args.max_per_source, len(df)), random_state=42)
        df["sentiment"] = df["csat_score"].apply(csat_to_sentiment)
        frames.append(pd.DataFrame({
            "text":        df["text"].str.strip(),
            "category":    df["category"],
            "sentiment":   df["sentiment"],
            "source":      "tickets",
            "has_summary": 0,
            "summary":     "",
        }))
        print(f"[tickets]  {len(df):>7,} rows")
    else:
        print("[tickets]  MISSING — run scripts/prepare_real_dataset.py")

    # ── 2. Twitter (Dataset 1) — infer category from text ──────────────────
    p = PROCESSED / "twitter_support_sample.csv"
    if p.is_file():
        df = pd.read_csv(p)
        df = df.sample(min(args.max_per_source, len(df)), random_state=42)
        df["category"] = df["text"].apply(guess_category)
        frames.append(pd.DataFrame({
            "text":        df["text"].str.strip(),
            "category":    df["category"],
            "sentiment":   df["sentiment_label"],
            "source":      "twitter",
            "has_summary": 0,
            "summary":     "",
        }))
        print(f"[twitter]  {len(df):>7,} rows")
    else:
        print("[twitter]  MISSING — run scripts/prepare_all_datasets.py")

    # ── 3. Amazon (Dataset 4) — reviews, sentiment proxy ───────────────────
    p = PROCESSED / "amazon_csat.csv"
    if p.is_file():
        df = pd.read_csv(p)
        df = df.sample(min(args.max_per_source, len(df)), random_state=42)
        df["category"] = df["text"].apply(guess_category)
        frames.append(pd.DataFrame({
            "text":        df["text"].str.strip(),
            "category":    df["category"],
            "sentiment":   df["sentiment_label"],
            "source":      "amazon",
            "has_summary": 0,
            "summary":     "",
        }))
        print(f"[amazon]   {len(df):>7,} rows")
    else:
        print("[amazon]   MISSING — run scripts/prepare_all_datasets.py")

    # ── 4. MAIA-DQE (Dataset 6) — emotion-annotated support sentences ───────
    p = PROCESSED / "maia_emotion.csv"
    if p.is_file():
        df = pd.read_csv(p)
        # Only keep customer (inbound) turns for triage-relevance
        df = df[df["floor"] == "inbound"].copy() if "floor" in df.columns else df
        df = df[df["text"].str.len() >= 10].copy()
        df = df.sample(min(args.max_per_source, len(df)), random_state=42)
        df["category"] = df["text"].apply(guess_category)
        df["sentiment"] = df["emotion"].apply(emotion_to_sentiment)
        frames.append(pd.DataFrame({
            "text":        df["text"].str.strip(),
            "category":    df["category"],
            "sentiment":   df["sentiment"],
            "source":      "maia",
            "has_summary": 0,
            "summary":     "",
        }))
        print(f"[maia]     {len(df):>7,} rows")
    else:
        print("[maia]     MISSING — run scripts/prepare_all_datasets.py")

    # ── 5. News Summary (Dataset 3) — summarization pairs ──────────────────
    p = PROCESSED / "news_summary_sample.csv"
    if p.is_file():
        df = pd.read_csv(p)
        df = df.sample(min(args.max_per_source, len(df)), random_state=42)
        df["category"] = df["text"].apply(guess_category)
        frames.append(pd.DataFrame({
            "text":        df["text"].str.strip(),
            "category":    df["category"],
            "sentiment":   "neutral",          # news articles are neutral baseline
            "source":      "news",
            "has_summary": 1,
            "summary":     df["summary"].str.strip(),
        }))
        print(f"[news]     {len(df):>7,} rows")
    else:
        print("[news]     MISSING — run scripts/prepare_all_datasets.py")

    if not frames:
        raise SystemExit("No datasets found. Run prepare_real_dataset.py and prepare_all_datasets.py first.")

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.dropna(subset=["text", "category"])
    combined = combined[combined["text"].str.len() >= 10].copy()
    combined = combined.sample(frac=1, random_state=42).reset_index(drop=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(args.out, index=False)

    print(f"\n{'='*50}")
    print(f"Combined dataset: {len(combined):,} rows → {args.out}")
    print(f"\nBy source:\n{combined['source'].value_counts().to_string()}")
    print(f"\nBy category:\n{combined['category'].value_counts().to_string()}")
    print(f"\nBy sentiment:\n{combined['sentiment'].value_counts().to_string()}")
    print(f"\nWith reference summary: {combined['has_summary'].sum():,} rows")


if __name__ == "__main__":
    main()

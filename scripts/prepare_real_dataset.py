#!/usr/bin/env python3
"""
Prepare the real Customer Support Data (Kaggle: akashbommidi/customer-support-data)
for use as triage training + evaluation data.

Maps the dataset's e-commerce categories → our 5-class taxonomy:
  billing           ← Refund Related, Payments related, Offers & Cashback
  technical_bug     ← App/website, Shopzilla Related (tech issues)
  feature_request   ← Feedback, Onboarding related
  general_inquiry   ← Product Queries, Others
  authentication    ← Fraudulent User sub-cat (mapped from Order Related/Returns)
  (Order Related, Returns, Cancellation → general_inquiry or billing by sub-cat)

Usage:
  python scripts/prepare_real_dataset.py
  python scripts/prepare_real_dataset.py --input data/raw/Customer_support_data.csv --out data/raw/tickets_labeled.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------------
# Category mapping: (category, sub-category) -> our taxonomy label
# Priority: sub-category rules first, then category fallback
# ---------------------------------------------------------------------------
SUBCATEGORY_MAP: dict[str, str] = {
    # Billing / money
    "refund enquiry": "billing",
    "refund related issues": "billing",
    "invoice request": "billing",
    "online payment issues": "billing",
    "payment related": "billing",
    # Authentication / security
    "fraudulent user": "authentication",
    "account related": "authentication",
    "login": "authentication",
    # Technical
    "app related": "technical_bug",
    "website related": "technical_bug",
    "app/website": "technical_bug",
    "technical": "technical_bug",
    # Feature / feedback
    "product specific information": "feature_request",
    "unprofessional behaviour": "feature_request",
    "general enquiry": "general_inquiry",
    # Shipping / order → general
    "order status enquiry": "general_inquiry",
    "unable to track": "general_inquiry",
    "priority delivery": "general_inquiry",
    "delayed": "general_inquiry",
    "reverse pickup enquiry": "general_inquiry",
    "return request": "billing",       # return = money back
    "seller cancelled order": "general_inquiry",
    "installation/demo": "general_inquiry",
    "service centres related": "general_inquiry",
    "missing": "general_inquiry",
    "wrong": "general_inquiry",
    "exchange / replacement": "billing",
    "not needed": "billing",
    "life insurance": "billing",
}

CATEGORY_MAP: dict[str, str] = {
    "refund related": "billing",
    "payments related": "billing",
    "offers & cashback": "billing",
    "returns": "billing",
    "cancellation": "billing",
    "app/website": "technical_bug",
    "shopzilla related": "technical_bug",
    "feedback": "feature_request",
    "onboarding related": "feature_request",
    "product queries": "general_inquiry",
    "others": "general_inquiry",
    "order related": "general_inquiry",
}


def map_label(category: str, sub_category: str) -> str:
    sub_key = str(sub_category).strip().lower()
    cat_key = str(category).strip().lower()
    if sub_key in SUBCATEGORY_MAP:
        return SUBCATEGORY_MAP[sub_key]
    if cat_key in CATEGORY_MAP:
        return CATEGORY_MAP[cat_key]
    return "general_inquiry"  # safe fallback


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare real customer support data for triage training.")
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT / "data/raw/Customer_support_data.csv",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "data/raw/tickets_labeled.csv",
    )
    parser.add_argument(
        "--min-text-len",
        type=int,
        default=15,
        help="Minimum character length for Customer Remarks (drop shorter rows)",
    )
    parser.add_argument(
        "--max-per-class",
        type=int,
        default=4000,
        help="Cap per class to keep dataset balanced for demo training",
    )
    args = parser.parse_args()

    try:
        import pandas as pd
    except ImportError as exc:
        raise SystemExit("Install pandas: pip install -e '.[eda]'") from exc

    if not args.input.is_file():
        raise SystemExit(f"Input file not found: {args.input}\nRun: python data/download_kaggle.py --dataset akashbommidi/customer-support-data")

    print(f"Loading {args.input} …")
    df = pd.read_csv(args.input, encoding="latin-1")
    print(f"  Raw rows: {len(df):,}")

    # Keep only rows with usable text
    df = df[df["Customer Remarks"].notna()].copy()
    df = df[df["Customer Remarks"].str.len() >= args.min_text_len].copy()
    print(f"  After text filter: {len(df):,}")

    # Map to our taxonomy
    df["category_mapped"] = df.apply(
        lambda r: map_label(r["category"], r.get("Sub-category", "")), axis=1
    )

    print("\nMapped class distribution (before capping):")
    print(df["category_mapped"].value_counts())

    # Cap per class so no class dominates (handles imbalance for demo training)
    if args.max_per_class:
        frames = []
        for cat, grp in df.groupby("category_mapped"):
            frames.append(grp.sample(min(len(grp), args.max_per_class), random_state=42))
        df = pd.concat(frames, ignore_index=True)
        print(f"\nAfter capping at {args.max_per_class}/class:")
        print(df["category_mapped"].value_counts())

    # Build output CSV matching our training contract: text, category
    out_df = pd.DataFrame({
        "text": df["Customer Remarks"].str.strip(),
        "category": df["category_mapped"],
        # extras — kept for EDA but ignored by train scripts
        "original_category": df["category"],
        "sub_category": df["Sub-category"],
        "csat_score": df["CSAT Score"],
        "channel": df["channel_name"],
    })

    args.out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out, index=False, encoding="utf-8")
    print(f"\nSaved {len(out_df):,} rows → {args.out.resolve()}")

    # Quick sanity: show a few rows per class
    print("\nSample rows per class:")
    for cat, grp in out_df.groupby("category"):
        sample = grp["text"].iloc[0][:100]
        print(f"  [{cat}] {sample!r}")


if __name__ == "__main__":
    main()

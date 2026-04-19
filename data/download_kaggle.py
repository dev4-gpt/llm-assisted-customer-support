#!/usr/bin/env python3
"""
Optional Kaggle dataset download via kagglehub.

Requires:
  pip install -e ".[data]"
  Kaggle API credentials (~/.kaggle/kaggle.json) per Kaggle documentation.

Usage:
  python data/download_kaggle.py --dataset thoughtvector/customer-support-on-twitter
  python data/download_kaggle.py --dataset akashbommidi/customer-support-data
"""
from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Download a Kaggle dataset via kagglehub.")
    parser.add_argument(
        "--dataset",
        required=True,
        help="Kaggle dataset slug, e.g. thoughtvector/customer-support-on-twitter",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/raw"),
        help="Directory to mirror files into (default: data/raw/<slug>)",
    )
    args = parser.parse_args()

    try:
        import kagglehub
    except ImportError as exc:
        raise SystemExit(
            "kagglehub is not installed. Install with: pip install -e \".[data]\""
        ) from exc

    path = kagglehub.dataset_download(args.dataset)
    print(f"Downloaded dataset to: {path}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    slug_dir = args.output_dir / args.dataset.replace("/", "__")
    slug_dir.mkdir(parents=True, exist_ok=True)
    print(f"Copy or symlink artifacts from the path above into: {slug_dir.resolve()}")


if __name__ == "__main__":
    main()

"""
Train / validation / test index splits with optional stratification.

Uses only the Python standard library so notebooks and CI can import without sklearn.
For large-scale stratified splitting on imbalanced data, prefer sklearn's train_test_split
with your dataframe directly.
"""

from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Hashable
from typing import TypeVar

T = TypeVar("T", bound=Hashable)


def stratified_split_indices(
    labels: list[T],
    *,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> tuple[list[int], list[int], list[int]]:
    """
    Return index lists for train / val / test with per-class stratification.

    Each class is split independently by ratio, so very small classes may end up with
    empty val or test buckets (caller should merge or use minimum counts in production).
    """
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"Ratios must sum to 1.0, got {total}")

    rng = random.Random(seed)  # noqa: S311 — reproducible split indices, not cryptography
    by_label: dict[T, list[int]] = defaultdict(list)
    for idx, lab in enumerate(labels):
        by_label[lab].append(idx)

    train_idx: list[int] = []
    val_idx: list[int] = []
    test_idx: list[int] = []

    for indices in by_label.values():
        rng.shuffle(indices)
        n = len(indices)
        n_train = int(round(train_ratio * n))
        n_val = int(round(val_ratio * n))
        # Ensure every index is assigned once
        n_train = max(0, min(n_train, n))
        n_val = max(0, min(n_val, n - n_train))
        n_test = n - n_train - n_val

        train_idx.extend(indices[:n_train])
        val_idx.extend(indices[n_train : n_train + n_val])
        test_idx.extend(indices[n_train + n_val : n_train + n_val + n_test])

    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)
    return train_idx, val_idx, test_idx

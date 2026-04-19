"""
Correlate LLM-as-judge scores (or model probabilities) with proxy satisfaction signals.

Uses only the standard library. For heavy-duty analysis, export CSV and use pandas/scipy.
"""

from __future__ import annotations


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def pearson_r(x: list[float], y: list[float]) -> float:
    """Pearson correlation coefficient; returns 0.0 if undefined (constant input)."""
    if len(x) != len(y):
        raise ValueError("x and y must have the same length")
    n = len(x)
    if n < 2:
        return 0.0

    mx, my = _mean(x), _mean(y)
    dx = [xi - mx for xi in x]
    dy = [yi - my for yi in y]
    var_x = sum(d * d for d in dx)
    var_y = sum(d * d for d in dy)
    if var_x == 0 or var_y == 0:
        return 0.0
    cov = sum(a * b for a, b in zip(dx, dy, strict=True))
    return cov / (var_x**0.5 * var_y**0.5)


def spearman_rho(x: list[float], y: list[float]) -> float:
    """Spearman rank correlation via ranking with average ties, then Pearson on ranks."""
    if len(x) != len(y):
        raise ValueError("x and y must have the same length")
    n = len(x)
    if n < 2:
        return 0.0

    def ranks(values: list[float]) -> list[float]:
        indexed = sorted(enumerate(values), key=lambda t: t[1])
        ranks_out = [0.0] * len(values)
        i = 0
        while i < len(indexed):
            j = i
            while j + 1 < len(indexed) and indexed[j + 1][1] == indexed[i][1]:
                j += 1
            avg_rank = (i + j) / 2.0 + 1.0  # 1-based average rank
            for k in range(i, j + 1):
                orig_idx = indexed[k][0]
                ranks_out[orig_idx] = avg_rank
            i = j + 1
        return ranks_out

    return pearson_r(ranks(x), ranks(y))


def judge_proxy_correlation(
    judge_scores: list[float],
    proxy_scores: list[float],
) -> dict[str, float]:
    """
    Compare continuous judge scores (e.g. LLM quality 0–1) with proxies (CSAT, stars / 5).

    Returns Pearson and Spearman coefficients for reporting agreement trends.
    """
    return {
        "pearson_r": pearson_r(judge_scores, proxy_scores),
        "spearman_rho": spearman_rho(judge_scores, proxy_scores),
        "n": float(len(judge_scores)),
    }

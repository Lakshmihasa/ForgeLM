"""Bootstrap confidence intervals — is the improvement real, or noise?

With ~1,000 dev examples a 1-2 point execution-accuracy difference can be
sampling noise. The gap between the fine-tuned model and the fair base baseline
is the headline claim, so it needs an interval, not just a point estimate.

The gap uses a PAIRED bootstrap: both systems are scored on the same examples,
so we resample example ids (with replacement) and recompute the difference on
each resample. Pairing removes the per-example difficulty variance that would
otherwise inflate the interval, giving a tighter, more honest CI on the delta.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["CI", "bootstrap_accuracy_ci", "paired_gap_ci"]


@dataclass
class CI:
    estimate: float
    low: float
    high: float
    n: int

    def __str__(self) -> str:
        return f"{self.estimate:.4f}  [{self.low:.4f}, {self.high:.4f}]  (n={self.n})"


def bootstrap_accuracy_ci(
    correct: list[bool],
    *,
    n_boot: int = 10000,
    alpha: float = 0.05,
    seed: int = 0,
) -> CI:
    """Percentile bootstrap CI for a single system's execution accuracy."""
    x = np.asarray(correct, dtype=float)
    n = len(x)
    if n == 0:
        return CI(0.0, 0.0, 0.0, 0)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    means = x[idx].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return CI(float(x.mean()), float(lo), float(hi), n)


def paired_gap_ci(
    a: dict[str, bool],
    b: dict[str, bool],
    *,
    n_boot: int = 10000,
    alpha: float = 0.05,
    seed: int = 0,
) -> CI:
    """Paired bootstrap CI for (accuracy_b - accuracy_a) on shared examples.

    `a` and `b` are {example_id: correct} (e.g. ExecutionReport.by_id()). Only
    ids present in both are used, so the comparison is strictly like-for-like.
    A positive interval that excludes 0 means b beats a beyond noise.
    """
    ids = sorted(set(a) & set(b))
    if not ids:
        return CI(0.0, 0.0, 0.0, 0)

    av = np.array([a[i] for i in ids], dtype=float)
    bv = np.array([b[i] for i in ids], dtype=float)
    diff = bv - av  # per-example paired difference
    n = len(ids)

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    gaps = diff[idx].mean(axis=1)
    lo, hi = np.quantile(gaps, [alpha / 2, 1 - alpha / 2])
    return CI(float(diff.mean()), float(lo), float(hi), n)
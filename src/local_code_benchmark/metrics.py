"""Metric implementations used by the benchmark."""

from __future__ import annotations

import math
import random
from collections.abc import Iterable


def pass_at_k(sample_results: Iterable[Iterable[bool]], k: int) -> float:
    """Return the unbiased HumanEval pass@k estimate averaged over tasks.

    Each inner iterable contains the correctness results for all samples from
    one task. Tasks with fewer than ``k`` samples are invalid because pass@k
    is not defined for them.
    """
    if k < 1:
        raise ValueError("k must be at least 1")

    estimates = task_pass_at_k(sample_results, k)
    return sum(estimates) / len(estimates)


def task_pass_at_k(sample_results: Iterable[Iterable[bool]], k: int) -> list[float]:
    """Return one unbiased pass@k estimate per task."""
    if k < 1:
        raise ValueError("k must be at least 1")

    estimates: list[float] = []
    for values in sample_results:
        results = list(values)
        n = len(results)
        if n < k:
            raise ValueError(f"pass@{k} requires at least {k} samples per task")
        c = sum(results)
        estimate = 1.0 if n - c < k else 1.0 - math.comb(n - c, k) / math.comb(n, k)
        estimates.append(estimate)

    if not estimates:
        raise ValueError("at least one task is required")
    return estimates


def bootstrap_mean_ci(
    values: Iterable[float], *, confidence: float = 0.95, replicates: int = 10_000, seed: int = 0
) -> tuple[float, float]:
    """Percentile bootstrap confidence interval for a mean over tasks."""
    observations = list(values)
    if not observations:
        raise ValueError("at least one observation is required")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be in (0, 1)")
    if replicates < 1:
        raise ValueError("replicates must be at least 1")

    rng = random.Random(seed)
    n = len(observations)
    means = sorted(
        sum(observations[rng.randrange(n)] for _ in range(n)) / n for _ in range(replicates)
    )
    tail = (1.0 - confidence) / 2.0
    low_index = max(0, math.floor(tail * replicates))
    high_index = min(replicates - 1, math.ceil((1.0 - tail) * replicates) - 1)
    return means[low_index], means[high_index]

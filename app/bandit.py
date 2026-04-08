"""
bandit.py - Multi-Armed Bandit algorithms.

Thompson Sampling (primary)
───────────────────────────
For each arm i, model clicks ~ Beta(α_i, β_i) where:
    α_i = total_clicks_i + 1      (Bayesian prior: 1 success)
    β_i = total_failures_i + 1    (Bayesian prior: 1 failure)

We draw N samples from each Beta distribution and compute
the fraction of draws where each arm wins → allocation %.

Upper Confidence Bound – UCB1 (alternative)
────────────────────────────────────────────
score_i = ctr_i + sqrt(2 * ln(total_impressions) / impressions_i)
Allocations are proportional to UCB scores.
"""

import numpy as np
from typing import List, Dict


def thompson_sampling(
    arms: List[Dict],
    n_samples: int = 50_000,
    seed: int | None = None,
) -> List[float]:
    """
    Parameters
    ----------
    arms : list of dicts with keys 'total_clicks' and 'total_impressions'
    n_samples : Monte-Carlo draws per arm
    seed : optional RNG seed for reproducibility

    Returns
    -------
    List of allocation percentages (sum = 100.0) in the same order as `arms`.
    """
    rng = np.random.default_rng(seed)
    n_arms = len(arms)

    samples = np.zeros((n_samples, n_arms))
    for i, arm in enumerate(arms):
        alpha = arm["total_clicks"] + 1          # successes + prior
        beta  = max(arm["total_impressions"] - arm["total_clicks"], 0) + 1  # failures + prior
        samples[:, i] = rng.beta(alpha, beta, size=n_samples)

    winners = samples.argmax(axis=1)             # which arm won each draw
    counts = np.bincount(winners, minlength=n_arms)
    percentages = (counts / n_samples) * 100.0

    return percentages.tolist()


def ucb1(arms: List[Dict]) -> List[float]:
    """
    UCB1 allocations.  Arms with zero impressions get 100/n_arms each.
    """
    n_arms = len(arms)
    total_impressions = sum(a["total_impressions"] for a in arms)

    if total_impressions == 0:
        return [100.0 / n_arms] * n_arms

    scores = []
    for arm in arms:
        imp = arm["total_impressions"]
        clicks = arm["total_clicks"]
        ctr = clicks / imp if imp > 0 else 0.0
        exploration = (
            np.sqrt(2 * np.log(total_impressions) / imp) if imp > 0 else float("inf")
        )
        scores.append(ctr + exploration)

    inf_count = sum(1 for s in scores if s == float("inf"))
    if inf_count:
        return [100.0 / inf_count if s == float("inf") else 0.0 for s in scores]

    total_score = sum(scores)
    percentages = [(s / total_score) * 100.0 for s in scores]
    return percentages


def compute_allocations(arms: List[Dict], algorithm: str = "thompson_sampling") -> List[float]:
    """
    Dispatch to the chosen algorithm.

    Parameters
    ----------
    arms : list of dicts – each needs 'total_clicks' and 'total_impressions'
    algorithm : "thompson_sampling" | "ucb1"
    """
    if algorithm == "ucb1":
        return ucb1(arms)
    return thompson_sampling(arms)

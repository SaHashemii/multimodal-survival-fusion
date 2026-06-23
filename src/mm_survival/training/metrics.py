"""Evaluation metrics for survival prediction."""

from __future__ import annotations

import math

import numpy as np


def concordance_index(risk: np.ndarray, time: np.ndarray, event: np.ndarray) -> float:
    """Compute Harrell-style concordance index for log-risk scores."""
    concordant = 0.0
    permissible = 0.0
    for i in range(len(risk)):
        for j in range(len(risk)):
            if time[i] < time[j] and event[i] == 1:
                permissible += 1
                if risk[i] > risk[j]:
                    concordant += 1
                elif risk[i] == risk[j]:
                    concordant += 0.5
    return float(concordant / permissible) if permissible else 0.5


def bootstrap_c_index(
    risk: np.ndarray,
    time: np.ndarray,
    event: np.ndarray,
    seed: int,
    n_boot: int = 500,
) -> tuple[float, float, float]:
    """Return point estimate and percentile bootstrap CI for C-index."""
    point = concordance_index(risk, time, event)
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(risk), len(risk))
        if len(np.unique(event[idx])) < 2:
            continue
        values.append(concordance_index(risk[idx], time[idx], event[idx]))
    if not values:
        return point, math.nan, math.nan
    return point, float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))

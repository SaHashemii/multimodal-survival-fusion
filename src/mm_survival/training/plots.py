"""
Plotting utilities for survival model outputs
=============================================

Creates Kaplan-Meier curves from saved model risk scores.

Pipeline
--------
  pooled test risk scores → median risk split → low/high risk KM curves
  log-rank test → PNG plot + metadata dictionary

Design rationale
----------------
* In cross-validation runs, the input risk table is usually the pooled
  out-of-fold test predictions from all folds.
* The median log-risk split is used as a simple model-derived high/low risk
  grouping for visualization.
* The plot is descriptive; C-index remains the primary ranking metric.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def kaplan_meier_curve(time: np.ndarray, event: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return step-function coordinates for a Kaplan-Meier survival curve."""
    event_times = np.sort(np.unique(time[event == 1]))
    times = [0.0]
    survival = [1.0]
    current = 1.0
    for t in event_times:

        # Kaplan-Meier update: survival is multiplied by the probability of not
        # having an event at this event time among patients still at risk.
        at_risk = int((time >= t).sum())
        events_at_t = int(((time == t) & (event == 1)).sum())
        if at_risk <= 0:
            continue
        current *= 1.0 - (events_at_t / at_risk)
        times.append(float(t))
        survival.append(float(current))
    return np.array(times, dtype=float), np.array(survival, dtype=float)


def logrank_test(time: np.ndarray, event: np.ndarray, high_risk: np.ndarray) -> dict[str, float]:
    """Compute a two-group log-rank test for high- vs low-risk groups."""
    event_times = np.sort(np.unique(time[event == 1]))
    observed_high = expected_high = variance_high = 0.0
    for t in event_times:

        # Compare observed events in the high-risk group to the number expected
        # under equal survival between high- and low-risk groups.
        at_risk = time >= t
        at_risk_high = at_risk & high_risk
        n = int(at_risk.sum())
        n_high = int(at_risk_high.sum())
        if n <= 1 or n_high == 0 or n_high == n:
            continue
        events_at_t = (time == t) & (event == 1)
        d = int(events_at_t.sum())
        d_high = int((events_at_t & high_risk).sum())
        observed_high += d_high
        expected_high += d * (n_high / n)
        variance_high += (n_high * (n - n_high) * d * (n - d)) / (n * n * (n - 1))
    if variance_high <= 0:
        return {
            "observed_events_high": observed_high,
            "expected_events_high": expected_high,
            "logrank_chi_square": math.nan,
            "logrank_p_value": math.nan,
        }
    chi_square = ((observed_high - expected_high) ** 2) / variance_high
    return {
        "observed_events_high": observed_high,
        "expected_events_high": expected_high,
        "logrank_chi_square": chi_square,
        "logrank_p_value": math.erfc(math.sqrt(chi_square / 2.0)),
    }


def write_kaplan_meier_plot(
    risk_scores: pd.DataFrame,
    output_path: str | Path,
    *,
    title: str,
) -> dict[str, Any]:
    """Write a median-risk high/low Kaplan-Meier plot and return log-rank metadata."""
    required = {"log_risk", "Event", "Time"}
    missing = required - set(risk_scores.columns)
    if missing:
        raise ValueError(f"Risk scores are missing columns: {sorted(missing)}")

    risk = risk_scores["log_risk"].to_numpy(float)
    time = risk_scores["Time"].to_numpy(float)
    event = risk_scores["Event"].to_numpy(int)
    median_risk = float(np.median(risk))

    # Patients at or above the median predicted log-risk are assigned to the
    # high-risk curve; the remaining patients define the low-risk curve.
    high_risk = risk >= median_risk

    stats = {
        "n_patients": int(len(risk_scores)),
        "median_risk": median_risk,
        "high_risk_n": int(high_risk.sum()),
        "low_risk_n": int((~high_risk).sum()),
        **logrank_test(time, event, high_risk),
    }

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        stats["plot_written"] = False
        stats["plot_error"] = f"matplotlib is not installed: {exc}"
        return stats

    low_t, low_s = kaplan_meier_curve(time[~high_risk], event[~high_risk])
    high_t, high_s = kaplan_meier_curve(time[high_risk], event[high_risk])
    p_value = stats["logrank_p_value"]
    p_text = f"p={p_value:.3g}" if np.isfinite(p_value) else "p=NA"

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.step(low_t, low_s, where="post", label=f"Low risk (n={stats['low_risk_n']})", linewidth=2)
    ax.step(high_t, high_s, where="post", label=f"High risk (n={stats['high_risk_n']})", linewidth=2)
    ax.set_xlabel("Time")
    ax.set_ylabel("Survival probability")
    ax.set_ylim(0.0, 1.05)
    ax.set_title(f"{title} | log-rank {p_text}")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    stats["plot_written"] = True
    stats["plot_path"] = str(output_path)
    return stats
